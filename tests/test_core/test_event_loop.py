# Copyright 2026 Zachary Brooks
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import json
from typing import Any

import pytest

from safe_agent.core import EventLoop, LLMResponse, Session, ToolCall
from safe_agent.core.event_loop import _sanitize_messages
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry


class _FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str, str | None]] = []

    async def dispatch(
        self,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        tool_call_id: str | None = None,
    ) -> ToolResult[Any]:
        self.calls.append((tool_name, params, session_id, tool_call_id))
        return ToolResult(success=True, data={"echo": params})


class _FailingDispatcher(_FakeDispatcher):
    async def dispatch(
        self,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        tool_call_id: str | None = None,
    ) -> ToolResult[Any]:
        self.calls.append((tool_name, params, session_id, tool_call_id))
        msg = f"boom for {tool_name}"
        raise RuntimeError(msg)


class _FakeLLM:
    def __init__(self, responses: list[LLMResponse], delay: float = 0.0) -> None:
        self._responses = responses
        self._delay = delay
        self.calls: list[tuple[list[dict], list[dict]]] = []
        self.active_calls = 0
        self.max_concurrent_calls = 0

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        self.calls.append((list(messages), list(tools)))
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return self._responses.pop(0)
        finally:
            self.active_calls -= 1


@pytest.fixture
def registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(_StaticToolModule())
    return registry


class _StaticToolModule(BaseModule):
    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            namespace="demo",
            description="Demo tools",
            tools=[
                ToolDescriptor(
                    name="demo:echo",
                    description="Echo params",
                    parameters={"type": "object"},
                    action="demo:Echo",
                )
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        return ToolResult(success=True, data=params)


def test_content_and_tool_calls_dispatched_both(registry: ModuleRegistry) -> None:
    """When LLM returns both content and tool_calls, both are processed."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(
                content="Let me look that up for you.",
                tool_calls=[ToolCall(name="demo:echo", params={"query": "hello"})],
            ),
            LLMResponse(content="Here is the result."),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "greet"))

    # Result should come from second LLM response after tool execution
    assert result == "Here is the result."
    # Dispatcher should have been called for the tool
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0] == ("demo:echo", {"query": "hello"}, "session-1", None)

    # Verify message history includes both content and tool_calls
    # Message at index 0: user message
    # Message at index 1: assistant content message
    assert session.messages[0] == {"role": "user", "content": "greet"}
    assert session.messages[1] == {
        "role": "assistant",
        "content": "Let me look that up for you.",
    }
    # Message at index 2: assistant tool_calls message
    assert session.messages[2]["role"] == "assistant"
    assert session.messages[2]["content"] is None
    assert session.messages[2]["tool_calls"][0]["name"] == "demo:echo"
    assert session.messages[2]["tool_calls"][0]["params"] == {"query": "hello"}
    # Message at index 3: tool result
    assert session.messages[3]["role"] == "tool"
    assert session.messages[3]["name"] == "demo:echo"


def test_text_response_immediate(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM([LLMResponse(content="done")])
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "hello"))

    assert result == "done"
    assert dispatcher.calls == []
    assert session.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]


def test_tool_call_then_text(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"value": 1})]),
            LLMResponse(content="tool complete"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "run tool"))

    assert result == "tool complete"
    assert dispatcher.calls == [("demo:echo", {"value": 1}, "session-1", None)]
    assert session.messages[0] == {"role": "user", "content": "run tool"}
    assert session.messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"name": "demo:echo", "params": {"value": 1}, "id": None}],
    }
    assert session.messages[2]["role"] == "tool"
    assert session.messages[2]["name"] == "demo:echo"
    assert isinstance(session.messages[2]["content"], str)
    assert json.loads(session.messages[2]["content"]) == {
        "success": True,
        "data": {"echo": {"value": 1}},
        "error": None,
        "metadata": {},
    }
    assert session.messages[3] == {"role": "assistant", "content": "tool complete"}


def test_tool_dispatch_failure_appends_error_tool_message(
    registry: ModuleRegistry,
) -> None:
    dispatcher = _FailingDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"value": 1})]),
            LLMResponse(content="tool failed cleanly"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "run tool"))

    assert result == "tool failed cleanly"
    assert dispatcher.calls == [("demo:echo", {"value": 1}, "session-1", None)]
    assert session.messages[2] == {
        "role": "tool",
        "name": "demo:echo",
        "content": json.dumps({"error": "Tool execution failed: demo:echo"}),
    }
    assert session.messages[3] == {
        "role": "assistant",
        "content": "tool failed cleanly",
    }


def test_exception_details_not_leaked_to_llm(registry: ModuleRegistry, caplog):
    """Exception details in tool results must not leak to LLM-facing messages."""
    dispatcher = _FailingDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"value": 1})]),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    with caplog.at_level("ERROR", logger="safe_agent.core.event_loop"):
        asyncio.run(event_loop.process_turn(session, "run tool"))

    # The error message visible to the LLM should include tool name
    # but not exception details
    tool_content = json.loads(session.messages[2]["content"])
    assert tool_content == {"error": "Tool execution failed: demo:echo"}
    # Exception message "boom for demo:echo" should NOT be in the LLM-facing error
    assert "boom for" not in tool_content["error"]

    # The actual exception should be logged (not sent to LLM)
    assert "boom for demo:echo" in caplog.text
    # Log should include tool name and call_id
    assert "Tool dispatch failed: demo:echo" in caplog.text


def test_turn_limit_enforced(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 1})]),
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 2})]),
            LLMResponse(content="forced final"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry, max_turns=2)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "loop"))

    assert result == "forced final"
    assert [call[1] for call in dispatcher.calls] == [{"step": 1}, {"step": 2}]
    assert llm.calls[-1][1] == []
    assert session.messages[-1] == {"role": "assistant", "content": "forced final"}


@pytest.mark.asyncio
async def test_session_lock_serializes_concurrent_calls(
    registry: ModuleRegistry,
) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(content="first"),
            LLMResponse(content="second"),
        ],
        delay=0.05,
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    results = await asyncio.gather(
        event_loop.process_turn(session, "one"),
        event_loop.process_turn(session, "two"),
    )

    assert results == ["first", "second"]
    assert llm.max_concurrent_calls == 1
    assert session.messages == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second"},
    ]


def test_release_session_removes_lock(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM([LLMResponse(content="done")])
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "hello"))

    assert session.id in event_loop._session_locks

    event_loop.release_session(session.id)

    assert session.id not in event_loop._session_locks


# ---------------------------------------------------------------------------
# Tool name sanitization tests
# ---------------------------------------------------------------------------


class TestSanitizeMessages:
    """Tests for the _sanitize_messages helper."""

    def test_sanitizes_tool_calls_in_assistant_message(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "filesystem:read_file", "params": {}}],
            }
        ]
        result = _sanitize_messages(messages)
        assert result[0]["tool_calls"][0]["name"] == "filesystem__read_file"

    def test_sanitizes_tool_result_name(self) -> None:
        messages = [{"role": "tool", "name": "filesystem:read_file", "content": "{}"}]
        result = _sanitize_messages(messages)
        assert result[0]["name"] == "filesystem__read_file"

    def test_does_not_mutate_original(self) -> None:
        original_tc = {"name": "filesystem:read_file", "params": {}}
        messages = [{"role": "assistant", "content": None, "tool_calls": [original_tc]}]
        _sanitize_messages(messages)
        assert original_tc["name"] == "filesystem:read_file"

    def test_passthrough_non_tool_messages(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _sanitize_messages(messages)
        assert result == messages


def test_llm_receives_sanitized_tool_names(registry: ModuleRegistry) -> None:
    """EventLoop must send sanitized names to the LLM client."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM([LLMResponse(content="done")])
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "hello"))

    _, tools_sent = llm.calls[0]
    for tool in tools_sent:
        assert ":" not in tool["name"], (
            f"Tool name '{tool['name']}' sent to LLM still contains a colon"
        )
    assert any(t["name"] == "demo__echo" for t in tools_sent)


def test_llm_tool_call_name_restored_before_dispatch(
    registry: ModuleRegistry,
) -> None:
    """LLM returns sanitized name; dispatcher must receive the canonical name."""
    dispatcher = _FakeDispatcher()
    # LLM returns the sanitized name as a provider would
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo__echo", params={"value": 42})]),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "run tool"))

    dispatched_name, _, _, _ = dispatcher.calls[0]
    assert dispatched_name == "demo:echo", (
        f"Expected canonical 'demo:echo' at dispatch, got '{dispatched_name}'"
    )


def test_session_transcript_stores_canonical_names(
    registry: ModuleRegistry,
) -> None:
    """Session history must always use canonical colon-style names."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo__echo", params={"v": 1})]),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "go"))

    assistant_msg = session.messages[1]
    assert assistant_msg["tool_calls"][0]["name"] == "demo:echo"
    tool_result_msg = session.messages[2]
    assert tool_result_msg["name"] == "demo:echo"


def test_sanitized_history_sent_to_llm_on_second_turn(
    registry: ModuleRegistry,
) -> None:
    """On second LLM call, prior tool_calls in history must be re-sanitized."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo__echo", params={"v": 1})]),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "go"))

    # Second LLM call receives the history; tool_calls in it must be sanitized
    second_call_messages, _ = llm.calls[1]
    assistant_history = [
        m
        for m in second_call_messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    for msg in assistant_history:
        for tc in msg["tool_calls"]:
            assert ":" not in tc["name"], (
                f"History tool_call name '{tc['name']}' "
                "not sanitized on second LLM call"
            )


# ---------------------------------------------------------------------------
# Tool call ID provenance tests
# ---------------------------------------------------------------------------


def test_tool_call_id_propagates_to_dispatcher(registry: ModuleRegistry) -> None:
    """ToolCall.id must flow through to the dispatcher."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="demo:echo", params={"value": 1}, id="call-abc123")
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "run tool"))

    assert dispatcher.calls == [("demo:echo", {"value": 1}, "session-1", "call-abc123")]


def test_tool_call_id_in_session_transcript(registry: ModuleRegistry) -> None:
    """Tool result messages must include tool_call_id when present."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="demo:echo", params={"value": 1}, id="call-xyz")
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "run tool"))

    # Assistant message with tool_calls
    assistant_msg = session.messages[1]
    assert assistant_msg["tool_calls"][0]["id"] == "call-xyz"

    # Tool result message
    tool_msg = session.messages[2]
    assert tool_msg["tool_call_id"] == "call-xyz"


def test_multiple_tool_calls_with_distinct_ids(registry: ModuleRegistry) -> None:
    """Multiple tool calls in one turn must preserve distinct IDs."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="demo:echo", params={"n": 1}, id="call-a"),
                    ToolCall(name="demo:echo", params={"n": 2}, id="call-b"),
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    asyncio.run(event_loop.process_turn(session, "run tools"))

    # Dispatcher sees both calls with correct IDs
    assert dispatcher.calls[0] == ("demo:echo", {"n": 1}, "session-1", "call-a")
    assert dispatcher.calls[1] == ("demo:echo", {"n": 2}, "session-1", "call-b")

    # Session transcript has both tool results with correct IDs
    tool_msg_1 = session.messages[2]
    assert tool_msg_1["tool_call_id"] == "call-a"
    tool_msg_2 = session.messages[3]
    assert tool_msg_2["tool_call_id"] == "call-b"
