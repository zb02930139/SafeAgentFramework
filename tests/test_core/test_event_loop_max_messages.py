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

"""Tests for EventLoop max_messages trimming during turn processing.

Issue #133: EventLoop should trim session messages after each turn.
"""

import asyncio
from typing import Any

import pytest

from safe_agent.core import EventLoop, LLMResponse, Session, ToolCall
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


class _FakeLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[list[dict], list[dict]]] = []

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append((list(messages), list(tools)))
        return self._responses.pop(0)


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


@pytest.fixture
def registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(_StaticToolModule())
    return registry


def test_max_messages_trimming_after_content_only_response(
    registry: ModuleRegistry,
) -> None:
    """When a turn completes, messages should be trimmed to max_messages."""
    dispatcher = _FakeDispatcher()
    # Single content response
    llm = _FakeLLM([LLMResponse(content="done")])
    event_loop = EventLoop(dispatcher, llm, registry)

    # Start with messages already at max_messages
    session = Session(id="session-1", max_messages=3)
    # Pre-populate with 2 messages (at the limit)
    session.messages = [
        {"role": "user", "content": "old-msg-1"},
        {"role": "assistant", "content": "old-response-1"},
    ]

    result = asyncio.run(event_loop.process_turn(session, "hello"))

    assert result == "done"
    # Should have 3 messages, but limited by max_messages=3
    # Initial 2 + user message + assistant = 4, but trim keeps last 3
    # The trim keeps: old-msg-1, user "hello", assistant "done"
    assert len(session.messages) <= 3
    assert session.messages[1] == {"role": "user", "content": "hello"}
    assert session.messages[-1] == {"role": "assistant", "content": "done"}


def test_max_messages_trimming_with_tool_calls(registry: ModuleRegistry) -> None:
    """When a turn with tool calls completes, trim messages to max_messages."""
    dispatcher = _FakeDispatcher()
    # Tool call then content response
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"value": 1})]),
            LLMResponse(content="done"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)

    # Set max_messages to 5 to allow the 4 messages from one tool call
    session = Session(id="session-1", max_messages=5)

    result = asyncio.run(event_loop.process_turn(session, "run tool"))

    assert result == "done"
    # Expected messages:
    # 0: user "run tool"
    # 1: assistant with tool_calls
    # 2: tool result
    # 3: assistant "done"
    # Total = 4, which is under max_messages=5, so no trimming
    assert len(session.messages) == 4


def test_max_messages_trimming_excess_messages(registry: ModuleRegistry) -> None:
    """When messages exceed max_messages, oldest messages are trimmed."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM([LLMResponse(content="final")])
    event_loop = EventLoop(dispatcher, llm, registry)

    session = Session(id="session-1", max_messages=3)
    # Pre-populate with 3 messages (already at limit)
    session.messages = [
        {"role": "user", "content": "old-1"},
        {"role": "assistant", "content": "old-response-1"},
        {"role": "user", "content": "old-2"},
    ]

    result = asyncio.run(event_loop.process_turn(session, "new"))

    assert result == "final"
    # Started with 3, added 2 (user + assistant) = 5
    # Trimmed to last 3
    assert len(session.messages) == 3
    # Should keep: old-2, user "new", assistant "final"
    assert session.messages[0]["content"] == "old-2"
    assert session.messages[1] == {"role": "user", "content": "new"}
    assert session.messages[2] == {"role": "assistant", "content": "final"}


def test_max_messages_applies_after_turn_limit_reached(
    registry: ModuleRegistry,
) -> None:
    """When max_turns is reached, trimming should still apply after final response."""
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 1})]),
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 2})]),
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 3})]),
            LLMResponse(content="final"),  # Forced final after max_turns
        ]
    )
    # max_turns=3 means we do 3 tool calls then forced final
    event_loop = EventLoop(dispatcher, llm, registry, max_turns=3)

    # Each tool call adds:
    # - assistant message with tool_calls
    # - tool result message
    # For 3 turns: 3 user + 3 assistant + 3 tool = 9 messages expected before final
    # Plus final response = 1 more
    # But start at limit 5
    session = Session(id="session-1", max_messages=5)

    result = asyncio.run(event_loop.process_turn(session, "loop"))

    assert result == "final"
    # Final message count should be capped at 5
    assert len(session.messages) <= 5
    assert session.messages[-1] == {"role": "assistant", "content": "final"}


def test_max_messages_is_session_not_global(registry: ModuleRegistry) -> None:
    """Each session has its own max_messages setting."""
    # Each session needs its own EventLoop with its own LLM to avoid shared state issues

    # Session with low limit
    dispatcher_low = _FakeDispatcher()
    llm_low = _FakeLLM([LLMResponse(content="done")])
    event_loop_low = EventLoop(dispatcher_low, llm_low, registry)
    session_low = Session(id="session-low", max_messages=2)
    session_low.messages = [{"role": "user", "content": "old"}]

    # Session with high limit (default 1000)
    dispatcher_high = _FakeDispatcher()
    llm_high = _FakeLLM([LLMResponse(content="done")])
    event_loop_high = EventLoop(dispatcher_high, llm_high, registry)
    session_high = Session(id="session-high", max_messages=10)
    session_high.messages = [{"role": "user", "content": "old"}]

    asyncio.run(event_loop_low.process_turn(session_low, "new"))
    asyncio.run(event_loop_high.process_turn(session_high, "new"))

    assert len(session_low.messages) <= 2
    assert len(session_high.messages) <= 10
