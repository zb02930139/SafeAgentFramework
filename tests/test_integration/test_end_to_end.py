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


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from safe_agent import Agent
from safe_agent.access.models import Decision, Policy, Statement
from safe_agent.core.llm import LLMClient, LLMResponse, ToolCall
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


class MockLLM(LLMClient):
    """Test double that returns a predefined sequence of chat responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(
            {
                "messages": [message.copy() for message in messages],
                "tools": [tool.copy() for tool in tools],
            }
        )
        if not self._responses:
            raise AssertionError("MockLLM received more chat calls than expected")
        return self._responses.pop(0)


class EchoModule(BaseModule):
    """Minimal module used to exercise end-to-end tool dispatch."""

    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            namespace="test",
            description="Test module",
            tools=[
                ToolDescriptor(
                    name="test:echo",
                    description="Echo a message",
                    parameters={
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["target", "message"],
                    },
                    action="test:Echo",
                    resource_param="target",
                )
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"tool": tool_name, "message_length": len(str(params["message"]))}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        if params.get("message") == "kaboom":
            raise RuntimeError("sensitive execution detail")
        return ToolResult(success=True, data={"echoed": params["message"]})


def write_policy(policy_dir: Path, target: str) -> None:
    policy = {
        "Version": "2025-01",
        "Statement": [
            {
                "Sid": "allow-test-echo",
                "Effect": "Allow",
                "Action": ["test:Echo"],
                "Resource": [target],
            }
        ],
    }
    (policy_dir / "policy.json").write_text(json.dumps(policy), encoding="utf-8")


@pytest.mark.asyncio
async def test_allowed_tool_call_succeeds_end_to_end(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    audit_log_path = tmp_path / "audit.jsonl"
    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        # LLM returns sanitized name (as a real provider would)
                        name="test__echo",
                        params={
                            "target": "allowed-resource",
                            "message": "hello world",
                        },
                    )
                ]
            ),
            LLMResponse(content="tool finished"),
        ]
    )

    agent = Agent(
        policy_dir=policy_dir,
        llm_client=llm,
        modules=[EchoModule()],
        audit_log_path=audit_log_path,
    )

    response, session_id = await agent.chat("say hi")

    assert response == "tool finished"
    assert session_id is not None
    assert len(llm.calls) == 2
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "say hi"}]
    # The event loop sends sanitized names to the LLM; tool result messages
    # in the history the LLM receives also use the sanitized form.
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["name"] == "test__echo"
    assert json.loads(tool_message["content"]) == {
        "success": True,
        "data": {"echoed": "hello world"},
        "error": None,
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_session_id_enables_multi_turn_conversation(tmp_path: Path) -> None:
    """Returned session_id can be passed back to continue the same session."""
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    llm = MockLLM([LLMResponse(content="turn one"), LLMResponse(content="turn two")])
    agent = Agent(
        policy_dir=policy_dir,
        llm_client=llm,
        modules=[EchoModule()],
        audit_log_path=tmp_path / "audit.jsonl",
    )

    _, session_id = await agent.chat("first message")
    response2, session_id2 = await agent.chat("second message", session_id=session_id)

    assert response2 == "turn two"
    assert session_id2 == session_id
    # Both turns are in the same session history
    session = agent.session_manager.get(session_id)
    assert session is not None
    assert len(session.messages) == 4  # user + assistant + user + assistant


@pytest.mark.asyncio
async def test_unknown_session_id_raises_key_error(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    agent = Agent(
        policy_dir=policy_dir,
        llm_client=MockLLM([]),
        modules=[EchoModule()],
        audit_log_path=tmp_path / "audit.jsonl",
    )

    with pytest.raises(KeyError):
        await agent.chat("hello", session_id="nonexistent-session-id")


@pytest.mark.asyncio
async def test_denied_tool_call_returns_generic_error(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="test__echo",
                        params={"target": "denied-resource", "message": "secret"},
                    )
                ]
            ),
            LLMResponse(content="done"),
        ]
    )

    agent = Agent(
        policy_dir=policy_dir,
        llm_client=llm,
        modules=[EchoModule()],
        audit_log_path=tmp_path / "audit.jsonl",
    )

    response, _ = await agent.chat("try denied tool")

    assert response == "done"
    tool_message = llm.calls[1]["messages"][-1]
    payload = json.loads(tool_message["content"])
    assert payload == {
        "success": False,
        "data": None,
        "error": "Dispatch failed",
        "metadata": {},
    }
    assert "denied-resource" not in tool_message["content"]
    assert "secret" not in tool_message["content"]


@pytest.mark.asyncio
async def test_audit_log_contains_entries_for_all_decisions(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="test__echo",
                        params={
                            "target": "allowed-resource",
                            "message": "hello",
                        },
                    )
                ]
            ),
            LLMResponse(content="allowed done"),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="test__echo",
                        params={
                            "target": "denied-resource",
                            "message": "hello",
                        },
                    )
                ]
            ),
            LLMResponse(content="denied done"),
        ]
    )
    audit_log_path = tmp_path / "audit.jsonl"
    agent = Agent(
        policy_dir=policy_dir,
        llm_client=llm,
        modules=[EchoModule()],
        audit_log_path=audit_log_path,
    )

    await agent.chat("allowed")
    await agent.chat("denied")

    entries = agent.audit_logger.read_entries()
    assert [entry.decision for entry in entries] == [
        Decision.ALLOWED,
        Decision.ALLOWED,
        Decision.DENIED_IMPLICIT,
    ]
    assert entries[0].matched_statements == ["allow-test-echo"]
    assert entries[1].matched_statements == ["__executed__"]
    assert entries[2].matched_statements == []


@pytest.mark.asyncio
async def test_policies_are_frozen_after_startup(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    agent = Agent(
        policy_dir=policy_dir,
        llm_client=MockLLM([LLMResponse(content="ok")]),
        modules=[EchoModule()],
        audit_log_path=tmp_path / "audit.jsonl",
    )

    extra_policy = Policy(
        Version="2025-01",
        Statement=[
            Statement(
                Sid="later",
                Effect="Allow",
                Action=["test:Echo"],
                Resource=["later-resource"],
            )
        ],
    )

    with pytest.raises(RuntimeError, match="frozen"):
        agent.policy_store.add_policy(extra_policy)


@pytest.mark.asyncio
async def test_session_isolation_keeps_message_histories_separate(
    tmp_path: Path,
) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    write_policy(policy_dir, "allowed-resource")
    llm = MockLLM([LLMResponse(content="first"), LLMResponse(content="second")])
    agent = Agent(
        policy_dir=policy_dir,
        llm_client=llm,
        modules=[EchoModule()],
        audit_log_path=tmp_path / "audit.jsonl",
    )

    response1, sid1 = await agent.chat("hello from one")
    response2, sid2 = await agent.chat("hello from two")

    assert response1 == "first"
    assert response2 == "second"
    assert sid1 != sid2

    session_one = agent.session_manager.get(sid1)
    session_two = agent.session_manager.get(sid2)
    assert session_one is not None
    assert session_two is not None
    assert session_one.messages == [
        {"role": "user", "content": "hello from one"},
        {"role": "assistant", "content": "first"},
    ]
    assert session_two.messages == [
        {"role": "user", "content": "hello from two"},
        {"role": "assistant", "content": "second"},
    ]
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "hello from one"}]
    assert llm.calls[1]["messages"] == [{"role": "user", "content": "hello from two"}]
