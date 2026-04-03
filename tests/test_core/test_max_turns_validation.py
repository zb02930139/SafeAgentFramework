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

"""Tests for max_turns validation in EventLoop and Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from safe_agent.core import EventLoop, Session
from safe_agent.core.event_loop import MAX_TURNS_LIMIT, validate_max_turns
from safe_agent.core.llm import LLMClient, LLMResponse
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubLLM(LLMClient):
    async def chat(self, messages: list, tools: list) -> LLMResponse:
        return LLMResponse(content="ok")


class _StubModule(BaseModule):
    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            namespace="stub",
            description="stub",
            tools=[
                ToolDescriptor(
                    name="stub:noop",
                    description="no-op",
                    parameters={"type": "object"},
                    action="stub:Noop",
                )
            ],
        )

    async def resolve_conditions(
        self, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        return {}

    async def execute(
        self, tool_name: str, params: dict[str, Any]
    ) -> ToolResult[Any]:
        return ToolResult(success=True, data={})


@pytest.fixture
def registry() -> ModuleRegistry:
    reg = ModuleRegistry()
    reg.register(_StubModule())
    return reg


@pytest.fixture
def llm() -> _StubLLM:
    return _StubLLM()


# ---------------------------------------------------------------------------
# validate_max_turns – standalone function
# ---------------------------------------------------------------------------


class TestValidateMaxTurns:
    """Unit tests for the validate_max_turns helper."""

    # --- happy path ---

    def test_accepts_one(self) -> None:
        assert validate_max_turns(1) == 1

    def test_accepts_default_ten(self) -> None:
        assert validate_max_turns(10) == 10

    def test_accepts_upper_limit(self) -> None:
        assert validate_max_turns(MAX_TURNS_LIMIT) == MAX_TURNS_LIMIT

    def test_accepts_mid_range(self) -> None:
        assert validate_max_turns(500) == 500

    # --- type errors ---

    def test_rejects_bool_true(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns(True)

    def test_rejects_bool_false(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns(False)

    def test_rejects_float(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns(3.5)

    def test_rejects_float_whole_number(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns(10.0)

    def test_rejects_string(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns("10")

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns(None)

    def test_rejects_list(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            validate_max_turns([10])

    # --- value errors ---

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_max_turns(0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_max_turns(-1)

    def test_rejects_large_negative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_max_turns(-9999)

    def test_rejects_above_upper_limit(self) -> None:
        with pytest.raises(ValueError, match=f"must be <= {MAX_TURNS_LIMIT}"):
            validate_max_turns(MAX_TURNS_LIMIT + 1)


# ---------------------------------------------------------------------------
# EventLoop constructor validation
# ---------------------------------------------------------------------------


class TestEventLoopMaxTurnsValidation:
    """Ensure EventLoop.__init__ validates max_turns properly."""

    def test_default_is_ten(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        el = EventLoop(Mock(), llm, registry)
        assert el.max_turns == 10

    def test_custom_value_stored(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        el = EventLoop(Mock(), llm, registry, max_turns=42)
        assert el.max_turns == 42

    def test_rejects_zero(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            EventLoop(Mock(), llm, registry, max_turns=0)

    def test_rejects_negative(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            EventLoop(Mock(), llm, registry, max_turns=-5)

    def test_rejects_bool(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            EventLoop(Mock(), llm, registry, max_turns=True)

    def test_rejects_float(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            EventLoop(Mock(), llm, registry, max_turns=5.0)

    def test_rejects_string(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            EventLoop(Mock(), llm, registry, max_turns="10")

    def test_rejects_none(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            EventLoop(Mock(), llm, registry, max_turns=None)

    def test_rejects_above_limit(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        with pytest.raises(ValueError, match=f"must be <= {MAX_TURNS_LIMIT}"):
            EventLoop(Mock(), llm, registry, max_turns=MAX_TURNS_LIMIT + 1)

    def test_accepts_upper_limit(
        self, registry: ModuleRegistry, llm: _StubLLM
    ) -> None:
        el = EventLoop(Mock(), llm, registry, max_turns=MAX_TURNS_LIMIT)
        assert el.max_turns == MAX_TURNS_LIMIT


# ---------------------------------------------------------------------------
# Agent constructor validation
# ---------------------------------------------------------------------------


class TestAgentMaxTurnsValidation:
    """Ensure Agent.__init__ validates max_turns before proceeding."""

    @pytest.fixture
    def policy_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "policies"
        d.mkdir()
        (d / "policy.json").write_text('{"Version": "2025-01", "Statement": []}')
        return d

    @pytest.fixture
    def mock_llm(self) -> _StubLLM:
        return _StubLLM()

    def test_rejects_zero(self, policy_dir: Path, mock_llm: _StubLLM) -> None:
        from safe_agent import Agent

        with pytest.raises(ValueError, match="must be >= 1"):
            Agent(
                policy_dir=policy_dir,
                llm_client=mock_llm,
                modules=[],
                max_turns=0,
            )

    def test_rejects_negative(self, policy_dir: Path, mock_llm: _StubLLM) -> None:
        from safe_agent import Agent

        with pytest.raises(ValueError, match="must be >= 1"):
            Agent(
                policy_dir=policy_dir,
                llm_client=mock_llm,
                modules=[],
                max_turns=-1,
            )

    def test_rejects_bool(self, policy_dir: Path, mock_llm: _StubLLM) -> None:
        from safe_agent import Agent

        with pytest.raises(TypeError, match="must be an int"):
            Agent(
                policy_dir=policy_dir,
                llm_client=mock_llm,
                modules=[],
                max_turns=True,
            )

    def test_rejects_float(self, policy_dir: Path, mock_llm: _StubLLM) -> None:
        from safe_agent import Agent

        with pytest.raises(TypeError, match="must be an int"):
            Agent(
                policy_dir=policy_dir,
                llm_client=mock_llm,
                modules=[],
                max_turns=5.0,
            )

    def test_rejects_string(self, policy_dir: Path, mock_llm: _StubLLM) -> None:
        from safe_agent import Agent

        with pytest.raises(TypeError, match="must be an int"):
            Agent(
                policy_dir=policy_dir,
                llm_client=mock_llm,
                modules=[],
                max_turns="10",
            )

    def test_rejects_above_limit(self, policy_dir: Path, mock_llm: _StubLLM) -> None:
        from safe_agent import Agent

        with pytest.raises(ValueError, match=f"must be <= {MAX_TURNS_LIMIT}"):
            Agent(
                policy_dir=policy_dir,
                llm_client=mock_llm,
                modules=[],
                max_turns=MAX_TURNS_LIMIT + 1,
            )

    def test_valid_value_propagates(
        self, policy_dir: Path, mock_llm: _StubLLM
    ) -> None:
        from safe_agent import Agent

        agent = Agent(
            policy_dir=policy_dir,
            llm_client=mock_llm,
            modules=[],
            max_turns=25,
        )
        assert agent.event_loop.max_turns == 25
