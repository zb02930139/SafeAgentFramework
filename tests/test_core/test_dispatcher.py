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

"""Tests for safe_agent.core.dispatcher — ToolDispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from safe_agent.access.evaluator import PolicyEvaluator
from safe_agent.access.models import Decision, Policy
from safe_agent.access.policy import PolicyStore
from safe_agent.core.audit import AuditLogger
from safe_agent.core.dispatcher import ToolDispatcher
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

_ALLOW_ALL = Policy.model_validate(
    {
        "Version": "2025-01",
        "Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}],
    }
)

_DENY_ALL = Policy.model_validate(
    {
        "Version": "2025-01",
        "Statement": [{"Effect": "Deny", "Action": ["*"], "Resource": ["*"]}],
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(
    namespace: str,
    tool_name: str,
    action: str,
    resource_param: list[str] | None = None,
    execute_result: ToolResult | None = None,
    conditions: dict[str, Any] | None = None,
    resolve_raises: bool = False,
    execute_raises: bool = False,
) -> BaseModule:
    _resource_param = resource_param or []
    _execute_result = execute_result or ToolResult(success=True, data="ok")
    _conditions = conditions or {}

    class _M(BaseModule):
        def describe(self) -> ModuleDescriptor:
            return ModuleDescriptor(
                namespace=namespace,
                description="Test.",
                tools=[
                    ToolDescriptor(
                        name=tool_name,
                        description="A tool.",
                        action=action,
                        resource_param=_resource_param,
                    )
                ],
            )

        async def resolve_conditions(
            self, tn: str, p: dict[str, Any]
        ) -> dict[str, Any]:
            if resolve_raises:
                raise RuntimeError("resolve boom")
            return _conditions

        async def execute(self, tn: str, p: dict[str, Any]) -> ToolResult[Any]:
            if execute_raises:
                raise RuntimeError("execute boom")
            return _execute_result

    return _M()


def _make_dispatcher(
    module: BaseModule,
    policy: Policy,
    tmp_path: Path,
) -> ToolDispatcher:
    registry = ModuleRegistry()
    registry.register(module)
    store = PolicyStore()
    store.add_policy(policy)
    return ToolDispatcher(
        registry,
        PolicyEvaluator(store),
        AuditLogger(log_path=tmp_path / "audit.jsonl"),
    )


def _audit(tmp_path: Path) -> AuditLogger:
    return AuditLogger(log_path=tmp_path / "audit.jsonl")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolDispatcher:
    """Tests for ToolDispatcher.dispatch()."""

    async def test_allowed_call_executes_and_returns_result(
        self, tmp_path: Path
    ) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")
        assert result.success is True
        assert result.data == "ok"

    async def test_denied_call_returns_generic_error(self, tmp_path: Path) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")
        assert result.success is False
        assert result.error == "Dispatch failed"

    async def test_unknown_tool_returns_dispatch_failed(self, tmp_path: Path) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch("ghost:Op", {}, "s1")
        assert result.success is False
        assert result.error == "Dispatch failed"

    async def test_unknown_and_denied_error_identical(self, tmp_path: Path) -> None:
        """Unknown tool and denied tool must return the same error string."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL, tmp_path)
        denied = await dispatcher.dispatch("fs:Read", {"path": "/x"}, "s1")
        unknown = await dispatcher.dispatch("ghost:Op", {}, "s1")
        assert denied.error == unknown.error == "Dispatch failed"

    async def test_multi_resource_all_allowed_executes(self, tmp_path: Path) -> None:
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")
        assert result.success is True

    async def test_multi_resource_any_denied_returns_failed(
        self, tmp_path: Path
    ) -> None:
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _DENY_ALL, tmp_path)
        result = await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")
        assert result.success is False
        assert result.error == "Dispatch failed"

    async def test_conditions_passed_to_evaluator_allow(self, tmp_path: Path) -> None:
        policy = Policy.model_validate(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["*"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            }
        )
        module = _make_module(
            "fs", "fs:Read", "filesystem:Read", conditions={"env": "prod"}
        )
        dispatcher = _make_dispatcher(module, policy, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is True

    async def test_conditions_mismatch_results_in_deny(self, tmp_path: Path) -> None:
        policy = Policy.model_validate(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["*"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            }
        )
        module = _make_module(
            "fs", "fs:Read", "filesystem:Read", conditions={"env": "staging"}
        )
        dispatcher = _make_dispatcher(module, policy, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is False

    async def test_audit_logged_for_allowed(self, tmp_path: Path) -> None:
        """A successful dispatch produces two audit entries.

        One for the policy evaluation (ALLOWED) and one post-execute entry
        (__executed__).
        """
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "sess-42")
        entries = _audit(tmp_path).read_entries()
        # policy-decision entry + post-execute entry
        assert len(entries) == 2
        assert all(e.session_id == "sess-42" for e in entries)
        decisions = {e.decision for e in entries}
        assert Decision.ALLOWED in decisions

    async def test_audit_logged_for_denied(self, tmp_path: Path) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "sess-7")
        entries = _audit(tmp_path).read_entries()
        assert len(entries) == 1
        assert entries[0].decision == Decision.DENIED_EXPLICIT

    async def test_audit_logged_for_unknown_tool(self, tmp_path: Path) -> None:
        """Unknown tool probes must be recorded in the audit log."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        await dispatcher.dispatch("ghost:Op", {}, "s1")
        entries = _audit(tmp_path).read_entries()
        assert len(entries) == 1
        assert entries[0].tool_name == "ghost:Op"
        assert entries[0].matched_statements == ["__unknown_tool__"]

    async def test_audit_per_resource(self, tmp_path: Path) -> None:
        """Two resources → two policy-decision entries + one execution entry."""
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")
        entries = _audit(tmp_path).read_entries()
        # 2 resource evaluations + 1 post-execute entry
        assert len(entries) == 3

    async def test_no_resource_param_uses_empty_resource(self, tmp_path: Path) -> None:
        module = _make_module("sys", "sys:Ping", "system:Ping")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch("sys:Ping", {}, "s1")
        assert result.success is True
        entries = _audit(tmp_path).read_entries()
        # 1 policy-decision entry + 1 post-execute entry
        assert len(entries) == 2

    async def test_execute_not_called_when_denied(self, tmp_path: Path) -> None:
        execute_called = False

        class _TrackingModule(BaseModule):
            def describe(self) -> ModuleDescriptor:
                return ModuleDescriptor(
                    namespace="t",
                    description=".",
                    tools=[ToolDescriptor(name="t:Op", description=".", action="t:Op")],
                )

            async def resolve_conditions(
                self, tn: str, p: dict[str, Any]
            ) -> dict[str, Any]:
                return {}

            async def execute(self, tn: str, p: dict[str, Any]) -> ToolResult[Any]:
                nonlocal execute_called
                execute_called = True
                return ToolResult(success=True)

        registry = ModuleRegistry()
        registry.register(_TrackingModule())
        store = PolicyStore()
        store.add_policy(_DENY_ALL)
        dispatcher = ToolDispatcher(
            registry,
            PolicyEvaluator(store),
            AuditLogger(log_path=tmp_path / "audit.jsonl"),
        )
        result = await dispatcher.dispatch("t:Op", {}, "s1")
        assert result.success is False
        assert execute_called is False

    async def test_resolve_conditions_exception_returns_failed(
        self, tmp_path: Path
    ) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read", resolve_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is False
        assert result.error == "Dispatch failed"
        entries = _audit(tmp_path).read_entries()
        assert len(entries) == 1
        assert entries[0].matched_statements == ["__internal_error__"]

    async def test_execute_exception_returns_failed(self, tmp_path: Path) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read", execute_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is False
        assert result.error == "Dispatch failed"
        entries = _audit(tmp_path).read_entries()
        assert any("__execute_error__" in (e.matched_statements or []) for e in entries)

    async def test_empty_session_id_raises(self, tmp_path: Path) -> None:
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        with pytest.raises(ValueError, match="session_id"):
            await dispatcher.dispatch("fs:Read", {}, "")

    async def test_oversized_params_truncated_in_audit(self, tmp_path: Path) -> None:
        """Params exceeding 8 KB should be replaced with the truncation marker."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        big_params: dict[str, Any] = {"data": "x" * (9 * 1024)}
        await dispatcher.dispatch("fs:Read", big_params, "s1")
        entries = _audit(tmp_path).read_entries()
        assert len(entries) >= 1
        assert all(e.params == {"__truncated__": True} for e in entries)

    async def test_oversized_params_truncated_for_unknown_tool(
        self, tmp_path: Path
    ) -> None:
        """Oversized params must be truncated even on the unknown-tool path."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        big_params: dict[str, Any] = {"data": "x" * (9 * 1024)}
        result = await dispatcher.dispatch("ghost:Op", big_params, "s1")
        assert result.success is False
        entries = _audit(tmp_path).read_entries()
        assert len(entries) == 1
        assert entries[0].params == {"__truncated__": True}
        assert entries[0].matched_statements == ["__unknown_tool__"]

    async def test_log_safe_failure_is_fail_closed_on_unknown_tool(
        self, tmp_path: Path
    ) -> None:
        """If audit logging fails on the unknown-tool path, dispatch still denies."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)

        disk_full = OSError("disk full")
        with patch.object(dispatcher._audit_logger, "log", side_effect=disk_full):
            result = await dispatcher.dispatch("ghost:Op", {}, "s1")

        assert result.success is False
        assert result.error == "Dispatch failed"

    async def test_log_safe_failure_is_fail_closed_on_resolve_error(
        self, tmp_path: Path
    ) -> None:
        """Audit failure after resolve_conditions error must still deny."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", resolve_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)

        disk_full = OSError("disk full")
        with patch.object(dispatcher._audit_logger, "log", side_effect=disk_full):
            result = await dispatcher.dispatch("fs:Read", {}, "s1")

        assert result.success is False
        assert result.error == "Dispatch failed"

    async def test_log_safe_failure_is_fail_closed_on_policy_decision(
        self, tmp_path: Path
    ) -> None:
        """Audit failure on the policy-decision path must deny execution."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)

        original_log = dispatcher._audit_logger.log
        call_count = 0

        def fail_on_first_log(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("disk full")
            original_log(*args, **kwargs)

        with patch.object(
            dispatcher._audit_logger, "log", side_effect=fail_on_first_log
        ):
            result = await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")

        assert result.success is False
        assert result.error == "Dispatch failed"
        assert call_count == 1
        assert _audit(tmp_path).read_entries() == []

    async def test_tool_call_id_propagates_to_audit_log(self, tmp_path: Path) -> None:
        """tool_call_id must appear in audit entries for allowed calls."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        result = await dispatcher.dispatch(
            "fs:Read", {"path": "/etc/hosts"}, "s1", tool_call_id="call-abc123"
        )
        assert result.success is True
        entries = _audit(tmp_path).read_entries()
        # All entries should have the tool_call_id
        for entry in entries:
            assert entry.tool_call_id == "call-abc123"

    async def test_tool_call_id_in_audit_for_denied(self, tmp_path: Path) -> None:
        """tool_call_id must appear in audit entries for denied calls."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL, tmp_path)
        await dispatcher.dispatch(
            "fs:Read", {"path": "/etc/hosts"}, "s1", tool_call_id="call-xyz"
        )
        entries = _audit(tmp_path).read_entries()
        assert len(entries) == 1
        assert entries[0].tool_call_id == "call-xyz"
        assert entries[0].decision == Decision.DENIED_EXPLICIT

    async def test_tool_call_id_in_audit_for_unknown_tool(self, tmp_path: Path) -> None:
        """tool_call_id must appear in audit entries for unknown tool probes."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        await dispatcher.dispatch("ghost:Op", {}, "s1", tool_call_id="call-unknown")
        entries = _audit(tmp_path).read_entries()
        assert len(entries) == 1
        assert entries[0].tool_call_id == "call-unknown"
        assert entries[0].matched_statements == ["__unknown_tool__"]

    async def test_tool_call_id_none_when_not_provided(self, tmp_path: Path) -> None:
        """When tool_call_id is not provided, audit entries should have None."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")
        entries = _audit(tmp_path).read_entries()
        for entry in entries:
            assert entry.tool_call_id is None
