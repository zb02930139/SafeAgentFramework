"""Tests for safe_agent.core.dispatcher — ToolDispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from safe_agent.core.audit import AuditLogger
from safe_agent.core.dispatcher import _DISPATCH_FAILED, ToolDispatcher
from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.models import Decision, Policy
from safe_agent.iam.policy import PolicyStore
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOW_ALL_POLICY = Policy.model_validate(
    {
        "Version": "2025-01",
        "Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}],
    }
)

_DENY_ALL_POLICY = Policy.model_validate(
    {
        "Version": "2025-01",
        "Statement": [{"Effect": "Deny", "Action": ["*"], "Resource": ["*"]}],
    }
)


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
    """Build a concrete BaseModule for testing."""
    _resource_param = resource_param or []
    _execute_result = execute_result or ToolResult(success=True, data="ok")
    _conditions = conditions or {}

    class _Module(BaseModule):
        def describe(self) -> ModuleDescriptor:
            return ModuleDescriptor(
                namespace=namespace,
                description=f"Test module {namespace}.",
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
            self, _tool_name: str, _params: dict[str, Any]
        ) -> dict[str, Any]:
            if resolve_raises:
                raise RuntimeError("resolve_conditions exploded")
            return _conditions

        async def execute(self, _tool_name: str, _params: dict[str, Any]) -> ToolResult:
            if execute_raises:
                raise RuntimeError("execute exploded")
            return _execute_result

    return _Module()


def _make_dispatcher(
    module: BaseModule,
    policy: Policy,
    tmp_path: Path,
) -> ToolDispatcher:
    """Wire up a ToolDispatcher with the given module and policy."""
    registry = ModuleRegistry()
    registry.register(module)
    store = PolicyStore()
    store.add_policy(policy)
    return ToolDispatcher(
        registry,
        PolicyEvaluator(store),
        AuditLogger(log_path=tmp_path / "audit.jsonl"),
    )


def _read_audit(tmp_path: Path) -> list:
    return AuditLogger(log_path=tmp_path / "audit.jsonl").read_entries()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolDispatcher:
    """Tests for ToolDispatcher.dispatch()."""

    async def test_allowed_call_executes_and_returns_result(
        self, tmp_path: Path
    ) -> None:
        """An allowed tool call should execute and return the module result."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")
        assert result.success is True
        assert result.data == "ok"

    async def test_denied_call_returns_generic_error(self, tmp_path: Path) -> None:
        """A denied call must return the opaque _DISPATCH_FAILED string."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")
        assert result.success is False
        assert result.error == _DISPATCH_FAILED

    async def test_unknown_tool_returns_same_error_as_denied(
        self, tmp_path: Path
    ) -> None:
        """Unknown tool and denied must return the same opaque error string."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)

        denied = await dispatcher.dispatch("fs:Read", {}, "s1")
        unknown = await dispatcher.dispatch("ghost:Op", {}, "s1")

        assert denied.error == unknown.error == _DISPATCH_FAILED

    async def test_unknown_tool_is_audited(self, tmp_path: Path) -> None:
        """An unknown tool call must produce an audit entry — no silent probing."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("ghost:Op", {}, "sess-probe")

        entries = _read_audit(tmp_path)
        assert len(entries) == 1
        assert entries[0].tool_name == "ghost:Op"
        assert entries[0].decision == Decision.DENIED_IMPLICIT
        assert "__unknown_tool__" in entries[0].matched_statements

    async def test_multi_resource_all_allowed_executes(self, tmp_path: Path) -> None:
        """With multiple resource params all allowed, execution should proceed."""
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")
        assert result.success is True

    async def test_multi_resource_any_denied_returns_denied(
        self, tmp_path: Path
    ) -> None:
        """If any resource is denied, the whole call is denied."""
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")
        assert result.success is False
        assert result.error == _DISPATCH_FAILED

    async def test_conditions_passed_to_evaluator(self, tmp_path: Path) -> None:
        """Conditions resolved by the module are forwarded to the evaluator."""
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
        """Unmatched conditions produce an implicit deny."""
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
        assert result.error == _DISPATCH_FAILED

    async def test_audit_logger_called_for_allowed(self, tmp_path: Path) -> None:
        """An allowed call should produce an audit log entry."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "sess-42")

        entries = _read_audit(tmp_path)
        assert len(entries) == 1
        assert entries[0].session_id == "sess-42"
        assert entries[0].tool_name == "fs:Read"
        assert entries[0].decision == Decision.ALLOWED

    async def test_audit_logger_called_for_denied(self, tmp_path: Path) -> None:
        """A denied call should produce an audit log entry."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "sess-7")

        entries = _read_audit(tmp_path)
        assert len(entries) == 1
        assert entries[0].decision == Decision.DENIED_EXPLICIT

    async def test_audit_logger_called_per_resource(self, tmp_path: Path) -> None:
        """Each resource should produce its own audit log entry."""
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")

        entries = _read_audit(tmp_path)
        assert len(entries) == 2

    async def test_no_resource_param_uses_empty_resource(self, tmp_path: Path) -> None:
        """A tool with no resource_param is evaluated against empty resource."""
        module = _make_module("sys", "sys:Ping", "system:Ping")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("sys:Ping", {}, "s1")
        assert result.success is True
        assert len(_read_audit(tmp_path)) == 1

    async def test_execute_not_called_when_denied(self, tmp_path: Path) -> None:
        """module.execute() must not be called when the call is denied."""
        execute_called = False

        class _TrackingModule(BaseModule):
            def describe(self) -> ModuleDescriptor:
                return ModuleDescriptor(
                    namespace="t",
                    description="Tracking.",
                    tools=[
                        ToolDescriptor(name="t:Op", description="Op.", action="t:Op")
                    ],
                )

            async def resolve_conditions(
                self, _tn: str, _p: dict[str, Any]
            ) -> dict[str, Any]:
                return {}

            async def execute(self, _tn: str, _p: dict[str, Any]) -> ToolResult:
                nonlocal execute_called
                execute_called = True
                return ToolResult(success=True)

        registry = ModuleRegistry()
        registry.register(_TrackingModule())
        store = PolicyStore()
        store.add_policy(_DENY_ALL_POLICY)
        dispatcher = ToolDispatcher(
            registry,
            PolicyEvaluator(store),
            AuditLogger(log_path=tmp_path / "audit.jsonl"),
        )
        result = await dispatcher.dispatch("t:Op", {}, "s1")
        assert result.success is False
        assert execute_called is False

    async def test_resolve_conditions_exception_returns_generic_error(
        self, tmp_path: Path
    ) -> None:
        """An exception in resolve_conditions returns the opaque error."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", resolve_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is False
        assert result.error == _DISPATCH_FAILED

    async def test_resolve_conditions_exception_is_audited(
        self, tmp_path: Path
    ) -> None:
        """An exception in resolve_conditions must still produce an audit entry."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", resolve_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Read", {}, "s1")

        entries = _read_audit(tmp_path)
        assert len(entries) == 1
        assert entries[0].decision == Decision.DENIED_IMPLICIT
        assert "__internal_error__" in entries[0].matched_statements

    async def test_execute_exception_returns_generic_error(
        self, tmp_path: Path
    ) -> None:
        """An exception in module.execute() returns the opaque error."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", execute_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is False
        assert result.error == _DISPATCH_FAILED

    async def test_execute_exception_is_audited(self, tmp_path: Path) -> None:
        """An exception in module.execute() must produce an audit entry."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", execute_raises=True)
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Read", {}, "s1")

        entries = _read_audit(tmp_path)
        # 1 for the resource evaluation + 1 for the execute error
        assert len(entries) >= 2
        decisions = {e.decision for e in entries}
        assert Decision.DENIED_IMPLICIT in decisions

    async def test_empty_session_id_raises(self, tmp_path: Path) -> None:
        """An empty session_id should raise ValueError."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        with pytest.raises(ValueError, match="session_id"):
            await dispatcher.dispatch("fs:Read", {}, "")
