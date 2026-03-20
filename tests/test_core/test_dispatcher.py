"""Tests for safe_agent.core.dispatcher — ToolDispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safe_agent.core.audit import AuditLogger
from safe_agent.core.dispatcher import ToolDispatcher
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
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["*"],
                "Resource": ["*"],
            }
        ],
    }
)

_DENY_ALL_POLICY = Policy.model_validate(
    {
        "Version": "2025-01",
        "Statement": [
            {
                "Effect": "Deny",
                "Action": ["*"],
                "Resource": ["*"],
            }
        ],
    }
)


def _make_module(
    namespace: str,
    tool_name: str,
    action: str,
    resource_param: list[str] | None = None,
    condition_keys: list[str] | None = None,
    execute_result: ToolResult | None = None,
    conditions: dict[str, Any] | None = None,
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
                        condition_keys=condition_keys or [],
                    )
                ],
            )

        async def resolve_conditions(
            self, tool_name: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            return _conditions

        async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
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
    evaluator = PolicyEvaluator(store)

    audit_logger = AuditLogger(log_path=tmp_path / "audit.jsonl")

    return ToolDispatcher(registry, evaluator, audit_logger)


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
        """A denied tool call must return 'Action denied' with no policy details."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "s1")
        assert result.success is False
        assert result.error == "Action denied"

    async def test_denied_error_exposes_no_policy_details(self, tmp_path: Path) -> None:
        """The denial error must not leak resource, policy, or statement info."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {"path": "/secret"}, "s1")
        assert result.error == "Action denied"
        # Must not contain path, policy hints, or decision type
        assert "/secret" not in (result.error or "")
        assert "Deny" not in (result.error or "")
        assert "DENIED" not in (result.error or "")

    async def test_unknown_tool_returns_unknown_tool_error(
        self, tmp_path: Path
    ) -> None:
        """An unregistered tool name should return 'Unknown tool'."""
        module = _make_module("fs", "fs:Read", "filesystem:Read")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("ghost:Op", {}, "s1")
        assert result.success is False
        assert result.error == "Unknown tool"

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
        assert result.error == "Action denied"

    async def test_conditions_passed_to_evaluator(self, tmp_path: Path) -> None:
        """Conditions resolved by the module should be forwarded to the evaluator."""
        # Allow only when env == "prod"
        policy = Policy.model_validate(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["*"],
                        "Resource": ["*"],
                        "Condition": {
                            "StringEquals": {"env": "prod"},
                        },
                    }
                ],
            }
        )
        # Module returns env=prod → should be allowed
        module = _make_module(
            "fs",
            "fs:Read",
            "filesystem:Read",
            conditions={"env": "prod"},
        )
        dispatcher = _make_dispatcher(module, policy, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is True

    async def test_conditions_mismatch_results_in_deny(self, tmp_path: Path) -> None:
        """If resolved conditions don't satisfy the policy, the call is denied."""
        policy = Policy.model_validate(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["*"],
                        "Resource": ["*"],
                        "Condition": {
                            "StringEquals": {"env": "prod"},
                        },
                    }
                ],
            }
        )
        # Module returns env=staging → conditions not satisfied → implicit deny
        module = _make_module(
            "fs",
            "fs:Read",
            "filesystem:Read",
            conditions={"env": "staging"},
        )
        dispatcher = _make_dispatcher(module, policy, tmp_path)
        result = await dispatcher.dispatch("fs:Read", {}, "s1")
        assert result.success is False
        assert result.error == "Action denied"

    async def test_audit_logger_called_for_allowed(self, tmp_path: Path) -> None:
        """An allowed call should produce an audit log entry."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "sess-42")

        entries = AuditLogger(log_path=tmp_path / "audit.jsonl").read_entries()
        assert len(entries) == 1
        assert entries[0].session_id == "sess-42"
        assert entries[0].tool_name == "fs:Read"
        assert entries[0].decision == Decision.ALLOWED

    async def test_audit_logger_called_for_denied(self, tmp_path: Path) -> None:
        """A denied call should produce an audit log entry."""
        module = _make_module("fs", "fs:Read", "filesystem:Read", ["path"])
        dispatcher = _make_dispatcher(module, _DENY_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Read", {"path": "/etc/hosts"}, "sess-7")

        entries = AuditLogger(log_path=tmp_path / "audit.jsonl").read_entries()
        assert len(entries) == 1
        assert entries[0].decision == Decision.DENIED_EXPLICIT

    async def test_audit_logger_called_per_resource(self, tmp_path: Path) -> None:
        """Each resource should produce its own audit log entry."""
        module = _make_module("fs", "fs:Copy", "filesystem:Copy", ["src", "dst"])
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        await dispatcher.dispatch("fs:Copy", {"src": "/a", "dst": "/b"}, "s1")

        entries = AuditLogger(log_path=tmp_path / "audit.jsonl").read_entries()
        assert len(entries) == 2

    async def test_no_resource_param_uses_empty_resource(self, tmp_path: Path) -> None:
        """A tool with no resource_param should still be evaluated (empty resource)."""
        module = _make_module("sys", "sys:Ping", "system:Ping")
        dispatcher = _make_dispatcher(module, _ALLOW_ALL_POLICY, tmp_path)
        result = await dispatcher.dispatch("sys:Ping", {}, "s1")
        assert result.success is True

        entries = AuditLogger(log_path=tmp_path / "audit.jsonl").read_entries()
        assert len(entries) == 1

    async def test_execute_not_called_when_denied(self, tmp_path: Path) -> None:
        """module.execute() must not be called when the call is denied."""
        execute_called = False
        expected = ToolResult(success=True, data="should not appear")

        class _TrackingModule(BaseModule):
            def describe(self) -> ModuleDescriptor:
                return ModuleDescriptor(
                    namespace="t",
                    description="Tracking module.",
                    tools=[
                        ToolDescriptor(
                            name="t:Op",
                            description="Op.",
                            action="t:Op",
                        )
                    ],
                )

            async def resolve_conditions(
                self, tool_name: str, params: dict[str, Any]
            ) -> dict[str, Any]:
                return {}

            async def execute(
                self, tool_name: str, params: dict[str, Any]
            ) -> ToolResult:
                nonlocal execute_called
                execute_called = True
                return expected

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
