"""ToolDispatcher: the single enforcement point for all tool calls."""

from __future__ import annotations

import logging

from safe_agent.core.audit import AuditEntry, AuditLogger
from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.models import AuthorizationRequest, Decision
from safe_agent.modules.base import ToolResult
from safe_agent.modules.registry import ModuleRegistry

logger = logging.getLogger(__name__)

# The single opaque error returned to callers on any dispatch failure.
# Keeping "unknown tool" and "denied" identical prevents callers from using
# the error string to probe which tool names exist in the registry.
_DISPATCH_FAILED = "Dispatch failed"


class ToolDispatcher:
    """The single enforcement point where every LLM tool call is authorised.

    There is exactly one code path from an LLM tool call to a module's
    ``execute()`` method, and it passes through policy evaluation for every
    affected resource. There is no bypass, no override flag, and no skip path.

    Evaluation follows these steps:

    1. Look up the tool in the registry. Unknown tools are rejected *and*
       audited immediately — probing the registry leaves a trace.
    2. Extract resource(s) from ``params`` using ``descriptor.resource_param``.
       If no resources are declared, evaluation proceeds against a single empty
       resource string (``""``) — the policy must explicitly allow this case.
       This contract is enforced here and documented so there is no ambiguity.
    3. Call ``module.resolve_conditions()`` to obtain runtime context values.
    4. For each resource: build an :class:`~safe_agent.iam.models.AuthorizationRequest`
       and call :meth:`~safe_agent.iam.evaluator.PolicyEvaluator.evaluate`.
    5. Log every decision via the :class:`~safe_agent.core.audit.AuditLogger`.
       Every decision — allowed, denied, unknown-tool, and internal errors —
       produces an audit entry. There is no silent path.
    6. On any internal error (exception from ``resolve_conditions``,
       ``evaluate``, or ``log``): log a ``DENIED_IMPLICIT`` entry and return
       a generic failure. Exceptions never propagate to the caller.
    7. If **any** resource is denied → return a generic failure with no policy
       detail, resource name, or matched statement exposed.
    8. If **all** resources are allowed → call ``module.execute()`` and return
       the result.

    Args:
        registry: The :class:`~safe_agent.modules.registry.ModuleRegistry`
            containing all registered modules.
        evaluator: The :class:`~safe_agent.iam.evaluator.PolicyEvaluator`
            to use for authorisation decisions.
        audit_logger: The :class:`~safe_agent.core.audit.AuditLogger` to
            record all decisions to.

    Example::

        dispatcher = ToolDispatcher(registry, evaluator, audit_logger)
        result = await dispatcher.dispatch(
            "fs:ReadFile", {"path": "/etc/hosts"}, "session-1"
        )
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        evaluator: PolicyEvaluator,
        audit_logger: AuditLogger,
    ) -> None:
        """Initialise the dispatcher.

        Args:
            registry: Module registry for tool lookup.
            evaluator: Policy evaluator for authorisation decisions.
            audit_logger: Audit logger for recording every decision.
        """
        self._registry = registry
        self._evaluator = evaluator
        self._audit_logger = audit_logger

    def _log_safe(self, entry: AuditEntry) -> None:
        """Log *entry*, swallowing and recording any logging error."""
        try:
            self._audit_logger.log(entry)
        except Exception:
            logger.exception(
                "safe_agent.dispatcher: audit logging failed for tool '%s' "
                "session '%s' — THIS IS A SECURITY EVENT",
                entry.tool_name,
                entry.session_id,
            )

    async def dispatch(
        self,
        tool_name: str,
        params: dict,
        session_id: str,
    ) -> ToolResult:
        """Authorise and execute a tool call.

        Every outcome — allowed, denied, unknown tool, or internal error —
        produces an audit log entry. No dispatch path is silent.

        Args:
            tool_name: The fully-qualified tool name (e.g. ``"fs:ReadFile"``).
            params: Raw input parameters for the tool.
            session_id: Non-empty identifier for the calling session, used in
                audit logs.

        Returns:
            A :class:`~safe_agent.modules.base.ToolResult`. On any failure
            (unknown tool, denial, or internal error) the error is the opaque
            string ``"Dispatch failed"`` — no policy, resource, or internal
            detail is exposed.
        """
        if not session_id:
            raise ValueError("session_id must be a non-empty string")

        timestamp = AuditLogger.now_iso()

        # Step 1 — look up the tool; audit and reject if unknown.
        lookup = self._registry.get_tool(tool_name)
        if lookup is None:
            self._log_safe(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions={},
                    decision=Decision.DENIED_IMPLICIT,
                    matched_statements=["__unknown_tool__"],
                )
            )
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        module, descriptor = lookup

        # Step 2 — extract resources from params.
        # Contract: if resource_param is empty, evaluate against "" (the policy
        # must explicitly allow resource="" for such tools).
        resources: list[str] = []
        for param_name in descriptor.resource_param:
            value = params.get(param_name)
            if value is not None:
                resources.append(str(value))
        if not resources:
            resources = [""]

        # Step 3 — resolve runtime conditions from the module.
        try:
            resolved_conditions = await module.resolve_conditions(tool_name, params)
        except Exception:
            logger.exception(
                "safe_agent.dispatcher: resolve_conditions raised for tool '%s'",
                tool_name,
            )
            self._log_safe(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions={},
                    decision=Decision.DENIED_IMPLICIT,
                    matched_statements=["__internal_error__"],
                )
            )
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        # Steps 4 & 5 — evaluate each resource and log every decision.
        denied = False
        for resource in resources:
            try:
                request = AuthorizationRequest(
                    action=descriptor.action,
                    resource=resource,
                    context=resolved_conditions,
                )
                result = self._evaluator.evaluate(request)
            except Exception:
                logger.exception(
                    "safe_agent.dispatcher: evaluator raised for tool '%s' "
                    "resource '%s'",
                    tool_name,
                    resource,
                )
                self._log_safe(
                    AuditEntry(
                        session_id=session_id,
                        timestamp=timestamp,
                        tool_name=tool_name,
                        params=params,
                        resolved_conditions=resolved_conditions,
                        decision=Decision.DENIED_IMPLICIT,
                        matched_statements=["__internal_error__"],
                    )
                )
                denied = True
                continue

            self._log_safe(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions=resolved_conditions,
                    decision=result.decision,
                    matched_statements=[stmt.sid for stmt in result.matched_statements],
                )
            )

            if result.decision != Decision.ALLOWED:
                denied = True

        # Step 6/7 — deny if any resource was denied, with no policy details.
        if denied:
            return ToolResult(success=False, error=_DISPATCH_FAILED)

        # Step 8 — all resources allowed; execute.
        try:
            return await module.execute(tool_name, params)
        except Exception:
            logger.exception(
                "safe_agent.dispatcher: module.execute raised for tool '%s'",
                tool_name,
            )
            self._log_safe(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions=resolved_conditions,
                    decision=Decision.DENIED_IMPLICIT,
                    matched_statements=["__execute_error__"],
                )
            )
            return ToolResult(success=False, error=_DISPATCH_FAILED)
