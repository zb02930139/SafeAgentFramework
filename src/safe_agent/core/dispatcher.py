"""ToolDispatcher: the single enforcement point for all tool calls."""

from __future__ import annotations

from datetime import UTC, datetime

from safe_agent.core.audit import AuditEntry, AuditLogger
from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.models import AuthorizationRequest, Decision
from safe_agent.modules.base import ToolResult
from safe_agent.modules.registry import ModuleRegistry


class ToolDispatcher:
    """The single enforcement point where every LLM tool call is authorised.

    There is exactly one code path from an LLM tool call to a module's
    ``execute()`` method, and it passes through policy evaluation for every
    affected resource. There is no bypass, no override flag, and no skip path.

    Evaluation follows these steps:

    1. Look up the tool in the registry. Unknown tools are rejected immediately.
    2. Extract resource(s) from ``params`` using ``descriptor.resource_param``.
       If no resources are defined, evaluation proceeds with a single empty
       resource string (the policy must explicitly allow this case).
    3. Call ``module.resolve_conditions()`` to obtain runtime context values.
    4. For each resource: build an :class:`~safe_agent.iam.models.AuthorizationRequest`
       and call :meth:`~safe_agent.iam.evaluator.PolicyEvaluator.evaluate`.
    5. Log every decision via the :class:`~safe_agent.core.audit.AuditLogger`.
    6. If **any** resource is denied → return a generic
       ``ToolResult(success=False, error="Action denied")``. No policy details,
       matched statement names, or resource identifiers are exposed to the caller.
    7. If **all** resources are allowed → call ``module.execute()`` and return
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

    async def dispatch(
        self,
        tool_name: str,
        params: dict,
        session_id: str,
    ) -> ToolResult:
        """Authorise and execute a tool call.

        Args:
            tool_name: The fully-qualified tool name (e.g. ``"fs:ReadFile"``).
            params: Raw input parameters for the tool.
            session_id: Identifier for the calling session, used in audit logs.

        Returns:
            A :class:`~safe_agent.modules.base.ToolResult` — either the result
            of a successful execution or a generic failure if the tool is
            unknown or the call was denied.
        """
        # Step 1 — look up the tool.
        lookup = self._registry.get_tool(tool_name)
        if lookup is None:
            return ToolResult(success=False, error="Unknown tool")

        module, descriptor = lookup

        # Step 2 — extract resources from params.
        resources: list[str] = []
        for param_name in descriptor.resource_param:
            value = params.get(param_name)
            if value is not None:
                resources.append(str(value))

        # If the tool declares no resource params, evaluate against a single
        # empty resource string (policy must explicitly allow this).
        if not resources:
            resources = [""]

        # Step 3 — resolve runtime conditions from the module.
        resolved_conditions = await module.resolve_conditions(tool_name, params)

        # Steps 4 & 5 — evaluate each resource and log every decision.
        timestamp = datetime.now(tz=UTC).isoformat()
        denied = False

        for resource in resources:
            request = AuthorizationRequest(
                action=descriptor.action,
                resource=resource,
                context=resolved_conditions,
            )
            result = self._evaluator.evaluate(request)

            self._audit_logger.log(
                AuditEntry(
                    session_id=session_id,
                    timestamp=timestamp,
                    tool_name=tool_name,
                    params=params,
                    resolved_conditions=resolved_conditions,
                    decision=result.decision,
                    matched_statements=[
                        stmt.sid or "" for stmt in result.matched_statements
                    ],
                )
            )

            if result.decision != Decision.ALLOWED:
                denied = True
                # Continue logging remaining resources before returning.

        # Step 6 — deny if any resource was denied, with no policy details.
        if denied:
            return ToolResult(success=False, error="Action denied")

        # Step 7 — all resources allowed; execute.
        return await module.execute(tool_name, params)
