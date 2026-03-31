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

"""Alerting module for SafeAgent.

This module provides alert management capabilities through a pluggable
backend protocol. The framework defines the interface; a plugin provides the
concrete implementation (PagerDuty, OpsGenie, Grafana, etc.).

Security Considerations:
    - Alert Visibility: Alert data returned in ToolResult.data may be visible
      to the LLM. Avoid querying sensitive alert content without proper access
      controls.

    - Privileged Actions: acknowledge_alert, escalate_alert, and silence_alert
      modify alert state and should be treated as privileged actions. Policy
      conditions should restrict access appropriately.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

# Note: This module is now at safe_agent.modules.observability.alerting
# but imports from safe_agent.modules.base remain unchanged since base.py
# is in the parent modules/ directory.

logger = logging.getLogger(__name__)


@runtime_checkable
class AlertingBackend(Protocol):
    """Protocol for alerting backend implementations.

    Implementations provide concrete integrations with alerting systems
    such as PagerDuty, OpsGenie, Grafana, VictorOps, etc.

    Methods:
        list_alerts: Retrieve alerts with optional filtering.
        acknowledge_alert: Mark an alert as acknowledged.
        escalate_alert: Escalate an alert to a higher priority or team.
        silence_alert: Silence an alert for a specified duration.
    """

    async def list_alerts(
        self,
        *,
        severity: str | None = None,
        source: str | None = None,
        state: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve alerts with optional filtering.

        Args:
            severity: Optional severity filter (e.g., "critical", "warning").
            source: Optional alert source filter (e.g., "prometheus", "cloudwatch").
            state: Optional alert state filter (e.g., "firing", "acknowledged").
            limit: Optional limit on number of alerts to return.

        Returns:
            Dict containing alert results with at minimum:
                - "alerts": List of alert dicts.
                - "total_count": Total number of alerts matching the query.

        Raises:
            Exception: If the query fails.
        """
        ...

    async def acknowledge_alert(
        self,
        alert_id: str,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Mark an alert as acknowledged.

        Args:
            alert_id: The unique identifier of the alert.
            note: Optional note to attach to the acknowledgement.

        Returns:
            Dict containing result with at minimum:
                - "acknowledged": Boolean indicating success.
                - "alert_id": The alert identifier.

        Raises:
            Exception: If the acknowledgement fails.
        """
        ...

    async def escalate_alert(
        self,
        alert_id: str,
        *,
        target: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Escalate an alert to a higher priority or team.

        Args:
            alert_id: The unique identifier of the alert.
            target: Optional team or individual to escalate to.
            note: Optional note for the escalation.

        Returns:
            Dict containing result with at minimum:
                - "escalated": Boolean indicating success.
                - "alert_id": The alert identifier.

        Raises:
            Exception: If the escalation fails.
        """
        ...

    async def silence_alert(
        self,
        alert_id: str,
        *,
        duration_minutes: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Silence an alert for a specified duration.

        Args:
            alert_id: The unique identifier of the alert.
            duration_minutes: Duration to silence the alert in minutes.
            reason: Optional reason for silencing.

        Returns:
            Dict containing result with at minimum:
                - "silenced": Boolean indicating success.
                - "alert_id": The alert identifier.
                - "silenced_until": ISO timestamp when silence expires.

        Raises:
            Exception: If silencing fails.
        """
        ...


class AlertingModule(BaseModule):
    """Alerting module using the pluggable adapter pattern.

    This module exposes alert management tools to the SafeAgent framework while
    delegating actual alerting operations to an injected backend provider.

    Attributes:
        _backend: The alerting backend implementation (private).
    """

    def __init__(self, backend: AlertingBackend) -> None:
        """Initialize the alerting module with a backend provider.

        Args:
            backend: An implementation of AlertingBackend that handles
                actual alerting operations for a specific platform.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the alerting module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="alerting",
            description="Manage alerts through an alerting system backend.",
            tools=[
                ToolDescriptor(
                    name="alerting:list_alerts",
                    description=(
                        "List alerts with optional filtering by severity and source."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": (
                                    "Filter by alert source "
                                    "(e.g., 'prometheus', 'cloudwatch')."
                                ),
                            },
                            "severity": {
                                "type": "string",
                                "description": (
                                    "Filter by severity (e.g., 'critical', 'warning')."
                                ),
                            },
                            "state": {
                                "type": "string",
                                "description": (
                                    "Filter by alert state "
                                    "(e.g., 'firing', 'acknowledged')."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "description": ("Maximum number of alerts to return."),
                            },
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="alerting:ListAlerts",
                    resource_param=["source"],
                    condition_keys=[
                        "alerting:Severity",
                        "alerting:AlertSource",
                    ],
                ),
                ToolDescriptor(
                    name="alerting:acknowledge_alert",
                    description="Acknowledge an alert by its ID.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "alert_id": {
                                "type": "string",
                                "description": ("The unique identifier of the alert."),
                            },
                            "note": {
                                "type": "string",
                                "description": (
                                    "Optional note to attach to the acknowledgement."
                                ),
                            },
                        },
                        "required": ["alert_id"],
                        "additionalProperties": False,
                    },
                    action="alerting:AcknowledgeAlert",
                    resource_param=["alert_id"],
                    condition_keys=[
                        "alerting:Severity",
                        "alerting:AlertSource",
                    ],
                ),
                ToolDescriptor(
                    name="alerting:escalate_alert",
                    description="Escalate an alert to a higher priority or team.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "alert_id": {
                                "type": "string",
                                "description": "The unique identifier of the alert.",
                            },
                            "target": {
                                "type": "string",
                                "description": (
                                    "Optional team or individual to escalate to."
                                ),
                            },
                            "note": {
                                "type": "string",
                                "description": "Optional note for the escalation.",
                            },
                        },
                        "required": ["alert_id"],
                        "additionalProperties": False,
                    },
                    action="alerting:EscalateAlert",
                    resource_param=["alert_id"],
                    condition_keys=[
                        "alerting:Severity",
                        "alerting:AlertSource",
                    ],
                ),
                ToolDescriptor(
                    name="alerting:silence_alert",
                    description="Silence an alert for a specified duration.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "alert_id": {
                                "type": "string",
                                "description": ("The unique identifier of the alert."),
                            },
                            "duration": {
                                "type": "string",
                                "description": (
                                    "Duration to silence the alert (e.g., '1h', '30m')."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": ("Optional reason for silencing."),
                            },
                        },
                        "required": ["alert_id", "duration"],
                        "additionalProperties": False,
                    },
                    action="alerting:SilenceAlert",
                    resource_param=["alert_id"],
                    condition_keys=[
                        "alerting:Severity",
                        "alerting:AlertSource",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve alerting condition values from parameters.

        Derives condition keys for policy evaluation:
            - alerting:Severity: The severity filter (for list_alerts) or
              alert severity (for other tools, if available).
            - alerting:AlertSource: The source filter (for list_alerts) or
              alert source (for other tools, if available).

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        conditions: dict[str, Any] = {}

        # For list_alerts, derive from filter parameters
        severity = params.get("severity")
        if severity is not None:
            conditions["alerting:Severity"] = str(severity)

        source = params.get("source")
        if source is not None:
            conditions["alerting:AlertSource"] = str(source)

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute an alerting tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("alerting:")
            if normalized_tool == "list_alerts":
                return await self._list_alerts(params)
            if normalized_tool == "acknowledge_alert":
                return await self._acknowledge_alert(params)
            if normalized_tool == "escalate_alert":
                return await self._escalate_alert(params)
            if normalized_tool == "silence_alert":
                return await self._silence_alert(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _list_alerts(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate list_alerts to the backend.

        Security Note:
            Logs the query parameters at DEBUG level. Alert content is NOT logged.
        """
        kwargs: dict[str, Any] = {}
        if "severity" in params:
            kwargs["severity"] = str(params["severity"])
        if "source" in params:
            kwargs["source"] = str(params["source"])
        if "state" in params:
            kwargs["state"] = str(params["state"])
        if "limit" in params:
            kwargs["limit"] = int(params["limit"])

        logger.debug("Alerting list_alerts: %s", kwargs)

        result = await self._backend.list_alerts(**kwargs)
        return ToolResult(success=True, data=result)

    async def _acknowledge_alert(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate acknowledge_alert to the backend.

        Security Note:
            Logs the alert_id at INFO level for audit trail.
            Note content is logged at DEBUG level to avoid logging PII.
        """
        alert_id = params.get("alert_id")
        if alert_id is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: alert_id",
            )
        alert_id = str(alert_id)
        if not alert_id or not alert_id.strip():
            return ToolResult(
                success=False,
                error="alert_id must not be empty",
            )

        kwargs: dict[str, Any] = {}
        if "note" in params:
            kwargs["note"] = str(params["note"])

        logger.info("Alerting acknowledge_alert: alert_id=%s", alert_id)
        logger.debug("Alerting acknowledge_alert kwargs: %s", kwargs)

        result = await self._backend.acknowledge_alert(alert_id, **kwargs)
        return ToolResult(success=True, data=result)

    async def _escalate_alert(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate escalate_alert to the backend.

        Security Note:
            Logs the alert_id at INFO level for audit trail.
            Escalation details are logged at DEBUG level.
        """
        alert_id = params.get("alert_id")
        if alert_id is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: alert_id",
            )
        alert_id = str(alert_id)
        if not alert_id or not alert_id.strip():
            return ToolResult(
                success=False,
                error="alert_id must not be empty",
            )

        kwargs: dict[str, Any] = {}
        if "target" in params:
            kwargs["target"] = str(params["target"])
        if "note" in params:
            kwargs["note"] = str(params["note"])

        logger.info("Alerting escalate_alert: alert_id=%s", alert_id)
        logger.debug("Alerting escalate_alert kwargs: %s", kwargs)

        result = await self._backend.escalate_alert(alert_id, **kwargs)
        return ToolResult(success=True, data=result)

    async def _silence_alert(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate silence_alert to the backend.

        Security Note:
            Logs the alert_id and duration at INFO level for audit trail.
            Reason is logged at DEBUG level to avoid logging PII.
        """
        alert_id = params.get("alert_id")
        if alert_id is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: alert_id",
            )
        alert_id = str(alert_id)
        if not alert_id or not alert_id.strip():
            return ToolResult(
                success=False,
                error="alert_id must not be empty",
            )

        duration = params.get("duration")
        if duration is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: duration",
            )
        duration_str = str(duration)
        duration_minutes = self._parse_duration(duration_str)
        if duration_minutes is None:
            return ToolResult(
                success=False,
                error=(
                    "Invalid duration format. Use formats like '1h', '30m', '1h30m'."
                ),
            )
        if duration_minutes < 1:
            return ToolResult(
                success=False,
                error="duration must be at least 1 minute",
            )

        kwargs: dict[str, Any] = {"duration_minutes": duration_minutes}
        if "reason" in params:
            kwargs["reason"] = str(params["reason"])

        logger.info(
            "Alerting silence_alert: alert_id=%s duration=%s (%d minutes)",
            alert_id,
            duration_str,
            duration_minutes,
        )
        logger.debug("Alerting silence_alert kwargs: %s", kwargs)

        result = await self._backend.silence_alert(alert_id, **kwargs)
        return ToolResult(success=True, data=result)

    def _parse_duration(self, duration: str) -> int | None:
        """Parse a duration string into minutes.

        Supports formats like:
        - "30m" - 30 minutes
        - "1h" - 1 hour (60 minutes)
        - "1h30m" - 1 hour 30 minutes (90 minutes)
        - "2h15m" - 2 hours 15 minutes (135 minutes)

        Args:
            duration: The duration string to parse.

        Returns:
            The duration in minutes, or None if parsing fails.
        """
        import re

        duration = duration.strip().lower()
        if not duration:
            return None

        total_minutes = 0
        pattern = re.compile(r"^(\d+h)?(\d+m)?$")

        # Handle combined format like "1h30m"
        match = pattern.match(duration)
        if match:
            hours_part = match.group(1)
            minutes_part = match.group(2)

            if hours_part:
                total_minutes += int(hours_part[:-1]) * 60
            if minutes_part:
                total_minutes += int(minutes_part[:-1])

            return total_minutes if total_minutes > 0 else None

        # Handle simple integer string (treat as minutes)
        if duration.isdigit():
            return int(duration)

        return None
