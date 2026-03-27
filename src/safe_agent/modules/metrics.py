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

"""Pluggable metrics module for SafeAgent.

This module provides a stable IAM interface for querying metrics backends
(Prometheus, Datadog, etc.) via the pluggable adapter pattern. The framework
defines the namespace, tool descriptors, and condition keys; a backend plugin
provides the concrete implementation.
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

logger = logging.getLogger(__name__)

__all__ = ["MetricsBackend", "MetricsModule"]


@runtime_checkable
class MetricsBackend(Protocol):
    """Protocol for metrics backend implementations.

    Backend implementations must provide this async method to execute
    metric queries. The framework's MetricsModule delegates execute() calls
    to the injected backend.
    """

    async def query_metrics(
        self,
        datasource: str,
        query: str,
        start: str,
        end: str,
        step: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a metric query against the backend.

        Args:
            datasource: The metrics datasource identifier.
            query: The query string (e.g., PromQL).
            start: Start time as ISO 8601 string.
            end: End time as ISO 8601 string.
            step: Query resolution step (e.g., "15s", "1m").
            **kwargs: Additional backend-specific options.

        Returns:
            A dict containing query results. Structure depends on backend.
        """
        ...


class MetricsModule(BaseModule):
    """Framework-defined metrics interface with pluggable backend adapter.

    This module provides a stable IAM surface for metrics operations. Policies
    are written against the metrics: namespace regardless of which backend
    (Prometheus, Datadog, etc.) is configured. The backend is injected at
    construction time.

    Example:
        >>> from my_prometheus_adapter import PrometheusBackend
        >>> backend = PrometheusBackend(url="http://prometheus:9090")
        >>> module = MetricsModule(backend)
        >>> result = await module.execute(
        ...     "metrics:query_metrics",
        ...     {
        ...         "datasource": "prometheus",
        ...         "query": "up",
        ...         "start": "2024-01-01T00:00:00Z",
        ...         "end": "2024-01-01T01:00:00Z",
        ...     }
        ... )
    """

    def __init__(self, backend: MetricsBackend) -> None:
        """Initialize the module with a metrics backend.

        Args:
            backend: An object implementing the MetricsBackend protocol.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the metrics module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="metrics",
            description="Submit structured metric queries for short time range data.",
            tools=[
                ToolDescriptor(
                    name="metrics:query_metrics",
                    description="Execute a metric query (PromQL, etc.)",
                    parameters={
                        "type": "object",
                        "properties": {
                            "datasource": {"type": "string"},
                            "query": {"type": "string"},
                            "start": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                            "end": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                            "step": {
                                "type": "string",
                                "description": "e.g., '15s', '1m'",
                            },
                        },
                        "required": ["datasource", "query", "start", "end"],
                        "additionalProperties": False,
                    },
                    action="metrics:QueryMetrics",
                    resource_param=["datasource"],
                    condition_keys=[
                        "metrics:QueryLanguage",
                        "metrics:TimeRange",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve metrics condition values from query parameters.

        Derives condition keys for policy evaluation:
        - metrics:QueryLanguage: Detected from query syntax (promql, etc.)
        - metrics:TimeRange: Calculated from start/end timestamps

        Args:
            tool_name: The tool being invoked (e.g., "metrics:query_metrics").
            params: The input parameters for the invocation.

        Returns:
            A dict mapping condition key names to resolved values.
        """
        if tool_name != "metrics:query_metrics":
            return {}

        conditions: dict[str, Any] = {}

        # Derive QueryLanguage from query syntax heuristics
        query = params.get("query", "")
        conditions["metrics:QueryLanguage"] = self._detect_query_language(query)

        # Derive TimeRange from start/end
        start = params.get("start")
        end = params.get("end")
        if start and end:
            conditions["metrics:TimeRange"] = self._compute_time_range(start, end)

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a metrics tool invocation by delegating to the backend.

        Args:
            tool_name: The tool to execute.
            params: The input parameters.

        Returns:
            A ToolResult with the query response or error.
        """
        try:
            if tool_name != "metrics:query_metrics":
                return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

            result = await self._backend.query_metrics(
                datasource=params["datasource"],
                query=params["query"],
                start=params["start"],
                end=params["end"],
                step=params.get("step"),
            )
            return ToolResult(success=True, data=result)
        except KeyError as exc:
            return ToolResult(success=False, error=f"Missing required parameter: {exc}")
        except Exception as exc:
            logger.exception("Metrics backend query failed")
            return ToolResult(success=False, error=str(exc))

    def _detect_query_language(self, query: str) -> str:
        """Detect the query language from syntax heuristics.

        This is a simple heuristic-based detection. Backends may override
        or provide more sophisticated detection.

        Args:
            query: The query string to analyze.

        Returns:
            A language identifier string (e.g., "promql", "influxql", "unknown").
        """
        query_lower = query.lower().strip()

        # PromQL heuristics
        if any(
            keyword in query_lower
            for keyword in [
                "rate(",
                "irate(",
                "increase(",
                "sum(",
                "avg(",
                "histogram_quantile",
            ]
        ):
            return "promql"
        if " by (" in query_lower or " by(" in query_lower:
            return "promql"
        if query_lower.startswith(
            ("sum(", "avg(", "min(", "max(", "count(", "rate(", "irate(")
        ):
            return "promql"

        # InfluxQL heuristics
        if query_lower.startswith("select "):
            return "influxql"
        if " from " in query_lower and " where " in query_lower:
            return "influxql"
        if query_lower.startswith("from ") and " where " in query_lower:
            return "influxql"

        # Default to unknown for unrecognized syntax
        return "unknown"

    def _compute_time_range(self, start: str, end: str) -> str:
        """Compute a human-readable time range from ISO timestamps.

        Args:
            start: ISO 8601 start timestamp.
            end: ISO 8601 end timestamp.

        Returns:
            A string describing the time range (e.g., "1h", "30m", "2d").
        """
        from datetime import datetime

        try:
            # Parse ISO 8601 timestamps (basic handling)
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            delta_seconds = int((end_dt - start_dt).total_seconds())

            if delta_seconds < 0:
                return "invalid"

            # Convert to human-readable format
            if delta_seconds < 60:
                return f"{delta_seconds}s"
            if delta_seconds < 3600:
                return f"{delta_seconds // 60}m"
            if delta_seconds < 86400:
                return f"{delta_seconds // 3600}h"
            return f"{delta_seconds // 86400}d"
        except (ValueError, TypeError):
            return "unknown"
