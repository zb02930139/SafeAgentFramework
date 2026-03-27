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

"""Dashboard module for SafeAgent - pluggable adapter for dashboard panels."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)


class DashboardBackend(Protocol):
    """Protocol for dashboard backend implementations.

    Concrete implementations provide integration with specific monitoring
    platforms (Grafana, Datadog, etc.).
    """

    async def get_panel(
        self,
        dashboard_id: str,
        panel_id: str,
        **kwargs: Any,
    ) -> dict:
        """Retrieve a specific panel's data or snapshot.

        Args:
            dashboard_id: The identifier of the dashboard.
            panel_id: The identifier of the panel within the dashboard.
            **kwargs: Optional parameters including:
                - time_range: Optional time range string (e.g., "7d", "90d")
                - format: Optional output format ("data" or "snapshot")

        Returns:
            Dict containing panel data or snapshot information.
        """
        ...

    async def list_dashboards(self, **kwargs: Any) -> list[dict]:
        """List available dashboards.

        Args:
            **kwargs: Optional parameters including:
                - filter: Optional string filter for dashboard names
                - tags: Optional list of tags to filter by

        Returns:
            List of dashboard metadata dicts.
        """
        ...


class DashboardModule(BaseModule):
    """Dashboard module providing access to curated dashboard panels.

    This module uses the pluggable adapter pattern - the framework provides
    the interface, while a plugin provides the concrete monitoring platform
    integration (Grafana, Datadog, etc.).
    """

    def __init__(self, backend: DashboardBackend) -> None:
        """Initialize the module with a dashboard backend.

        Args:
            backend: A concrete implementation of the DashboardBackend protocol.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the dashboard module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="dashboard",
            description="Access to curated dashboard panels for trend analysis.",
            tools=[
                ToolDescriptor(
                    name="dashboard:get_panel",
                    description="Retrieve a specific panel's data or snapshot.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "dashboard_id": {"type": "string"},
                            "panel_id": {"type": "string"},
                            "time_range": {
                                "type": "string",
                                "description": "Optional time range (e.g. '7d', '90d')",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["data", "snapshot"],
                                "description": "Output format: 'data' or 'snapshot'",
                            },
                        },
                        "required": ["dashboard_id", "panel_id"],
                        "additionalProperties": False,
                    },
                    action="dashboard:GetPanel",
                    resource_param=["dashboard_id"],
                    condition_keys=["dashboard:DashboardId"],
                ),
                ToolDescriptor(
                    name="dashboard:list_dashboards",
                    description="List available dashboards.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "filter": {
                                "type": "string",
                                "description": "Optional filter for dashboard names",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags to filter dashboards",
                            },
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    action="dashboard:ListDashboards",
                    resource_param=["*"],
                    condition_keys=[],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve dashboard condition values from parameters.

        Derives dashboard:DashboardId from the dashboard_id parameter when
        available (get_panel tool). list_dashboards has no specific resource.
        """
        dashboard_id = params.get("dashboard_id")
        if dashboard_id is not None:
            return {"dashboard:DashboardId": str(dashboard_id)}
        return {}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Execute a dashboard tool invocation."""
        try:
            normalized_tool = tool_name.removeprefix("dashboard:")
            if normalized_tool == "get_panel":
                return await self._get_panel(params)
            if normalized_tool == "list_dashboards":
                return await self._list_dashboards(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            logger.exception("Dashboard tool execution failed: %s", tool_name)
            return ToolResult(success=False, error=str(exc))

    async def _get_panel(self, params: dict[str, Any]) -> ToolResult:
        """Delegate to backend to retrieve a panel."""
        dashboard_id = str(params["dashboard_id"])
        panel_id = str(params["panel_id"])
        time_range = params.get("time_range")
        format_val = params.get("format")

        kwargs: dict[str, Any] = {}
        if time_range is not None:
            kwargs["time_range"] = str(time_range)
        if format_val is not None:
            kwargs["format"] = str(format_val)

        result = await self._backend.get_panel(dashboard_id, panel_id, **kwargs)
        return ToolResult(success=True, data=result)

    async def _list_dashboards(self, params: dict[str, Any]) -> ToolResult:
        """Delegate to backend to list dashboards."""
        filter_val = params.get("filter")
        tags = params.get("tags")

        kwargs: dict[str, Any] = {}
        if filter_val is not None:
            kwargs["filter"] = str(filter_val)
        if tags is not None:
            kwargs["tags"] = list(tags)

        result = await self._backend.list_dashboards(**kwargs)
        return ToolResult(success=True, data={"dashboards": result})
