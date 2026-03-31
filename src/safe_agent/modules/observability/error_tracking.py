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

"""Pluggable error tracking module for SafeAgent.

This module provides an error tracking interface that delegates to an injected
backend provider. The framework defines the interface; a plugin provides the
concrete implementation (Sentry, Rollbar, Bugsnag, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

# Note: This module is now at safe_agent.modules.observability.error_tracking
# but imports from safe_agent.modules.base remain unchanged since base.py
# is in the parent modules/ directory.


@runtime_checkable
class ErrorTrackingBackend(Protocol):
    """Protocol for error tracking backend implementations.

    Implementations provide concrete integrations with error tracking platforms
    such as Sentry, Rollbar, Bugsnag, etc.

    Methods:
        query_errors: Query errors for a project with optional filters.
    """

    async def query_errors(
        self,
        project: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query errors for a project.

        Args:
            project: Project identifier to query errors for.
            **kwargs: Optional filters (query, time_range, limit, etc.).

        Returns:
            List of error dicts with details (id, type, message, count, etc.).
        """
        ...


class ErrorTrackingModule(BaseModule):
    """Error tracking module using the pluggable adapter pattern.

    This module exposes error tracking tools to the SafeAgent framework while
    delegating actual error queries to an injected backend provider.

    Attributes:
        _backend: The error tracking backend implementation (private).
    """

    def __init__(self, backend: ErrorTrackingBackend) -> None:
        """Initialize the error tracking module with a backend provider.

        Args:
            backend: An implementation of ErrorTrackingBackend that handles
                actual error queries for a specific platform.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the error tracking module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="error_tracking",
            description="Query and analyze errors from error tracking platforms.",
            tools=[
                ToolDescriptor(
                    name="error_tracking:query_errors",
                    description="Query errors for a project with optional filters.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "project": {
                                "type": "string",
                                "description": (
                                    "Project identifier to query errors for."
                                ),
                            },
                            "query": {
                                "type": "string",
                                "description": (
                                    "Optional search query to filter errors."
                                ),
                            },
                            "time_range": {
                                "type": "string",
                                "description": (
                                    "Optional time range filter (e.g., '24h', '7d')."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "Optional maximum number of errors to return."
                                ),
                            },
                        },
                        "required": ["project"],
                        "additionalProperties": False,
                    },
                    action="error_tracking:QueryErrors",
                    resource_param=["project"],
                    condition_keys=[
                        "error_tracking:Project",
                        "error_tracking:ErrorType",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve error tracking condition values from the parameters.

        Derives condition keys for policy evaluation:
            - error_tracking:Project: The project identifier.
            - error_tracking:ErrorType: Inferred from query if present.

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        project = params.get("project")
        if project is None:
            return {}

        project_str = str(project)
        conditions: dict[str, Any] = {
            "error_tracking:Project": project_str,
        }

        # Infer ErrorType from query if it contains type information.
        # This is a simple heuristic - backends may have more sophisticated
        # type detection.
        query = params.get("query")
        if query is not None and isinstance(query, str):
            # Look for common error type patterns in query
            # e.g., "type:TypeError" or "error_type:ValueError"
            query_lower = query.lower()
            for prefix in ("type:", "error_type:", "error:"):
                if prefix in query_lower:
                    # Extract the type value after the prefix
                    start = query_lower.find(prefix) + len(prefix)
                    remaining = query[start:]
                    # Extract until next space or end
                    end = remaining.find(" ")
                    if end == -1:
                        end = len(remaining)
                    error_type = remaining[:end].strip()
                    if error_type:
                        conditions["error_tracking:ErrorType"] = error_type
                        break

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute an error tracking tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("error_tracking:")
            if normalized_tool == "query_errors":
                return await self._query_errors(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _query_errors(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate error querying to the backend."""
        project = str(params["project"])
        kwargs: dict[str, Any] = {}
        if "query" in params:
            kwargs["query"] = params["query"]
        if "time_range" in params:
            kwargs["time_range"] = params["time_range"]
        if "limit" in params:
            kwargs["limit"] = params["limit"]

        errors = await self._backend.query_errors(project, **kwargs)
        return ToolResult(success=True, data={"errors": errors})
