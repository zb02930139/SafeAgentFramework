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

"""Pluggable logging interface module for SafeAgent.

This module provides a logging interface that delegates to pluggable backends
(syslog, Splunk, Elasticsearch, Loki, etc.). The framework defines the interface
and IAM surface; the backend provides the concrete implementation.
"""

from __future__ import annotations

import logging as stdlib_logging
from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = stdlib_logging.getLogger(__name__)


@runtime_checkable
class LoggingBackend(Protocol):
    """Protocol for pluggable logging backend implementations.

    Adapters must implement this protocol to provide logging functionality.
    The interface module delegates execute() calls to the backend.
    """

    async def query_logs(
        self,
        source: str,
        query: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query log entries from a logging system.

        Args:
            source: The log source identifier (e.g., "app", "system", "audit").
            query: Query string in the backend's native format.
            start: ISO 8601 timestamp for query start time.
            end: ISO 8601 timestamp for query end time.
            **kwargs: Additional backend-specific parameters (e.g., limit).

        Returns:
            A list of log entry dictionaries matching the query.
        """
        ...

    async def write_log(
        self,
        source: str,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write a log entry to a logging system.

        Args:
            source: The log source identifier.
            level: Log severity level (e.g., "info", "warning", "error").
            message: The log message content.
            **kwargs: Additional backend-specific parameters (e.g., metadata).

        Returns:
            A dictionary containing the result of the write operation.
        """
        ...


class LoggingModule(BaseModule):
    """Framework-defined logging interface with pluggable backend.

    This module provides a stable IAM surface for logging operations.
    The backend is injected at construction and handles actual execution.
    """

    def __init__(self, backend: LoggingBackend) -> None:
        """Initialize the logging module with a backend implementation.

        Args:
            backend: A LoggingBackend implementation that provides the actual
                logging functionality.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the logging module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="logging",
            description="Query and write to logging systems.",
            tools=[
                ToolDescriptor(
                    name="logging:query_logs",
                    description="Search log entries.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "query": {"type": "string"},
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["source", "query", "start", "end"],
                        "additionalProperties": False,
                    },
                    action="logging:QueryLogs",
                    resource_param=["source"],
                    condition_keys=["logging:LogSource"],
                ),
                ToolDescriptor(
                    name="logging:write_log",
                    description="Write a log entry.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "level": {"type": "string"},
                            "message": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["source", "level", "message"],
                        "additionalProperties": False,
                    },
                    action="logging:WriteLog",
                    resource_param=["source"],
                    condition_keys=["logging:LogSource", "logging:Severity"],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve logging condition values from tool parameters.

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the tool.

        Returns:
            A dict mapping condition key names to their resolved values.
        """
        conditions: dict[str, Any] = {}

        source = params.get("source")
        if source is not None:
            conditions["logging:LogSource"] = str(source)

        # logging:Severity is derived from level parameter (for write_log)
        level = params.get("level")
        if level is not None:
            conditions["logging:Severity"] = str(level)

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a logging tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("logging:")

            if normalized_tool == "query_logs":
                return await self._execute_query_logs(params)
            if normalized_tool == "write_log":
                return await self._execute_write_log(params)

            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            # Propagate backend errors as failed ToolResult
            logger.exception("Logging backend error for %s", tool_name)
            return ToolResult(success=False, error=str(exc))

    async def _execute_query_logs(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Execute query_logs by delegating to the backend."""
        source = str(params["source"])
        query = str(params["query"])
        start = str(params["start"])
        end = str(params["end"])

        # Extract optional limit parameter
        kwargs: dict[str, Any] = {}
        if "limit" in params:
            kwargs["limit"] = params["limit"]

        result = await self._backend.query_logs(
            source=source,
            query=query,
            start=start,
            end=end,
            **kwargs,
        )
        return ToolResult(success=True, data={"entries": result})

    async def _execute_write_log(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Execute write_log by delegating to the backend."""
        source = str(params["source"])
        level = str(params["level"])
        message = str(params["message"])

        # Extract optional metadata parameter
        kwargs: dict[str, Any] = {}
        if "metadata" in params:
            kwargs["metadata"] = params["metadata"]

        result = await self._backend.write_log(
            source=source,
            level=level,
            message=message,
            **kwargs,
        )
        return ToolResult(success=True, data=result)
