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

"""Calendar module for SafeAgent using the pluggable adapter pattern."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


@runtime_checkable
class CalendarBackend(Protocol):
    """Protocol for calendar backend implementations.

    Concrete backends (Google Calendar, Microsoft Outlook, etc.) must implement
    this protocol. The CalendarModule delegates all tool execution to the backend.
    """

    async def create_event(
        self,
        calendar_id: str,
        title: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a calendar event.

        Args:
            calendar_id: Identifier for the target calendar.
            title: Event title/summary.
            start: ISO 8601 start datetime string.
            end: ISO 8601 end datetime string.
            **kwargs: Additional optional parameters (description, attendees, etc.).

        Returns:
            Dict containing the created event details (provider-specific).
        """
        ...

    async def list_events(
        self,
        calendar_id: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List events in a time range.

        Args:
            calendar_id: Identifier for the target calendar.
            start: ISO 8601 start datetime string.
            end: ISO 8601 end datetime string.
            **kwargs: Additional optional parameters.

        Returns:
            List of event dicts (provider-specific format).
        """
        ...

    async def check_conflicts(
        self,
        calendar_id: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Check for scheduling conflicts in a time window.

        Args:
            calendar_id: Identifier for the target calendar.
            start: ISO 8601 start datetime string.
            end: ISO 8601 end datetime string.
            **kwargs: Additional optional parameters.

        Returns:
            Dict containing conflict information (provider-specific format).
        """
        ...


class CalendarModule(BaseModule):
    """Calendar module using the pluggable adapter pattern.

    The framework ships this interface; a plugin provides the concrete backend
    implementation (Google Calendar, Microsoft Outlook, etc.).
    """

    def __init__(self, backend: CalendarBackend) -> None:
        """Initialize the calendar module with a backend implementation.

        Args:
            backend: A CalendarBackend-compliant object that provides the
                concrete calendar operations.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the calendar module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="calendar",
            description=(
                "Manage maintenance windows, schedule events, and detect conflicts."
            ),
            tools=[
                ToolDescriptor(
                    name="calendar:create_event",
                    description="Create a calendar event.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "calendar_id": {"type": "string"},
                            "title": {"type": "string"},
                            "start": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                            "end": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                            "description": {"type": "string"},
                            "attendees": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["calendar_id", "title", "start", "end"],
                        "additionalProperties": False,
                    },
                    action="calendar:CreateEvent",
                    resource_param=["calendar_id"],
                    condition_keys=["calendar:CalendarId"],
                ),
                ToolDescriptor(
                    name="calendar:list_events",
                    description="List events in a time range.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "calendar_id": {"type": "string"},
                            "start": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                            "end": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                        },
                        "required": ["calendar_id", "start", "end"],
                        "additionalProperties": False,
                    },
                    action="calendar:ListEvents",
                    resource_param=["calendar_id"],
                    condition_keys=["calendar:CalendarId"],
                ),
                ToolDescriptor(
                    name="calendar:check_conflicts",
                    description="Check for scheduling conflicts in a window.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "calendar_id": {"type": "string"},
                            "start": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                            "end": {
                                "type": "string",
                                "description": "ISO 8601 datetime",
                            },
                        },
                        "required": ["calendar_id", "start", "end"],
                        "additionalProperties": False,
                    },
                    action="calendar:CheckConflicts",
                    resource_param=["calendar_id"],
                    condition_keys=["calendar:CalendarId"],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve calendar condition values from the calendar_id parameter."""
        calendar_id = params.get("calendar_id")
        if calendar_id is None:
            return {}
        return {"calendar:CalendarId": str(calendar_id)}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a calendar tool invocation by delegating to the backend."""
        try:
            normalized_tool = tool_name.removeprefix("calendar:")
            if normalized_tool == "create_event":
                return await self._create_event(params)
            if normalized_tool == "list_events":
                return await self._list_events(params)
            if normalized_tool == "check_conflicts":
                return await self._check_conflicts(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _create_event(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate create_event to the backend."""
        calendar_id = str(params.get("calendar_id", ""))
        title = str(params.get("title", ""))
        start = str(params.get("start", ""))
        end = str(params.get("end", ""))

        # Extract optional parameters
        kwargs: dict[str, Any] = {}
        if "description" in params:
            kwargs["description"] = params["description"]
        if "attendees" in params:
            kwargs["attendees"] = params["attendees"]

        result = await self._backend.create_event(
            calendar_id=calendar_id,
            title=title,
            start=start,
            end=end,
            **kwargs,
        )
        return ToolResult(success=True, data=result)

    async def _list_events(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate list_events to the backend."""
        calendar_id = str(params.get("calendar_id", ""))
        start = str(params.get("start", ""))
        end = str(params.get("end", ""))

        # Extract optional parameters for backend extensibility
        kwargs: dict[str, Any] = {}
        for key in params:
            if key not in ("calendar_id", "start", "end"):
                kwargs[key] = params[key]

        result = await self._backend.list_events(
            calendar_id=calendar_id,
            start=start,
            end=end,
            **kwargs,
        )
        return ToolResult(success=True, data={"events": result})

    async def _check_conflicts(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate check_conflicts to the backend."""
        calendar_id = str(params.get("calendar_id", ""))
        start = str(params.get("start", ""))
        end = str(params.get("end", ""))

        # Extract optional parameters for backend extensibility
        kwargs: dict[str, Any] = {}
        for key in params:
            if key not in ("calendar_id", "start", "end"):
                kwargs[key] = params[key]

        result = await self._backend.check_conflicts(
            calendar_id=calendar_id,
            start=start,
            end=end,
            **kwargs,
        )
        return ToolResult(success=True, data=result)
