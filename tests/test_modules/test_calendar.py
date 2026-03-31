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

"""Tests for the calendar module."""

from typing import Any

from safe_agent.modules.communication.calendar import CalendarBackend, CalendarModule


class MockCalendarBackend:
    """Mock calendar backend for testing."""

    def __init__(self) -> None:
        """Initialize mock backend with call tracking."""
        self.calls: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._should_fail: bool = False

    def set_fail_mode(self, should_fail: bool) -> None:
        """Configure the mock to raise exceptions."""
        self._should_fail = should_fail

    async def create_event(
        self,
        calendar_id: str,
        title: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock create_event implementation."""
        if self._should_fail:
            raise RuntimeError("Backend failure")
        event = {
            "id": f"event-{len(self._events) + 1}",
            "calendar_id": calendar_id,
            "title": title,
            "start": start,
            "end": end,
            **kwargs,
        }
        self._events.append(event)
        self.calls.append(
            {
                "method": "create_event",
                "calendar_id": calendar_id,
                "title": title,
                "start": start,
                "end": end,
                "kwargs": kwargs,
            }
        )
        return event

    async def list_events(
        self,
        calendar_id: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Mock list_events implementation."""
        if self._should_fail:
            raise RuntimeError("Backend failure")
        self.calls.append(
            {
                "method": "list_events",
                "calendar_id": calendar_id,
                "start": start,
                "end": end,
                "kwargs": kwargs,
            }
        )
        return [
            e
            for e in self._events
            if e["calendar_id"] == calendar_id and start <= e["start"] < end
        ]

    async def check_conflicts(
        self,
        calendar_id: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock check_conflicts implementation."""
        if self._should_fail:
            raise RuntimeError("Backend failure")
        self.calls.append(
            {
                "method": "check_conflicts",
                "calendar_id": calendar_id,
                "start": start,
                "end": end,
                "kwargs": kwargs,
            }
        )
        conflicts = [
            e
            for e in self._events
            if e["calendar_id"] == calendar_id
            and not (end <= e["start"] or start >= e["end"])
        ]
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
        }


class TestCalendarModuleDescriptor:
    """Tests for CalendarModule.describe()."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "calendar"
        assert "maintenance windows" in descriptor.description.lower()
        assert "schedule events" in descriptor.description.lower()
        assert "conflicts" in descriptor.description.lower()
        assert len(descriptor.tools) == 3

    def test_tool_names_are_correct(self) -> None:
        """All tool names should follow the namespace:snake_case format."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()
        tool_names = {tool.name for tool in descriptor.tools}

        assert tool_names == {
            "calendar:create_event",
            "calendar:list_events",
            "calendar:check_conflicts",
        }

    def test_tool_actions_are_correct(self) -> None:
        """All tool actions should follow the namespace:PascalCase format."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()
        actions = {tool.action for tool in descriptor.tools}

        assert actions == {
            "calendar:CreateEvent",
            "calendar:ListEvents",
            "calendar:CheckConflicts",
        }

    def test_all_tools_declare_calendar_id_resource_param(self) -> None:
        """All tools should declare calendar_id as their resource param."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.resource_param == ["calendar_id"]

    def test_all_tools_declare_calendar_condition_key(self) -> None:
        """All tools should declare calendar:CalendarId condition key."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.condition_keys == ["calendar:CalendarId"]

    def test_create_event_parameters_schema(self) -> None:
        """create_event should have correct parameter schema."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()
        create_tool = next(
            t for t in descriptor.tools if t.name == "calendar:create_event"
        )

        assert create_tool.parameters["additionalProperties"] is False
        props = create_tool.parameters["properties"]
        assert "calendar_id" in props
        assert "title" in props
        assert "start" in props
        assert "end" in props
        assert "description" in props
        assert "attendees" in props
        assert create_tool.parameters["required"] == [
            "calendar_id",
            "title",
            "start",
            "end",
        ]

    def test_list_events_parameters_schema(self) -> None:
        """list_events should have correct parameter schema."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()
        list_tool = next(
            t for t in descriptor.tools if t.name == "calendar:list_events"
        )

        assert list_tool.parameters["additionalProperties"] is False
        props = list_tool.parameters["properties"]
        assert set(props.keys()) == {"calendar_id", "start", "end"}
        assert list_tool.parameters["required"] == ["calendar_id", "start", "end"]

    def test_check_conflicts_parameters_schema(self) -> None:
        """check_conflicts should have correct parameter schema."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        descriptor = module.describe()
        check_tool = next(
            t for t in descriptor.tools if t.name == "calendar:check_conflicts"
        )

        assert check_tool.parameters["additionalProperties"] is False
        props = check_tool.parameters["properties"]
        assert set(props.keys()) == {"calendar_id", "start", "end"}
        assert check_tool.parameters["required"] == ["calendar_id", "start", "end"]


class TestCalendarModuleResolveConditions:
    """Tests for CalendarModule.resolve_conditions()."""

    async def test_resolve_conditions_derives_calendar_id(self) -> None:
        """resolve_conditions should derive calendar:CalendarId from params."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        conditions = await module.resolve_conditions(
            "calendar:create_event",
            {
                "calendar_id": "primary",
                "title": "Meeting",
                "start": "2026-01-01T10:00:00Z",
                "end": "2026-01-01T11:00:00Z",
            },
        )

        assert conditions == {"calendar:CalendarId": "primary"}

    async def test_resolve_conditions_handles_string_calendar_id(self) -> None:
        """resolve_conditions should convert calendar_id to string."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        conditions = await module.resolve_conditions(
            "calendar:list_events",
            {
                "calendar_id": "work-calendar",
                "start": "2026-01-01",
                "end": "2026-01-31",
            },
        )

        assert conditions == {"calendar:CalendarId": "work-calendar"}

    async def test_resolve_conditions_returns_empty_dict_without_calendar_id(
        self,
    ) -> None:
        """resolve_conditions should return empty dict if calendar_id is missing."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        conditions = await module.resolve_conditions(
            "calendar:check_conflicts",
            {"start": "2026-01-01", "end": "2026-01-31"},
        )

        assert conditions == {}


class TestCalendarModuleExecute:
    """Tests for CalendarModule.execute()."""

    async def test_execute_delegates_create_event_to_backend(self) -> None:
        """execute() should delegate create_event to the backend."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        result = await module.execute(
            "calendar:create_event",
            {
                "calendar_id": "primary",
                "title": "Team Standup",
                "start": "2026-01-01T09:00:00Z",
                "end": "2026-01-01T09:30:00Z",
            },
        )

        assert result.success is True
        assert result.data["title"] == "Team Standup"
        assert result.data["calendar_id"] == "primary"
        assert len(backend.calls) == 1
        assert backend.calls[0]["method"] == "create_event"

    async def test_execute_delegates_create_event_with_optional_params(self) -> None:
        """create_event should pass optional params to backend."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        result = await module.execute(
            "calendar:create_event",
            {
                "calendar_id": "primary",
                "title": "Sprint Planning",
                "start": "2026-01-01T10:00:00Z",
                "end": "2026-01-01T12:00:00Z",
                "description": "Quarterly planning session",
                "attendees": ["alice@example.com", "bob@example.com"],
            },
        )

        assert result.success is True
        assert result.data["description"] == "Quarterly planning session"
        assert result.data["attendees"] == ["alice@example.com", "bob@example.com"]

    async def test_execute_delegates_list_events_to_backend(self) -> None:
        """execute() should delegate list_events to the backend."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        # Create an event first
        await module.execute(
            "calendar:create_event",
            {
                "calendar_id": "primary",
                "title": "Test Event",
                "start": "2026-01-15T10:00:00Z",
                "end": "2026-01-15T11:00:00Z",
            },
        )

        result = await module.execute(
            "calendar:list_events",
            {
                "calendar_id": "primary",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-02-01T00:00:00Z",
            },
        )

        assert result.success is True
        assert "events" in result.data
        assert len(result.data["events"]) == 1
        assert len(backend.calls) == 2
        assert backend.calls[1]["method"] == "list_events"

    async def test_execute_delegates_check_conflicts_to_backend(self) -> None:
        """execute() should delegate check_conflicts to the backend."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        # Create an event first
        await module.execute(
            "calendar:create_event",
            {
                "calendar_id": "primary",
                "title": "Existing Meeting",
                "start": "2026-01-15T10:00:00Z",
                "end": "2026-01-15T11:00:00Z",
            },
        )

        result = await module.execute(
            "calendar:check_conflicts",
            {
                "calendar_id": "primary",
                "start": "2026-01-15T10:30:00Z",
                "end": "2026-01-15T11:30:00Z",
            },
        )

        assert result.success is True
        assert result.data["has_conflicts"] is True
        assert len(result.data["conflicts"]) == 1

    async def test_execute_delegates_check_conflicts_no_conflicts(self) -> None:
        """check_conflicts should return no conflicts for free time slots."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        result = await module.execute(
            "calendar:check_conflicts",
            {
                "calendar_id": "primary",
                "start": "2026-01-15T10:00:00Z",
                "end": "2026-01-15T11:00:00Z",
            },
        )

        assert result.success is True
        assert result.data["has_conflicts"] is False
        assert result.data["conflicts"] == []

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """execute() should return error for unknown tool names."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        result = await module.execute(
            "calendar:unknown_tool",
            {"calendar_id": "primary"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_execute_propagates_backend_errors(self) -> None:
        """execute() should propagate backend exceptions as ToolResult errors."""
        backend = MockCalendarBackend()
        backend.set_fail_mode(True)
        module = CalendarModule(backend)

        result = await module.execute(
            "calendar:create_event",
            {
                "calendar_id": "primary",
                "title": "Failing Event",
                "start": "2026-01-01T10:00:00Z",
                "end": "2026-01-01T11:00:00Z",
            },
        )

        assert result.success is False
        assert "Backend failure" in result.error


class TestCalendarBackendProtocol:
    """Tests for CalendarBackend protocol compliance."""

    def test_mock_backend_is_calendar_backend(self) -> None:
        """MockCalendarBackend should satisfy the CalendarBackend protocol."""
        backend = MockCalendarBackend()
        assert isinstance(backend, CalendarBackend)

    def test_module_accepts_protocol_compliant_backend(self) -> None:
        """CalendarModule should accept any CalendarBackend-compliant object."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)
        assert module._backend is backend


class TestCalendarModuleKwargsForwarding:
    """Tests for kwargs forwarding to backend for extensibility."""

    async def test_list_events_forwards_extra_kwargs_to_backend(self) -> None:
        """list_events should forward non-standard params to backend."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        await module.execute(
            "calendar:list_events",
            {
                "calendar_id": "primary",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-02-01T00:00:00Z",
                "timeZone": "America/New_York",
                "maxResults": 50,
            },
        )

        assert backend.calls[0]["kwargs"]["timeZone"] == "America/New_York"
        assert backend.calls[0]["kwargs"]["maxResults"] == 50

    async def test_check_conflicts_forwards_extra_kwargs_to_backend(self) -> None:
        """check_conflicts should forward non-standard params to backend."""
        backend = MockCalendarBackend()
        module = CalendarModule(backend)

        await module.execute(
            "calendar:check_conflicts",
            {
                "calendar_id": "primary",
                "start": "2026-01-15T10:00:00Z",
                "end": "2026-01-15T11:00:00Z",
                "timeZone": "Europe/London",
            },
        )

        assert backend.calls[0]["kwargs"]["timeZone"] == "Europe/London"
