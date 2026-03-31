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

"""Tests for the pluggable error tracking module."""

from typing import Any

from safe_agent.modules.observability.error_tracking import (
    ErrorTrackingBackend,
    ErrorTrackingModule,
)


class MockErrorTrackingBackend:
    """Mock backend for testing ErrorTrackingModule."""

    def __init__(self) -> None:
        """Initialize mock backend with storage for errors."""
        self.stored_errors: dict[str, list[dict[str, Any]]] = {}
        # Track kwargs passed to query_errors
        self.last_query_kwargs: dict[str, Any] = {}

    async def query_errors(
        self,
        project: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Mock query_errors that returns stored errors for the project."""
        self.last_query_kwargs = kwargs  # Track what kwargs were passed
        errors = self.stored_errors.get(project, [])
        return errors


class TestErrorTrackingModule:
    """Tests for ErrorTrackingModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "error_tracking"
        assert len(descriptor.tools) == 1
        assert descriptor.tools[0].name == "error_tracking:query_errors"

    def test_describe_tool_has_correct_action_and_resource_param(self) -> None:
        """Tool should have correct action name and resource param."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        descriptor = module.describe()

        tool = descriptor.tools[0]
        assert tool.action == "error_tracking:QueryErrors"
        assert tool.resource_param == ["project"]
        assert tool.condition_keys == [
            "error_tracking:Project",
            "error_tracking:ErrorType",
        ]

    def test_backend_protocol_check(self) -> None:
        """MockErrorTrackingBackend should satisfy the ErrorTrackingBackend protocol."""
        backend = MockErrorTrackingBackend()
        assert isinstance(backend, ErrorTrackingBackend)

    async def test_query_errors_delegates_to_backend(self) -> None:
        """query_errors should delegate to the backend and return success."""
        backend = MockErrorTrackingBackend()
        backend.stored_errors["my-project"] = [
            {"id": "err-1", "type": "TypeError", "message": "Cannot read property"},
            {"id": "err-2", "type": "ValueError", "message": "Invalid value"},
        ]
        module = ErrorTrackingModule(backend)

        result = await module.execute(
            "error_tracking:query_errors",
            {"project": "my-project"},
        )

        assert result.success is True
        assert result.data == {
            "errors": [
                {"id": "err-1", "type": "TypeError", "message": "Cannot read property"},
                {"id": "err-2", "type": "ValueError", "message": "Invalid value"},
            ]
        }

    async def test_query_errors_with_optional_params(self) -> None:
        """query_errors should pass through optional query, time_range, limit."""
        backend = MockErrorTrackingBackend()
        backend.stored_errors["my-project"] = [
            {"id": "err-1", "type": "TypeError", "message": "Cannot read property"},
        ]
        module = ErrorTrackingModule(backend)

        result = await module.execute(
            "error_tracking:query_errors",
            {
                "project": "my-project",
                "query": "TypeError",
                "time_range": "24h",
                "limit": 10,
            },
        )

        assert result.success is True
        assert result.data == {
            "errors": [
                {"id": "err-1", "type": "TypeError", "message": "Cannot read property"},
            ]
        }
        # Verify the kwargs were actually forwarded to the backend
        assert backend.last_query_kwargs.get("query") == "TypeError"
        assert backend.last_query_kwargs.get("time_range") == "24h"
        assert backend.last_query_kwargs.get("limit") == 10

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """Execute should return error for unknown tool names."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        result = await module.execute(
            "error_tracking:unknown_tool",
            {"project": "my-project"},
        )

        assert result.success is False
        assert result.error == "Unknown tool: error_tracking:unknown_tool"

    async def test_backend_error_propagates_as_tool_result_error(self) -> None:
        """Backend errors should be caught and returned as ToolResult errors."""

        class FailingBackend(MockErrorTrackingBackend):
            async def query_errors(
                self, project: str, **kwargs: Any
            ) -> list[dict[str, Any]]:
                raise RuntimeError("API connection failed")

        backend = FailingBackend()
        module = ErrorTrackingModule(backend)

        result = await module.execute(
            "error_tracking:query_errors",
            {"project": "my-project"},
        )

        assert result.success is False
        assert result.error == "API connection failed"

    async def test_resolve_conditions_derives_project(self) -> None:
        """resolve_conditions should derive error_tracking:Project from params."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"project": "my-project"},
        )

        assert conditions == {
            "error_tracking:Project": "my-project",
        }

    async def test_resolve_conditions_infers_error_type_from_query(self) -> None:
        """resolve_conditions should infer error_tracking:ErrorType from query."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        # Test type: prefix
        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"project": "my-project", "query": "type:TypeError"},
        )

        assert conditions == {
            "error_tracking:Project": "my-project",
            "error_tracking:ErrorType": "TypeError",
        }

        # Test error_type: prefix
        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"project": "my-project", "query": "error_type:ValueError"},
        )

        assert conditions == {
            "error_tracking:Project": "my-project",
            "error_tracking:ErrorType": "ValueError",
        }

        # Test error: prefix
        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"project": "my-project", "query": "error:RuntimeError"},
        )

        assert conditions == {
            "error_tracking:Project": "my-project",
            "error_tracking:ErrorType": "RuntimeError",
        }

    async def test_resolve_conditions_no_error_type_without_query(self) -> None:
        """resolve_conditions should not add ErrorType if no query present."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"project": "my-project", "limit": 10},
        )

        assert conditions == {
            "error_tracking:Project": "my-project",
        }

    async def test_resolve_conditions_no_error_type_without_type_in_query(
        self,
    ) -> None:
        """resolve_conditions should not add ErrorType if query lacks type prefix."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"project": "my-project", "query": "database connection failed"},
        )

        assert conditions == {
            "error_tracking:Project": "my-project",
        }

    async def test_resolve_conditions_returns_empty_for_missing_project(
        self,
    ) -> None:
        """resolve_conditions should return empty dict if project is missing."""
        backend = MockErrorTrackingBackend()
        module = ErrorTrackingModule(backend)

        conditions = await module.resolve_conditions(
            "error_tracking:query_errors",
            {"query": "TypeError"},
        )

        assert conditions == {}
