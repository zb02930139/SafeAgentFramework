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

"""Tests for the pluggable logging interface module."""

from unittest.mock import AsyncMock

from safe_agent.modules.observability.logging import LoggingBackend, LoggingModule


class MockLoggingBackend:
    """Mock implementation of LoggingBackend for testing."""

    def __init__(self) -> None:
        """Initialize the mock with AsyncMock methods."""
        self.query_logs = AsyncMock(return_value=[])
        self.write_log = AsyncMock(return_value={"id": "log-123", "status": "written"})


class TestLoggingModuleDescriptor:
    """Tests for LoggingModule.describe() functionality."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        descriptor = module.describe()

        assert descriptor.namespace == "logging"
        assert len(descriptor.tools) == 2
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "logging:query_logs",
            "logging:write_log",
        }

    def test_query_logs_tool_descriptor(self) -> None:
        """query_logs tool should have correct action and resource param."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        descriptor = module.describe()
        query_tool = next(
            tool for tool in descriptor.tools if tool.name == "logging:query_logs"
        )

        assert query_tool.action == "logging:QueryLogs"
        assert query_tool.resource_param == ["source"]
        assert query_tool.condition_keys == ["logging:LogSource"]
        assert query_tool.parameters["required"] == [
            "source",
            "query",
            "start",
            "end",
        ]
        assert query_tool.parameters["additionalProperties"] is False

    def test_write_log_tool_descriptor(self) -> None:
        """write_log tool should have correct action and condition keys."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        descriptor = module.describe()
        write_tool = next(
            tool for tool in descriptor.tools if tool.name == "logging:write_log"
        )

        assert write_tool.action == "logging:WriteLog"
        assert write_tool.resource_param == ["source"]
        assert set(write_tool.condition_keys) == {
            "logging:LogSource",
            "logging:Severity",
        }
        assert write_tool.parameters["required"] == ["source", "level", "message"]
        assert write_tool.parameters["additionalProperties"] is False


class TestLoggingModuleResolveConditions:
    """Tests for LoggingModule.resolve_conditions() functionality."""

    async def test_resolve_conditions_derives_log_source(self) -> None:
        """resolve_conditions should derive logging:LogSource from source param."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        conditions = await module.resolve_conditions(
            "logging:query_logs",
            {"source": "application", "query": "error"},
        )

        assert conditions["logging:LogSource"] == "application"

    async def test_resolve_conditions_derives_severity_for_write_log(self) -> None:
        """resolve_conditions should derive logging:Severity from level param."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        conditions = await module.resolve_conditions(
            "logging:write_log",
            {"source": "system", "level": "error", "message": "Disk full"},
        )

        assert conditions["logging:LogSource"] == "system"
        assert conditions["logging:Severity"] == "error"

    async def test_resolve_conditions_handles_missing_params(self) -> None:
        """resolve_conditions should handle missing parameters gracefully."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        conditions = await module.resolve_conditions("logging:query_logs", {})

        assert conditions == {}

    async def test_resolve_conditions_coerces_to_string(self) -> None:
        """resolve_conditions should coerce source and level to strings."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        conditions = await module.resolve_conditions(
            "logging:write_log",
            {"source": 123, "level": 456, "message": "test"},
        )

        assert conditions["logging:LogSource"] == "123"
        assert conditions["logging:Severity"] == "456"


class TestLoggingModuleExecute:
    """Tests for LoggingModule.execute() functionality."""

    async def test_execute_query_logs_delegates_to_backend(self) -> None:
        """execute() should delegate query_logs to the backend."""
        mock_backend = MockLoggingBackend()
        mock_backend.query_logs.return_value = [
            {"timestamp": "2026-03-27T10:00:00Z", "message": "Test log entry"},
        ]
        module = LoggingModule(backend=mock_backend)

        result = await module.execute(
            "logging:query_logs",
            {
                "source": "app",
                "query": "error",
                "start": "2026-03-27T00:00:00Z",
                "end": "2026-03-27T23:59:59Z",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert "entries" in result.data
        assert len(result.data["entries"]) == 1
        mock_backend.query_logs.assert_awaited_once_with(
            source="app",
            query="error",
            start="2026-03-27T00:00:00Z",
            end="2026-03-27T23:59:59Z",
        )

    async def test_execute_query_logs_with_limit(self) -> None:
        """execute() should pass limit parameter to backend."""
        mock_backend = MockLoggingBackend()
        mock_backend.query_logs.return_value = []
        module = LoggingModule(backend=mock_backend)

        result = await module.execute(
            "logging:query_logs",
            {
                "source": "audit",
                "query": "user:admin",
                "start": "2026-03-27T00:00:00Z",
                "end": "2026-03-27T23:59:59Z",
                "limit": 100,
            },
        )

        assert result.success is True
        mock_backend.query_logs.assert_awaited_once_with(
            source="audit",
            query="user:admin",
            start="2026-03-27T00:00:00Z",
            end="2026-03-27T23:59:59Z",
            limit=100,
        )

    async def test_execute_write_log_delegates_to_backend(self) -> None:
        """execute() should delegate write_log to the backend."""
        mock_backend = MockLoggingBackend()
        mock_backend.write_log.return_value = {
            "id": "log-456",
            "status": "indexed",
        }
        module = LoggingModule(backend=mock_backend)

        result = await module.execute(
            "logging:write_log",
            {
                "source": "service-api",
                "level": "warning",
                "message": "Rate limit approaching",
            },
        )

        assert result.success is True
        assert result.data == {"id": "log-456", "status": "indexed"}
        mock_backend.write_log.assert_awaited_once_with(
            source="service-api",
            level="warning",
            message="Rate limit approaching",
        )

    async def test_execute_write_log_with_metadata(self) -> None:
        """execute() should pass metadata parameter to backend."""
        mock_backend = MockLoggingBackend()
        mock_backend.write_log.return_value = {"id": "log-789", "status": "written"}
        module = LoggingModule(backend=mock_backend)

        result = await module.execute(
            "logging:write_log",
            {
                "source": "webapp",
                "level": "error",
                "message": "Database connection failed",
                "metadata": {"retry_count": 3, "db_host": "db.example.com"},
            },
        )

        assert result.success is True
        mock_backend.write_log.assert_awaited_once_with(
            source="webapp",
            level="error",
            message="Database connection failed",
            metadata={"retry_count": 3, "db_host": "db.example.com"},
        )

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """execute() should return error for unknown tool names."""
        mock_backend = MockLoggingBackend()
        module = LoggingModule(backend=mock_backend)

        result = await module.execute("logging:unknown_tool", {"source": "test"})

        assert result.success is False
        assert result.error is not None
        assert "Unknown tool" in result.error
        assert "logging:unknown_tool" in result.error

    async def test_execute_propagates_backend_error(self) -> None:
        """execute() should propagate backend errors as failed ToolResult."""
        mock_backend = MockLoggingBackend()
        mock_backend.query_logs.side_effect = ConnectionError("Backend unreachable")
        module = LoggingModule(backend=mock_backend)

        result = await module.execute(
            "logging:query_logs",
            {
                "source": "app",
                "query": "error",
                "start": "2026-03-27T00:00:00Z",
                "end": "2026-03-27T23:59:59Z",
            },
        )

        assert result.success is False
        assert result.error is not None
        assert "Backend unreachable" in result.error

    async def test_execute_propagates_backend_write_error(self) -> None:
        """execute() should propagate write_log backend errors."""
        mock_backend = MockLoggingBackend()
        mock_backend.write_log.side_effect = ValueError("Invalid log level")
        module = LoggingModule(backend=mock_backend)

        result = await module.execute(
            "logging:write_log",
            {"source": "app", "level": "critical", "message": "test"},
        )

        assert result.success is False
        assert result.error is not None
        assert "Invalid log level" in result.error


class TestLoggingBackendProtocol:
    """Tests to verify LoggingBackend protocol compatibility."""

    def test_mock_backend_satisfies_protocol(self) -> None:
        """MockLoggingBackend should satisfy the LoggingBackend protocol."""
        mock_backend: LoggingBackend = MockLoggingBackend()
        # This assignment verifies protocol compliance at runtime
        # mypy will verify at static type-check time
        assert hasattr(mock_backend, "query_logs")
        assert hasattr(mock_backend, "write_log")

    async def test_backend_query_logs_signature(self) -> None:
        """query_logs should accept the documented parameters."""
        mock_backend = MockLoggingBackend()
        await mock_backend.query_logs(
            source="test",
            query="error",
            start="2026-01-01T00:00:00Z",
            end="2026-01-02T00:00:00Z",
            limit=50,
        )
        mock_backend.query_logs.assert_awaited_once()

    async def test_backend_write_log_signature(self) -> None:
        """write_log should accept the documented parameters."""
        mock_backend = MockLoggingBackend()
        await mock_backend.write_log(
            source="test",
            level="info",
            message="test message",
            metadata={"key": "value"},
        )
        mock_backend.write_log.assert_awaited_once()


class TestLoggingModuleIntegration:
    """Integration tests for LoggingModule with mock backend."""

    async def test_full_query_workflow(self) -> None:
        """Test complete query_logs workflow from describe to execute."""
        mock_backend = MockLoggingBackend()
        mock_backend.query_logs.return_value = [
            {"timestamp": "2026-03-27T12:00:00Z", "level": "error", "message": "OOM"},
            {"timestamp": "2026-03-27T12:01:00Z", "level": "info", "message": "OK"},
        ]
        module = LoggingModule(backend=mock_backend)

        # Verify descriptor
        descriptor = module.describe()
        assert descriptor.namespace == "logging"

        # Resolve conditions
        conditions = await module.resolve_conditions(
            "logging:query_logs",
            {
                "source": "system",
                "query": "OOM",
                "start": "2026-03-27T00:00:00Z",
                "end": "2026-03-27T23:59:59Z",
            },
        )
        assert conditions["logging:LogSource"] == "system"

        # Execute
        result = await module.execute(
            "logging:query_logs",
            {
                "source": "system",
                "query": "OOM",
                "start": "2026-03-27T00:00:00Z",
                "end": "2026-03-27T23:59:59Z",
            },
        )
        assert result.success is True
        assert result.data is not None
        assert len(result.data["entries"]) == 2

    async def test_full_write_workflow(self) -> None:
        """Test complete write_log workflow from describe to execute."""
        mock_backend = MockLoggingBackend()
        mock_backend.write_log.return_value = {"id": "log-abc", "indexed": True}
        module = LoggingModule(backend=mock_backend)

        # Verify descriptor includes both condition keys
        descriptor = module.describe()
        write_tool = next(t for t in descriptor.tools if t.name == "logging:write_log")
        assert "logging:LogSource" in write_tool.condition_keys
        assert "logging:Severity" in write_tool.condition_keys

        # Resolve conditions
        conditions = await module.resolve_conditions(
            "logging:write_log",
            {"source": "api", "level": "warning", "message": "Rate limit"},
        )
        assert conditions["logging:LogSource"] == "api"
        assert conditions["logging:Severity"] == "warning"

        # Execute
        result = await module.execute(
            "logging:write_log",
            {"source": "api", "level": "warning", "message": "Rate limit"},
        )
        assert result.success is True
        assert result.data is not None
        assert result.data["indexed"] is True
