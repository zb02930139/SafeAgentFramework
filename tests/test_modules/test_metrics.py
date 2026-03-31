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

"""Tests for the pluggable metrics module."""

from unittest.mock import AsyncMock

from safe_agent.modules.observability.metrics import MetricsModule


class MockMetricsBackend:
    """Mock backend implementing MetricsBackend for testing."""

    def __init__(self) -> None:
        self.query_metrics = AsyncMock(return_value={"results": []})


class TestMetricsModule:
    """Tests for MetricsModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "metrics"
        assert len(descriptor.tools) == 1
        tool = descriptor.tools[0]
        assert tool.name == "metrics:query_metrics"
        assert tool.action == "metrics:QueryMetrics"
        assert tool.resource_param == ["datasource"]
        assert "metrics:QueryLanguage" in tool.condition_keys
        assert "metrics:TimeRange" in tool.condition_keys

    def test_tool_parameters_schema(self) -> None:
        """query_metrics tool should have correct parameter schema."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        descriptor = module.describe()
        tool = descriptor.tools[0]

        props = tool.parameters["properties"]
        assert "datasource" in props
        assert "query" in props
        assert "start" in props
        assert "end" in props
        assert "step" in props
        assert props["step"].get("description") == "e.g., '15s', '1m'"

        required = tool.parameters["required"]
        assert "datasource" in required
        assert "query" in required
        assert "start" in required
        assert "end" in required
        assert "step" not in required  # step is optional

    async def test_execute_delegates_to_backend(self) -> None:
        """execute() should delegate query_metrics to the backend."""
        backend = MockMetricsBackend()
        backend.query_metrics.return_value = {
            "results": [
                {"metric": "up", "value": [1704067200, "1"]},
            ]
        }
        module = MetricsModule(backend)

        result = await module.execute(
            "metrics:query_metrics",
            {
                "datasource": "prometheus",
                "query": "up",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T01:00:00Z",
                "step": "15s",
            },
        )

        assert result.success is True
        assert result.data == {
            "results": [{"metric": "up", "value": [1704067200, "1"]}]
        }
        backend.query_metrics.assert_awaited_once_with(
            datasource="prometheus",
            query="up",
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T01:00:00Z",
            step="15s",
        )

    async def test_execute_without_optional_step(self) -> None:
        """execute() should work without optional step parameter."""
        backend = MockMetricsBackend()
        backend.query_metrics.return_value = {"results": []}
        module = MetricsModule(backend)

        result = await module.execute(
            "metrics:query_metrics",
            {
                "datasource": "prometheus",
                "query": "up",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T01:00:00Z",
            },
        )

        assert result.success is True
        backend.query_metrics.assert_awaited_once_with(
            datasource="prometheus",
            query="up",
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T01:00:00Z",
            step=None,
        )

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """execute() should return error for unrecognized tool names."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        result = await module.execute("metrics:unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_execute_handles_backend_exception(self) -> None:
        """execute() should return error when backend raises exception."""
        backend = MockMetricsBackend()
        backend.query_metrics.side_effect = ConnectionError("Backend unavailable")
        module = MetricsModule(backend)

        result = await module.execute(
            "metrics:query_metrics",
            {
                "datasource": "prometheus",
                "query": "up",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T01:00:00Z",
            },
        )

        assert result.success is False
        assert "Backend unavailable" in result.error

    async def test_resolve_conditions_detects_promql_rate(self) -> None:
        """resolve_conditions should detect PromQL for rate() queries."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        conditions = await module.resolve_conditions(
            "metrics:query_metrics",
            {
                "datasource": "prometheus",
                "query": "rate(http_requests_total[5m])",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T01:00:00Z",
            },
        )

        assert conditions["metrics:QueryLanguage"] == "promql"
        assert conditions["metrics:TimeRange"] == "1h"

    async def test_resolve_conditions_detects_promql_sum_by(self) -> None:
        """resolve_conditions should detect PromQL for sum(...) by (...) queries."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        conditions = await module.resolve_conditions(
            "metrics:query_metrics",
            {
                "datasource": "prometheus",
                "query": "sum(rate(http_requests_total[5m])) by (job)",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T00:30:00Z",
            },
        )

        assert conditions["metrics:QueryLanguage"] == "promql"
        assert conditions["metrics:TimeRange"] == "30m"

    async def test_resolve_conditions_detects_influxql_select(self) -> None:
        """resolve_conditions should detect InfluxQL for SELECT queries."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        conditions = await module.resolve_conditions(
            "metrics:query_metrics",
            {
                "datasource": "influxdb",
                "query": "SELECT mean(value) FROM cpu WHERE time > now() - 1h",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T00:15:00Z",
            },
        )

        assert conditions["metrics:QueryLanguage"] == "influxql"
        assert conditions["metrics:TimeRange"] == "15m"

    async def test_resolve_conditions_returns_unknown_for_plain_query(self) -> None:
        """resolve_conditions should return 'unknown' for unrecognized syntax."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        conditions = await module.resolve_conditions(
            "metrics:query_metrics",
            {
                "datasource": "custom",
                "query": "get metric:up",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T00:05:00Z",
            },
        )

        assert conditions["metrics:QueryLanguage"] == "unknown"
        assert conditions["metrics:TimeRange"] == "5m"

    async def test_resolve_conditions_returns_empty_for_unknown_tool(self) -> None:
        """resolve_conditions should return empty dict for unknown tools."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        conditions = await module.resolve_conditions(
            "metrics:unknown_tool",
            {"query": "up"},
        )

        assert conditions == {}

    def test_detect_query_language_promql_aggregations(self) -> None:
        """_detect_query_language should identify PromQL aggregation functions."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert module._detect_query_language("sum(up)") == "promql"
        assert module._detect_query_language("avg(http_latency)") == "promql"
        assert module._detect_query_language("max(cpu_usage)") == "promql"
        assert module._detect_query_language("min(memory_bytes)") == "promql"
        assert module._detect_query_language("count(instances)") == "promql"

    def test_detect_query_language_promql_functions(self) -> None:
        """_detect_query_language should identify PromQL rate functions."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._detect_query_language("rate(http_requests_total[5m])") == "promql"
        )
        assert module._detect_query_language("irate(cpu_usage[1m])") == "promql"
        assert module._detect_query_language("increase(errors_total[1h])") == "promql"
        assert (
            module._detect_query_language(
                "histogram_quantile(0.95, "
                "rate(http_request_duration_seconds_bucket[5m]))"
            )
            == "promql"
        )

    def test_detect_query_language_promql_by_clause(self) -> None:
        """_detect_query_language should identify PromQL 'by' clause."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert module._detect_query_language("up by (job)") == "promql"
        assert module._detect_query_language("sum(up) by (instance)") == "promql"

    def test_detect_query_language_influxql_select(self) -> None:
        """_detect_query_language should identify InfluxQL SELECT statements."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert module._detect_query_language("SELECT * FROM cpu") == "influxql"
        assert (
            module._detect_query_language("SELECT mean(value) FROM memory")
            == "influxql"
        )

    def test_detect_query_language_influxql_from_where(self) -> None:
        """_detect_query_language should identify InfluxQL with FROM/WHERE."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._detect_query_language("FROM cpu WHERE host='server1'") == "influxql"
        )

    def test_detect_query_language_unknown(self) -> None:
        """_detect_query_language should return 'unknown' for unrecognized syntax."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert module._detect_query_language("") == "unknown"
        assert module._detect_query_language("simple_metric_name") == "unknown"
        assert module._detect_query_language("get foo") == "unknown"

    def test_compute_time_range_seconds(self) -> None:
        """_compute_time_range should format sub-minute ranges as seconds."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-01T00:00:30Z")
            == "30s"
        )
        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-01T00:00:45Z")
            == "45s"
        )

    def test_compute_time_range_minutes(self) -> None:
        """_compute_time_range should format sub-hour ranges as minutes."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-01T00:30:00Z")
            == "30m"
        )
        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-01T00:45:00Z")
            == "45m"
        )

    def test_compute_time_range_hours(self) -> None:
        """_compute_time_range should format sub-day ranges as hours."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z")
            == "6h"
        )
        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-01T23:59:59Z")
            == "23h"
        )

    def test_compute_time_range_days(self) -> None:
        """_compute_time_range should format multi-day ranges as days."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z")
            == "2d"
        )
        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "2024-01-08T00:00:00Z")
            == "7d"
        )

    def test_compute_time_range_invalid_end_before_start(self) -> None:
        """_compute_time_range should return 'invalid' for reversed ranges."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._compute_time_range("2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z")
            == "invalid"
        )

    def test_compute_time_range_unknown_for_malformed(self) -> None:
        """_compute_time_range should return 'unknown' for unparseable timestamps."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        assert (
            module._compute_time_range("not-a-date", "2024-01-01T00:00:00Z")
            == "unknown"
        )
        assert (
            module._compute_time_range("2024-01-01T00:00:00Z", "not-a-date")
            == "unknown"
        )

    def test_repr_shows_namespace(self) -> None:
        """__repr__ should include the module namespace."""
        backend = MockMetricsBackend()
        module = MetricsModule(backend)

        repr_str = repr(module)
        assert "MetricsModule" in repr_str
        assert "metrics" in repr_str
