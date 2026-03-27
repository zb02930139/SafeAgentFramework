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


"""Tests for the dashboard module."""

from typing import Any

from safe_agent.modules.dashboard import DashboardModule


class MockDashboardBackend:
    """Mock backend for testing DashboardModule."""

    def __init__(self) -> None:
        self._dashboards: list[dict] = [
            {"id": "dash-1", "name": "System Metrics", "tags": ["infra"]},
            {"id": "dash-2", "name": "Application Metrics", "tags": ["app"]},
        ]
        self._panels: dict[str, dict] = {
            "dash-1:panel-a": {"data": [1, 2, 3], "title": "CPU Usage"},
            "dash-1:panel-b": {"data": [4, 5, 6], "title": "Memory Usage"},
        }

    async def get_panel(
        self,
        dashboard_id: str,
        panel_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock get_panel implementation."""
        key = f"{dashboard_id}:{panel_id}"
        if key in self._panels:
            result = dict(self._panels[key])
            if kwargs.get("time_range"):
                result["time_range"] = kwargs["time_range"]
            if kwargs.get("output_format"):
                result["output_format"] = kwargs["output_format"]
            return result
        raise ValueError(f"Panel not found: {key}")

    async def list_dashboards(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Mock list_dashboards implementation."""
        result = list(self._dashboards)
        if kwargs.get("filter"):
            filter_str = kwargs["filter"].lower()
            result = [d for d in result if filter_str in d["name"].lower()]
        if kwargs.get("tags"):
            tags = set(kwargs["tags"])
            result = [d for d in result if tags & set(d.get("tags", []))]
        return result


class TestDashboardModule:
    """Tests for DashboardModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return expected module metadata."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "dashboard"
        assert len(descriptor.tools) == 2
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "dashboard:get_panel",
            "dashboard:list_dashboards",
        }

    def test_describe_get_panel_tool(self) -> None:
        """get_panel tool descriptor should have correct properties."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        descriptor = module.describe()
        get_panel_tool = next(
            tool for tool in descriptor.tools if tool.name == "dashboard:get_panel"
        )

        assert get_panel_tool.action == "dashboard:GetPanel"
        assert get_panel_tool.resource_param == ["dashboard_id"]
        assert "dashboard:DashboardId" in get_panel_tool.condition_keys
        assert "dashboard_id" in get_panel_tool.parameters["properties"]
        assert "panel_id" in get_panel_tool.parameters["properties"]
        assert "time_range" in get_panel_tool.parameters["properties"]
        assert "output_format" in get_panel_tool.parameters["properties"]
        assert get_panel_tool.parameters["required"] == ["dashboard_id", "panel_id"]
        assert get_panel_tool.parameters["additionalProperties"] is False

    def test_describe_list_dashboards_tool(self) -> None:
        """list_dashboards tool descriptor should have correct properties."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        descriptor = module.describe()
        list_tool = next(
            tool
            for tool in descriptor.tools
            if tool.name == "dashboard:list_dashboards"
        )

        assert list_tool.action == "dashboard:ListDashboards"
        assert list_tool.resource_param == ["*"]
        assert list_tool.condition_keys == []
        assert "filter" in list_tool.parameters["properties"]
        assert "tags" in list_tool.parameters["properties"]
        assert list_tool.parameters["required"] == []
        assert list_tool.parameters["additionalProperties"] is False

    async def test_resolve_conditions_get_panel(self) -> None:
        """resolve_conditions should derive dashboard:DashboardId for get_panel."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        conditions = await module.resolve_conditions(
            "dashboard:get_panel",
            {"dashboard_id": "my-dashboard", "panel_id": "panel-1"},
        )

        assert conditions == {"dashboard:DashboardId": "my-dashboard"}

    async def test_resolve_conditions_list_dashboards(self) -> None:
        """resolve_conditions should return empty dict for list_dashboards."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        conditions = await module.resolve_conditions(
            "dashboard:list_dashboards",
            {"filter": "system"},
        )

        assert conditions == {}

    async def test_resolve_conditions_no_dashboard_id(self) -> None:
        """resolve_conditions should handle missing dashboard_id gracefully."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        conditions = await module.resolve_conditions(
            "dashboard:get_panel",
            {"panel_id": "panel-1"},
        )

        assert conditions == {}

    async def test_execute_get_panel(self) -> None:
        """Execute should delegate to backend.get_panel."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:get_panel",
            {"dashboard_id": "dash-1", "panel_id": "panel-a"},
        )

        assert result.success is True
        assert result.data == {"data": [1, 2, 3], "title": "CPU Usage"}

    async def test_execute_get_panel_with_optional_params(self) -> None:
        """Execute should pass optional params to backend.get_panel."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:get_panel",
            {
                "dashboard_id": "dash-1",
                "panel_id": "panel-a",
                "time_range": "7d",
                "output_format": "snapshot",
            },
        )

        assert result.success is True
        assert result.data["time_range"] == "7d"
        assert result.data["output_format"] == "snapshot"

    async def test_execute_list_dashboards(self) -> None:
        """Execute should delegate to backend.list_dashboards."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:list_dashboards",
            {},
        )

        assert result.success is True
        assert len(result.data["dashboards"]) == 2
        assert result.data["dashboards"][0]["id"] == "dash-1"

    async def test_execute_list_dashboards_with_filter(self) -> None:
        """Execute should pass filter to backend.list_dashboards."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:list_dashboards",
            {"filter": "System"},
        )

        assert result.success is True
        assert len(result.data["dashboards"]) == 1
        assert result.data["dashboards"][0]["name"] == "System Metrics"

    async def test_execute_list_dashboards_with_tags(self) -> None:
        """Execute should pass tags to backend.list_dashboards."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:list_dashboards",
            {"tags": ["app"]},
        )

        assert result.success is True
        assert len(result.data["dashboards"]) == 1
        assert result.data["dashboards"][0]["name"] == "Application Metrics"

    async def test_execute_unknown_tool(self) -> None:
        """Execute should return error for unknown tool names."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:unknown_tool",
            {"dashboard_id": "dash-1"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_execute_backend_error_propagation(self) -> None:
        """Backend errors should propagate as ToolResult(success=False)."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        result = await module.execute(
            "dashboard:get_panel",
            {"dashboard_id": "dash-1", "panel_id": "nonexistent"},
        )

        assert result.success is False
        assert "Panel not found" in result.error

    async def test_execute_backend_error_for_list_dashboards(self) -> None:
        """Backend errors during list_dashboards should propagate correctly."""

        class FailingBackend:
            async def get_panel(
                self, dashboard_id: str, panel_id: str, **kwargs: Any
            ) -> dict:
                return {}

            async def list_dashboards(self, **kwargs: Any) -> list[dict]:
                raise RuntimeError("Backend unavailable")

        module = DashboardModule(FailingBackend())

        result = await module.execute(
            "dashboard:list_dashboards",
            {},
        )

        assert result.success is False
        assert "Backend unavailable" in result.error

    async def test_protocol_compatibility(self) -> None:
        """Mock backend should satisfy DashboardBackend protocol."""
        backend = MockDashboardBackend()
        module = DashboardModule(backend)

        # This test verifies the type system accepts the mock
        assert module._backend is backend
