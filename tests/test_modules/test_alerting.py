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

"""Tests for the alerting module."""

from unittest.mock import AsyncMock

from safe_agent.modules.alerting import AlertingModule


class MockAlertingBackend:
    """Mock implementation of AlertingBackend for testing."""

    def __init__(self) -> None:
        """Initialize the mock backend with async mocks."""
        self.list_alerts = AsyncMock(return_value={"alerts": [], "total_count": 0})
        self.acknowledge_alert = AsyncMock(
            return_value={"acknowledged": True, "alert_id": "test-123"}
        )
        self.escalate_alert = AsyncMock(
            return_value={"escalated": True, "alert_id": "test-123"}
        )
        self.silence_alert = AsyncMock(
            return_value={
                "silenced": True,
                "alert_id": "test-123",
                "silenced_until": "2024-01-01T00:00:00Z",
            }
        )


class TestAlertingModule:
    """Tests for AlertingModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "alerting"
        assert len(descriptor.tools) == 4
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "alerting:list_alerts",
            "alerting:acknowledge_alert",
            "alerting:escalate_alert",
            "alerting:silence_alert",
        }

    def test_describe_has_correct_tool_names(self) -> None:
        """Tool names should follow namespace:snake_case format."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.name.startswith("alerting:")
            # Check snake_case format (no PascalCase)
            suffix = tool.name.removeprefix("alerting:")
            assert suffix.islower() or "_" in suffix

    def test_describe_has_correct_actions(self) -> None:
        """Actions should follow alerting:PascalCase format."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        descriptor = module.describe()

        actions = {tool.action for tool in descriptor.tools}
        assert actions == {
            "alerting:ListAlerts",
            "alerting:AcknowledgeAlert",
            "alerting:EscalateAlert",
            "alerting:SilenceAlert",
        }

    def test_describe_has_correct_resource_params(self) -> None:
        """list_alerts uses source as resource_param, other tools use alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            if tool.name == "alerting:list_alerts":
                # list_alerts uses source as resource
                assert tool.resource_param == ["source"]
            else:
                # Other tools use alert_id as resource
                assert tool.resource_param == ["alert_id"]

    def test_describe_has_correct_condition_keys(self) -> None:
        """Tools should declare alerting:Severity and alerting:AlertSource."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert "alerting:Severity" in tool.condition_keys
            assert "alerting:AlertSource" in tool.condition_keys

    def test_describe_parameters_have_additional_properties_false(self) -> None:
        """Parameter schemas should have additionalProperties: False."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        descriptor = module.describe()

        for tool in descriptor.tools:
            assert tool.parameters.get("additionalProperties") is False


class TestAlertingModuleResolveConditions:
    """Tests for resolve_conditions method."""

    async def test_resolve_conditions_derives_severity(self) -> None:
        """resolve_conditions should derive alerting:Severity from params."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        conditions = await module.resolve_conditions(
            "alerting:list_alerts",
            {"severity": "critical"},
        )

        assert conditions.get("alerting:Severity") == "critical"

    async def test_resolve_conditions_derives_source(self) -> None:
        """resolve_conditions should derive alerting:AlertSource from params."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        conditions = await module.resolve_conditions(
            "alerting:list_alerts",
            {"source": "prometheus"},
        )

        assert conditions.get("alerting:AlertSource") == "prometheus"

    async def test_resolve_conditions_derives_both_severity_and_source(self) -> None:
        """resolve_conditions should derive both condition keys when present."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        conditions = await module.resolve_conditions(
            "alerting:list_alerts",
            {"severity": "warning", "source": "cloudwatch"},
        )

        assert conditions.get("alerting:Severity") == "warning"
        assert conditions.get("alerting:AlertSource") == "cloudwatch"

    async def test_resolve_conditions_returns_empty_without_params(self) -> None:
        """resolve_conditions should handle missing filter parameters."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        conditions = await module.resolve_conditions(
            "alerting:list_alerts",
            {},
        )

        assert conditions == {}

    async def test_resolve_conditions_coerces_non_string_severity(self) -> None:
        """resolve_conditions should coerce non-string severity to string."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        conditions = await module.resolve_conditions(
            "alerting:list_alerts",
            {"severity": 123},
        )

        assert conditions.get("alerting:Severity") == "123"


class TestAlertingModuleExecute:
    """Tests for execute method delegation to backend."""

    async def test_execute_delegates_list_alerts_to_backend(self) -> None:
        """execute() should delegate list_alerts to the backend."""
        backend = MockAlertingBackend()
        backend.list_alerts.return_value = {
            "alerts": [{"id": "alert-1", "severity": "critical"}],
            "total_count": 1,
        }
        module = AlertingModule(backend)

        result = await module.execute("alerting:list_alerts", {})

        assert result.success is True
        assert result.data == {
            "alerts": [{"id": "alert-1", "severity": "critical"}],
            "total_count": 1,
        }
        backend.list_alerts.assert_called_once_with()

    async def test_execute_delegates_list_alerts_with_filters(self) -> None:
        """execute() should pass filter parameters to backend."""
        backend = MockAlertingBackend()
        backend.list_alerts.return_value = {"alerts": [], "total_count": 0}
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:list_alerts",
            {
                "severity": "critical",
                "source": "prometheus",
                "state": "firing",
                "limit": 10,
            },
        )

        assert result.success is True
        backend.list_alerts.assert_called_once_with(
            severity="critical",
            source="prometheus",
            state="firing",
            limit=10,
        )

    async def test_execute_delegates_list_alerts_with_state_filter(self) -> None:
        """execute() should pass state parameter to backend."""
        backend = MockAlertingBackend()
        backend.list_alerts.return_value = {"alerts": [], "total_count": 0}
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:list_alerts",
            {"state": "acknowledged"},
        )

        assert result.success is True
        backend.list_alerts.assert_called_once_with(state="acknowledged")

    async def test_execute_delegates_acknowledge_alert_to_backend(self) -> None:
        """execute() should delegate acknowledge_alert to the backend."""
        backend = MockAlertingBackend()
        backend.acknowledge_alert.return_value = {
            "acknowledged": True,
            "alert_id": "alert-123",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:acknowledge_alert",
            {"alert_id": "alert-123"},
        )

        assert result.success is True
        assert result.data == {"acknowledged": True, "alert_id": "alert-123"}
        backend.acknowledge_alert.assert_called_once_with("alert-123")

    async def test_execute_delegates_acknowledge_alert_with_note(self) -> None:
        """execute() should pass note parameter to backend."""
        backend = MockAlertingBackend()
        backend.acknowledge_alert.return_value = {
            "acknowledged": True,
            "alert_id": "alert-123",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:acknowledge_alert",
            {"alert_id": "alert-123", "note": "Investigating"},
        )

        assert result.success is True
        backend.acknowledge_alert.assert_called_once_with(
            "alert-123",
            note="Investigating",
        )

    async def test_execute_delegates_escalate_alert_to_backend(self) -> None:
        """execute() should delegate escalate_alert to the backend."""
        backend = MockAlertingBackend()
        backend.escalate_alert.return_value = {
            "escalated": True,
            "alert_id": "alert-456",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:escalate_alert",
            {"alert_id": "alert-456"},
        )

        assert result.success is True
        assert result.data == {"escalated": True, "alert_id": "alert-456"}
        backend.escalate_alert.assert_called_once_with("alert-456")

    async def test_execute_delegates_escalate_alert_with_options(self) -> None:
        """execute() should pass escalation options to backend."""
        backend = MockAlertingBackend()
        backend.escalate_alert.return_value = {
            "escalated": True,
            "alert_id": "alert-456",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:escalate_alert",
            {
                "alert_id": "alert-456",
                "target": "oncall-primary",
                "note": "Customer impact detected",
            },
        )

        assert result.success is True
        backend.escalate_alert.assert_called_once_with(
            "alert-456",
            target="oncall-primary",
            note="Customer impact detected",
        )

    async def test_execute_delegates_silence_alert_to_backend(self) -> None:
        """execute() should delegate silence_alert to the backend."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-789",
            "silenced_until": "2024-01-01T01:00:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-789", "duration": "1h"},
        )

        assert result.success is True
        assert result.data == {
            "silenced": True,
            "alert_id": "alert-789",
            "silenced_until": "2024-01-01T01:00:00Z",
        }
        backend.silence_alert.assert_called_once_with(
            "alert-789",
            duration_minutes=60,
        )

    async def test_execute_delegates_silence_alert_with_reason(self) -> None:
        """execute() should pass reason parameter to backend."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-789",
            "silenced_until": "2024-01-01T01:00:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {
                "alert_id": "alert-789",
                "duration": "1h",
                "reason": "Planned maintenance",
            },
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-789",
            duration_minutes=60,
            reason="Planned maintenance",
        )

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """execute() should return error for unknown tool names."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute("alerting:unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_backend_error_propagates_as_tool_result_failure(self) -> None:
        """Backend exceptions should become ToolResult(success=False)."""
        backend = MockAlertingBackend()
        backend.list_alerts.side_effect = Exception("Connection refused")
        module = AlertingModule(backend)

        result = await module.execute("alerting:list_alerts", {})

        assert result.success is False
        assert result.error == "Connection refused"

    async def test_acknowledge_alert_backend_error_propagates(self) -> None:
        """acknowledge_alert backend errors should propagate as failures."""
        backend = MockAlertingBackend()
        backend.acknowledge_alert.side_effect = Exception("Alert not found")
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:acknowledge_alert",
            {"alert_id": "nonexistent"},
        )

        assert result.success is False
        assert result.error == "Alert not found"

    async def test_escalate_alert_backend_error_propagates(self) -> None:
        """escalate_alert backend errors should propagate as failures."""
        backend = MockAlertingBackend()
        backend.escalate_alert.side_effect = Exception("Permission denied")
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:escalate_alert",
            {"alert_id": "alert-123"},
        )

        assert result.success is False
        assert result.error == "Permission denied"

    async def test_silence_alert_backend_error_propagates(self) -> None:
        """silence_alert backend errors should propagate as failures."""
        backend = MockAlertingBackend()
        backend.silence_alert.side_effect = Exception("Invalid duration")
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "1h"},
        )

        assert result.success is False
        assert result.error == "Invalid duration"


class TestAlertingModuleValidation:
    """Tests for parameter validation."""

    async def test_acknowledge_alert_validates_missing_alert_id(self) -> None:
        """acknowledge_alert should return error for missing alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute("alerting:acknowledge_alert", {})

        assert result.success is False
        assert "Missing required parameter: alert_id" in result.error
        backend.acknowledge_alert.assert_not_called()

    async def test_acknowledge_alert_validates_empty_alert_id(self) -> None:
        """acknowledge_alert should return error for empty alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:acknowledge_alert",
            {"alert_id": ""},
        )

        assert result.success is False
        assert "alert_id must not be empty" in result.error
        backend.acknowledge_alert.assert_not_called()

    async def test_acknowledge_alert_validates_whitespace_alert_id(self) -> None:
        """acknowledge_alert should return error for whitespace-only alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:acknowledge_alert",
            {"alert_id": "   "},
        )

        assert result.success is False
        assert "alert_id must not be empty" in result.error
        backend.acknowledge_alert.assert_not_called()

    async def test_escalate_alert_validates_missing_alert_id(self) -> None:
        """escalate_alert should return error for missing alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute("alerting:escalate_alert", {})

        assert result.success is False
        assert "Missing required parameter: alert_id" in result.error
        backend.escalate_alert.assert_not_called()

    async def test_escalate_alert_validates_empty_alert_id(self) -> None:
        """escalate_alert should return error for empty alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:escalate_alert",
            {"alert_id": ""},
        )

        assert result.success is False
        assert "alert_id must not be empty" in result.error
        backend.escalate_alert.assert_not_called()

    async def test_silence_alert_validates_missing_alert_id(self) -> None:
        """silence_alert should return error for missing alert_id."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"duration": "1h"},
        )

        assert result.success is False
        assert "Missing required parameter: alert_id" in result.error
        backend.silence_alert.assert_not_called()

    async def test_silence_alert_validates_missing_duration(self) -> None:
        """silence_alert should return error for missing duration."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123"},
        )

        assert result.success is False
        assert "Missing required parameter: duration" in result.error
        backend.silence_alert.assert_not_called()

    async def test_silence_alert_validates_invalid_duration_format(self) -> None:
        """silence_alert should return error for invalid duration format."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "invalid"},
        )

        assert result.success is False
        assert "Invalid duration format" in result.error
        backend.silence_alert.assert_not_called()

    async def test_silence_alert_validates_empty_duration(self) -> None:
        """silence_alert should return error for empty duration."""
        backend = MockAlertingBackend()
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": ""},
        )

        assert result.success is False
        assert "Invalid duration format" in result.error
        backend.silence_alert.assert_not_called()

    async def test_silence_alert_accepts_minimum_duration(self) -> None:
        """silence_alert should accept duration of '1m'."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T00:01:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "1m"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=1,
        )


class TestAlertingModuleEdgeCases:
    """Tests for edge cases and parameter handling."""

    async def test_list_alerts_coerces_non_string_severity(self) -> None:
        """list_alerts should coerce non-string severity to string."""
        backend = MockAlertingBackend()
        backend.list_alerts.return_value = {"alerts": [], "total_count": 0}
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:list_alerts",
            {"severity": 123},
        )

        assert result.success is True
        backend.list_alerts.assert_called_once_with(severity="123")

    async def test_acknowledge_alert_coerces_non_string_alert_id(self) -> None:
        """acknowledge_alert should coerce non-string alert_id to string."""
        backend = MockAlertingBackend()
        backend.acknowledge_alert.return_value = {
            "acknowledged": True,
            "alert_id": "12345",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:acknowledge_alert",
            {"alert_id": 12345},
        )

        assert result.success is True
        backend.acknowledge_alert.assert_called_once_with("12345")

    async def test_silence_alert_parses_hour_duration(self) -> None:
        """silence_alert should parse '1h' duration format."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T01:00:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "1h"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=60,
        )

    async def test_silence_alert_parses_minute_duration(self) -> None:
        """silence_alert should parse '30m' duration format."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T00:30:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "30m"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=30,
        )

    async def test_silence_alert_parses_combined_duration(self) -> None:
        """silence_alert should parse '1h30m' combined duration format."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T01:30:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "1h30m"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=90,
        )

    async def test_silence_alert_parses_hours_only(self) -> None:
        """silence_alert should parse '2h' duration format."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T02:00:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "2h"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=120,
        )

    async def test_silence_alert_parses_plain_number_as_minutes(self) -> None:
        """silence_alert should parse plain number string as minutes."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T00:45:00Z",
        }
        module = AlertingModule(backend)

        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "45"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=45,
        )

    async def test_list_alerts_handles_empty_params_dict(self) -> None:
        """list_alerts should handle empty params dict."""
        backend = MockAlertingBackend()
        backend.list_alerts.return_value = {"alerts": [], "total_count": 0}
        module = AlertingModule(backend)

        result = await module.execute("alerting:list_alerts", {})

        assert result.success is True
        backend.list_alerts.assert_called_once_with()

    async def test_list_alerts_handles_partial_filters(self) -> None:
        """list_alerts should handle partial filter parameters."""
        backend = MockAlertingBackend()
        backend.list_alerts.return_value = {"alerts": [], "total_count": 0}
        module = AlertingModule(backend)

        # Only severity, no source or limit
        result = await module.execute(
            "alerting:list_alerts",
            {"severity": "warning"},
        )

        assert result.success is True
        backend.list_alerts.assert_called_once_with(severity="warning")

    async def test_escalate_alert_handles_empty_optional_params(self) -> None:
        """escalate_alert should handle empty optional parameters."""
        backend = MockAlertingBackend()
        backend.escalate_alert.return_value = {
            "escalated": True,
            "alert_id": "alert-123",
        }
        module = AlertingModule(backend)

        # No escalation_policy or message
        result = await module.execute(
            "alerting:escalate_alert",
            {"alert_id": "alert-123"},
        )

        assert result.success is True
        backend.escalate_alert.assert_called_once_with("alert-123")

    async def test_silence_alert_handles_empty_reason(self) -> None:
        """silence_alert should handle missing reason parameter."""
        backend = MockAlertingBackend()
        backend.silence_alert.return_value = {
            "silenced": True,
            "alert_id": "alert-123",
            "silenced_until": "2024-01-01T01:00:00Z",
        }
        module = AlertingModule(backend)

        # No reason provided
        result = await module.execute(
            "alerting:silence_alert",
            {"alert_id": "alert-123", "duration": "1h"},
        )

        assert result.success is True
        backend.silence_alert.assert_called_once_with(
            "alert-123",
            duration_minutes=60,
        )
