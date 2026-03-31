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

"""Tests for the audit module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_agent.modules.observability.audit import DEFAULT_MAX_RESULTS, AuditModule


class TestAuditModuleDescriptor:
    """Tests for module descriptor and describe() method."""

    def test_describe_returns_valid_descriptor(self, tmp_path: Path) -> None:
        """describe() should return correct module metadata."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        descriptor = module.describe()

        assert descriptor.namespace == "audit"
        assert len(descriptor.tools) == 1

        tool = descriptor.tools[0]
        assert tool.name == "audit:query_audit_log"
        assert tool.action == "audit:QueryAuditLog"
        assert tool.resource_param == ["*"]
        assert "audit:TimeRange" in tool.condition_keys
        assert "audit:SessionId" in tool.condition_keys

    def test_tool_schema_additional_properties_false(self, tmp_path: Path) -> None:
        """Tool schema should have additionalProperties: False."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        descriptor = module.describe()
        tool = descriptor.tools[0]

        assert tool.parameters.get("additionalProperties") is False

    def test_max_results_default(self, tmp_path: Path) -> None:
        """Default max_results should be DEFAULT_MAX_RESULTS."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        assert module.max_results == DEFAULT_MAX_RESULTS

    def test_max_results_custom(self, tmp_path: Path) -> None:
        """max_results should be configurable."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file, max_results=100)

        assert module.max_results == 100

    def test_max_results_zero_raises_valueerror(self, tmp_path: Path) -> None:
        """max_results=0 should raise ValueError."""
        log_file = tmp_path / "audit.jsonl"
        with pytest.raises(ValueError, match="max_results must be positive"):
            AuditModule(audit_log_path=log_file, max_results=0)

    def test_max_results_negative_raises_valueerror(self, tmp_path: Path) -> None:
        """max_results<0 should raise ValueError."""
        log_file = tmp_path / "audit.jsonl"
        with pytest.raises(ValueError, match="max_results must be positive"):
            AuditModule(audit_log_path=log_file, max_results=-1)

    def test_audit_log_path_is_resolved(self, tmp_path: Path) -> None:
        """audit_log_path should be resolved to absolute path."""
        log_file = tmp_path / "nested" / ".." / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        # Path should be resolved (no .. components)
        assert ".." not in str(module.audit_log_path)
        assert module.audit_log_path.is_absolute()


class TestAuditModuleResolveConditions:
    """Tests for resolve_conditions() method."""

    async def test_resolve_time_range_conditions(self, tmp_path: Path) -> None:
        """Should resolve TimeRange condition from start/end."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        conditions = await module.resolve_conditions(
            "audit:query_audit_log",
            {"start": "2026-03-20T00:00:00Z", "end": "2026-03-21T00:00:00Z"},
        )

        assert (
            conditions["audit:TimeRange"] == "2026-03-20T00:00:00Z/2026-03-21T00:00:00Z"
        )

    async def test_resolve_session_id_condition(self, tmp_path: Path) -> None:
        """Should resolve SessionId condition when session_id present."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        conditions = await module.resolve_conditions(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-21T00:00:00Z",
                "session_id": "test-session-123",
            },
        )

        assert conditions["audit:SessionId"] == "test-session-123"

    async def test_resolve_no_session_id_without_param(self, tmp_path: Path) -> None:
        """Should not include SessionId condition without session_id param."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        conditions = await module.resolve_conditions(
            "audit:query_audit_log",
            {"start": "2026-03-20T00:00:00Z", "end": "2026-03-21T00:00:00Z"},
        )

        assert "audit:SessionId" not in conditions

    async def test_resolve_unknown_tool_returns_empty(self, tmp_path: Path) -> None:
        """resolve_conditions should return empty dict for unknown tools."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        conditions = await module.resolve_conditions(
            "audit:unknown_tool",
            {"start": "2026-03-20T00:00:00Z", "end": "2026-03-21T00:00:00Z"},
        )

        assert conditions == {}


class TestAuditModuleQuery:
    """Tests for query_audit_log tool execution."""

    def _create_log_entry(
        self,
        session_id: str,
        timestamp: str,
        tool_name: str,
        action: str,
        decision: str,
    ) -> dict:
        """Create a sample audit log entry."""
        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "tool_name": tool_name,
            "action": action,
            "decision": decision.upper(),
            "params": {},
            "resolved_conditions": {},
            "matched_statements": [],
        }

    async def test_query_empty_log_file(self, tmp_path: Path) -> None:
        """Query on empty/nonexistent log should return empty results."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-21T00:00:00Z",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["entries"] == []
        assert result.data["count"] == 0

    async def test_query_filters_by_time_range(self, tmp_path: Path) -> None:
        """query_audit_log should filter by time range."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        # Create log with entries at different times
        entries = [
            self._create_log_entry(
                "s1", "2026-03-20T10:00:00Z", "tool:action1", "test:Action", "allow"
            ),
            self._create_log_entry(
                "s2", "2026-03-21T10:00:00Z", "tool:action2", "test:Action", "deny"
            ),
            self._create_log_entry(
                "s3", "2026-03-22T10:00:00Z", "tool:action3", "test:Action", "allow"
            ),
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        # Query for middle day only
        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-21T00:00:00Z",
                "end": "2026-03-21T23:59:59Z",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["count"] == 1
        assert result.data["entries"][0]["session_id"] == "s2"

    async def test_query_filters_by_session_id(self, tmp_path: Path) -> None:
        """query_audit_log should filter by session_id."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        entries = [
            self._create_log_entry(
                "session-a",
                "2026-03-20T10:00:00Z",
                "tool:action",
                "test:Action",
                "allow",
            ),
            self._create_log_entry(
                "session-b",
                "2026-03-20T11:00:00Z",
                "tool:action",
                "test:Action",
                "deny",
            ),
            self._create_log_entry(
                "session-a",
                "2026-03-20T12:00:00Z",
                "tool:action",
                "test:Action",
                "allow",
            ),
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "session_id": "session-a",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["count"] == 2
        for entry in result.data["entries"]:
            assert entry["session_id"] == "session-a"

    async def test_query_filters_by_action_glob(self, tmp_path: Path) -> None:
        """query_audit_log should filter by action pattern (glob)."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        entries = [
            self._create_log_entry(
                "s1",
                "2026-03-20T10:00:00Z",
                "tool:action1",
                "filesystem:ReadFile",
                "allow",
            ),
            self._create_log_entry(
                "s2",
                "2026-03-20T11:00:00Z",
                "tool:action2",
                "filesystem:WriteFile",
                "deny",
            ),
            self._create_log_entry(
                "s3", "2026-03-20T12:00:00Z", "tool:action3", "shell:Execute", "allow"
            ),
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "action_filter": "filesystem:*",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["count"] == 2
        for entry in result.data["entries"]:
            assert entry["action"].startswith("filesystem:")

    async def test_query_filters_by_decision(self, tmp_path: Path) -> None:
        """query_audit_log should filter by decision (allow/deny)."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        entries = [
            self._create_log_entry(
                "s1", "2026-03-20T10:00:00Z", "tool:action1", "test:Action", "allow"
            ),
            self._create_log_entry(
                "s2", "2026-03-20T11:00:00Z", "tool:action2", "test:Action", "deny"
            ),
            self._create_log_entry(
                "s3", "2026-03-20T12:00:00Z", "tool:action3", "test:Action", "DENY"
            ),
            self._create_log_entry(
                "s4", "2026-03-20T13:00:00Z", "tool:action4", "test:Action", "ALLOW"
            ),
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        # Query for allow only
        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "decision": "allow",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["count"] == 2
        for entry in result.data["entries"]:
            assert entry["decision"].lower() == "allow"

        # Query for deny only
        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "decision": "deny",
            },
        )

        assert result.data["count"] == 2
        for entry in result.data["entries"]:
            assert entry["decision"].lower() == "deny"

    async def test_query_respects_limit_parameter(self, tmp_path: Path) -> None:
        """query_audit_log should respect the limit parameter."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        # Create 5 entries
        entries = [
            self._create_log_entry(
                f"s{i}",
                f"2026-03-20T{i:02d}:00:00Z",
                f"tool:action{i}",
                "test:Action",
                "allow",
            )
            for i in range(1, 6)
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "limit": 3,
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["count"] == 3
        # Sorted by timestamp newest first
        assert result.data["truncated"] is True

    async def test_query_respects_max_results_cap(self, tmp_path: Path) -> None:
        """query_audit_log should cap results at max_results."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file, max_results=2)

        entries = [
            self._create_log_entry(
                f"s{i}",
                f"2026-03-20T{i:02d}:00:00Z",
                f"tool:action{i}",
                "test:Action",
                "allow",
            )
            for i in range(1, 6)
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "limit": 10,  # Request more than max_results
            },
        )

        assert result.data["count"] == 2
        assert result.data["truncated"] is True

    async def test_query_sorts_newest_first(self, tmp_path: Path) -> None:
        """Query results should be sorted by timestamp newest first."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        entries = [
            self._create_log_entry(
                "s1", "2026-03-20T10:00:00Z", "tool:action1", "test:Action", "allow"
            ),
            self._create_log_entry(
                "s2", "2026-03-20T14:00:00Z", "tool:action2", "test:Action", "allow"
            ),
            self._create_log_entry(
                "s3", "2026-03-20T08:00:00Z", "tool:action3", "test:Action", "allow"
            ),
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
            },
        )

        assert result.success is True
        timestamps = [e["timestamp"] for e in result.data["entries"]]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_query_malformed_lines_skipped_gracefully(
        self, tmp_path: Path
    ) -> None:
        """Malformed JSON lines should be skipped without failing."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        # Valid entry, malformed entry, valid entry
        log_content = (
            json.dumps(
                self._create_log_entry(
                    "s1", "2026-03-20T10:00:00Z", "tool:action1", "test:Action", "allow"
                )
            )
            + "\nthis is not json\n"
            + json.dumps(
                self._create_log_entry(
                    "s2", "2026-03-20T11:00:00Z", "tool:action2", "test:Action", "allow"
                )
            )
            + "\n"
        )

        log_file.write_text(log_content, encoding="utf-8")

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
            },
        )

        assert result.success is True
        assert result.data["count"] == 2

    async def test_query_blank_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines in log file should be skipped."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        log_content = (
            json.dumps(
                self._create_log_entry(
                    "s1", "2026-03-20T10:00:00Z", "tool:action1", "test:Action", "allow"
                )
            )
            + "\n\n\n"  # Blank lines
            + json.dumps(
                self._create_log_entry(
                    "s2", "2026-03-20T11:00:00Z", "tool:action2", "test:Action", "allow"
                )
            )
            + "\n"
        )

        log_file.write_text(log_content, encoding="utf-8")

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
            },
        )

        assert result.success is True
        assert result.data["count"] == 2


class TestAuditModuleErrorHandling:
    """Tests for error handling in AuditModule."""

    async def test_execute_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        """execute() should return error for unknown tool names."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        result = await module.execute(
            "audit:unknown_tool",
            {"start": "2026-03-20T00:00:00Z", "end": "2026-03-21T00:00:00Z"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_execute_missing_required_param_start(self, tmp_path: Path) -> None:
        """execute() should error when required param 'start' is missing."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        result = await module.execute(
            "audit:query_audit_log",
            {"end": "2026-03-21T00:00:00Z"},
        )

        assert result.success is False
        assert "Missing required parameter" in result.error

    async def test_execute_invalid_limit_rejected(self, tmp_path: Path) -> None:
        """execute() should reject invalid limit values."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-21T00:00:00Z",
                "limit": 0,
            },
        )

        assert result.success is False
        assert result.error == "limit must be a positive integer"

    async def test_execute_negative_limit_rejected(self, tmp_path: Path) -> None:
        """execute() should reject negative limit values."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-21T00:00:00Z",
                "limit": -5,
            },
        )

        assert result.success is False
        assert result.error == "limit must be a positive integer"


class TestAuditModuleCombinations:
    """Tests for combined filter operations."""

    def _create_log_entry(
        self,
        session_id: str,
        timestamp: str,
        tool_name: str,
        action: str,
        decision: str,
    ) -> dict:
        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "tool_name": tool_name,
            "action": action,
            "decision": decision.upper(),
            "params": {},
            "resolved_conditions": {},
            "matched_statements": [],
        }

    async def test_combined_filters(self, tmp_path: Path) -> None:
        """Multiple filters should be applied together (AND logic)."""
        log_file = tmp_path / "audit.jsonl"
        module = AuditModule(audit_log_path=log_file)

        entries = [
            # Match: session-a, filesystem, allow
            self._create_log_entry(
                "session-a",
                "2026-03-20T10:00:00Z",
                "tool:action1",
                "filesystem:ReadFile",
                "allow",
            ),
            # No match: wrong session
            self._create_log_entry(
                "session-b",
                "2026-03-20T11:00:00Z",
                "tool:action2",
                "filesystem:ReadFile",
                "allow",
            ),
            # No match: wrong action
            self._create_log_entry(
                "session-a",
                "2026-03-20T12:00:00Z",
                "tool:action3",
                "shell:Execute",
                "allow",
            ),
            # Match: session-a, filesystem, allow
            self._create_log_entry(
                "session-a",
                "2026-03-20T13:00:00Z",
                "tool:action4",
                "filesystem:ListDirectory",
                "allow",
            ),
            # No match: wrong decision
            self._create_log_entry(
                "session-a",
                "2026-03-20T14:00:00Z",
                "tool:action5",
                "filesystem:ReadFile",
                "deny",
            ),
        ]

        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = await module.execute(
            "audit:query_audit_log",
            {
                "start": "2026-03-20T00:00:00Z",
                "end": "2026-03-20T23:59:59Z",
                "session_id": "session-a",
                "action_filter": "filesystem:*",
                "decision": "allow",
            },
        )

        assert result.success is True
        assert result.data["count"] == 2
        for entry in result.data["entries"]:
            assert entry["session_id"] == "session-a"
            assert entry["action"].startswith("filesystem:")
            assert entry["decision"] == "ALLOW"
