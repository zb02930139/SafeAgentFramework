"""Tests for safe_agent.core.audit — AuditLogger and AuditEntry."""

from __future__ import annotations

import json
from pathlib import Path

from safe_agent.core.audit import AuditEntry, AuditLogger


def _make_entry(**overrides) -> AuditEntry:
    """Create a minimal AuditEntry for testing."""
    defaults = {
        "session_id": "sess-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "tool_name": "fs:ReadFile",
        "params": {"path": "/etc/hosts"},
        "resolved_conditions": {"env": "prod"},
        "decision": "ALLOWED",
        "matched_statements": ["AllowRead"],
    }
    defaults.update(overrides)
    return AuditEntry(**defaults)


class TestAuditEntry:
    """Tests for AuditEntry model."""

    def test_required_fields(self) -> None:
        """AuditEntry should store all required fields."""
        entry = _make_entry()
        assert entry.session_id == "sess-1"
        assert entry.tool_name == "fs:ReadFile"
        assert entry.decision == "ALLOWED"

    def test_defaults(self) -> None:
        """Optional fields default to empty collections."""
        entry = AuditEntry(
            session_id="s",
            timestamp="2026-01-01T00:00:00+00:00",
            tool_name="t",
            decision="DENIED_IMPLICIT",
        )
        assert entry.params == {}
        assert entry.resolved_conditions == {}
        assert entry.matched_statements == []

    def test_serialises_to_json(self) -> None:
        """AuditEntry.model_dump_json() should produce valid JSON."""
        entry = _make_entry()
        raw = entry.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["session_id"] == "sess-1"
        assert parsed["decision"] == "ALLOWED"


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_creates_file_on_first_log(self, tmp_path: Path) -> None:
        """log() should create the log file if it does not exist."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        assert not log_file.exists()
        logger.log(_make_entry())
        assert log_file.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """log() should create missing parent directories."""
        log_file = tmp_path / "deep" / "nested" / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry())
        assert log_file.exists()

    def test_entry_written_as_json_line(self, tmp_path: Path) -> None:
        """Each logged entry should be a valid JSON line."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry())
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tool_name"] == "fs:ReadFile"

    def test_multiple_entries_appended(self, tmp_path: Path) -> None:
        """Multiple log() calls should append separate JSON lines."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry(decision="ALLOWED"))
        logger.log(_make_entry(decision="DENIED_EXPLICIT"))
        logger.log(_make_entry(decision="DENIED_IMPLICIT"))
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3
        decisions = [json.loads(ln)["decision"] for ln in lines]
        assert decisions == ["ALLOWED", "DENIED_EXPLICIT", "DENIED_IMPLICIT"]

    def test_entry_contains_all_required_fields(self, tmp_path: Path) -> None:
        """Each written entry must contain all required AuditEntry fields."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        logger.log(_make_entry())
        parsed = json.loads(log_file.read_text().strip())
        for field in (
            "session_id",
            "timestamp",
            "tool_name",
            "params",
            "resolved_conditions",
            "decision",
            "matched_statements",
        ):
            assert field in parsed, f"Missing field: {field}"

    def test_read_entries_empty_when_no_file(self, tmp_path: Path) -> None:
        """read_entries() should return [] when the log file does not exist."""
        logger = AuditLogger(log_path=tmp_path / "missing.jsonl")
        assert logger.read_entries() == []

    def test_read_entries_roundtrip(self, tmp_path: Path) -> None:
        """read_entries() should return AuditEntry objects matching what was logged."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        entry = _make_entry()
        logger.log(entry)
        entries = logger.read_entries()
        assert len(entries) == 1
        assert entries[0].session_id == entry.session_id
        assert entries[0].decision == entry.decision
