"""Tests for safe_agent.core.audit — AuditLogger and AuditEntry."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from safe_agent.core.audit import AuditEntry, AuditLogger
from safe_agent.iam.models import Decision


def _make_entry(**overrides) -> AuditEntry:
    """Create a minimal AuditEntry for testing."""
    defaults = {
        "session_id": "sess-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "tool_name": "fs:ReadFile",
        "params": {"path": "/etc/hosts"},
        "resolved_conditions": {"env": "prod"},
        "decision": Decision.ALLOWED,
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
        assert entry.decision == Decision.ALLOWED

    def test_decision_must_be_decision_enum(self) -> None:
        """AuditEntry.decision must be a Decision enum value."""
        entry = _make_entry(decision=Decision.DENIED_EXPLICIT)
        assert entry.decision == Decision.DENIED_EXPLICIT

    def test_defaults(self) -> None:
        """Optional fields default to empty collections."""
        entry = AuditEntry(
            session_id="s",
            timestamp="2026-01-01T00:00:00+00:00",
            tool_name="t",
            decision=Decision.DENIED_IMPLICIT,
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

    def test_matched_statements_allows_none(self) -> None:
        """matched_statements may contain None for anonymous statements."""
        entry = _make_entry(matched_statements=[None, "SomePolicy"])
        assert entry.matched_statements == [None, "SomePolicy"]

    def test_tool_call_id_defaults_to_none(self) -> None:
        """tool_call_id should default to None when not provided."""
        entry = _make_entry()
        assert entry.tool_call_id is None

    def test_tool_call_id_stored_when_provided(self) -> None:
        """tool_call_id should be stored when explicitly provided."""
        entry = _make_entry(tool_call_id="call-abc123")
        assert entry.tool_call_id == "call-abc123"

    def test_tool_call_id_in_json(self) -> None:
        """tool_call_id should appear in JSON output when set."""
        entry = _make_entry(tool_call_id="call-xyz")
        parsed = json.loads(entry.model_dump_json())
        assert parsed["tool_call_id"] == "call-xyz"

    def test_tool_call_id_null_in_json_when_none(self) -> None:
        """tool_call_id should be null in JSON when not set."""
        entry = _make_entry()
        parsed = json.loads(entry.model_dump_json())
        assert parsed["tool_call_id"] is None


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
        """AuditLogger.__init__ should create missing parent directories."""
        log_file = tmp_path / "deep" / "nested" / "audit.jsonl"
        AuditLogger(log_path=log_file)
        assert log_file.parent.exists()

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
        logger.log(_make_entry(decision=Decision.ALLOWED))
        logger.log(_make_entry(decision=Decision.DENIED_EXPLICIT))
        logger.log(_make_entry(decision=Decision.DENIED_IMPLICIT))
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
        logger = AuditLogger(log_path=tmp_path / "sub" / "missing.jsonl")
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

    def test_read_entries_limit(self, tmp_path: Path) -> None:
        """read_entries(limit=N) should return at most N entries."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        for _ in range(5):
            logger.log(_make_entry())
        assert len(logger.read_entries(limit=3)) == 3
        assert len(logger.read_entries(limit=10)) == 5

    def test_now_iso_returns_utc_string(self) -> None:
        """now_iso() should return a non-empty ISO-8601 UTC string."""
        ts = AuditLogger.now_iso()
        assert isinstance(ts, str)
        assert len(ts) > 0
        # Python's datetime.isoformat() with tz=UTC always produces "+00:00"
        assert "+00:00" in ts

    def test_iter_entries_skips_malformed_lines(self, tmp_path: Path, caplog) -> None:
        """iter_entries() should skip unparseable lines and log a warning."""
        import logging

        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)

        # Write one valid entry, one corrupt line, one valid entry.
        logger.log(_make_entry(decision=Decision.ALLOWED))
        with log_file.open("a") as fh:
            fh.write("not valid json at all\n")
        logger.log(_make_entry(decision=Decision.DENIED_EXPLICIT))

        with caplog.at_level(logging.WARNING, logger="safe_agent.core.audit"):
            entries = logger.read_entries()

        assert len(entries) == 2
        assert entries[0].decision == Decision.ALLOWED
        assert entries[1].decision == Decision.DENIED_EXPLICIT
        assert any("skipping unparseable" in r.message for r in caplog.records)

    def test_concurrent_writes_produce_valid_entries(self, tmp_path: Path) -> None:
        """Concurrent log() calls from multiple threads must not corrupt entries."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_file)
        n_threads = 10
        entries_per_thread = 20
        errors: list[Exception] = []

        def _write(thread_idx: int) -> None:
            try:
                for i in range(entries_per_thread):
                    logger.log(
                        _make_entry(
                            session_id=f"thread-{thread_idx}",
                            tool_name=f"tool:{i}",
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        entries = logger.read_entries()
        assert len(entries) == n_threads * entries_per_thread
        # Every line must be valid JSON — no interleaving
        for raw in log_file.read_text().splitlines():
            if raw.strip():
                json.loads(raw)  # raises if corrupt
