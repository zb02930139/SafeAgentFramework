"""Audit logger for SafeAgent tool dispatch decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """A structured record of a single authorization decision.

    Attributes:
        session_id: Identifier for the session that triggered the tool call.
        timestamp: ISO-8601 UTC timestamp of the decision.
        tool_name: The fully-qualified tool name that was evaluated.
        params: The raw input parameters for the tool call.
        resolved_conditions: Condition values resolved by the module at
            dispatch time, passed to the policy evaluator.
        decision: The authorization decision string (e.g. ``"ALLOWED"``,
            ``"DENIED_EXPLICIT"``, ``"DENIED_IMPLICIT"``).
        matched_statements: Sids (or empty strings for anonymous statements)
            of policy statements that contributed to the decision.
    """

    session_id: str
    timestamp: str
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    resolved_conditions: dict[str, Any] = Field(default_factory=dict)
    decision: str
    matched_statements: list[str] = Field(default_factory=list)


class AuditLogger:
    """Appends structured :class:`AuditEntry` records as JSON lines to a file.

    Each call to :meth:`log` serialises the entry and appends a single
    newline-delimited JSON record to the configured log file, creating the
    file (and any parent directories) if necessary.

    Args:
        log_path: Path to the JSON-lines audit log file.

    Example::

        logger = AuditLogger(log_path=Path("/var/log/safe_agent/audit.jsonl"))
        logger.log(entry)
    """

    def __init__(self, log_path: Path) -> None:
        """Initialise the audit logger.

        Args:
            log_path: Destination file for JSON-lines audit records. Parent
                directories are created on the first :meth:`log` call if they
                do not exist.
        """
        self._log_path = log_path

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(tz=UTC).isoformat()

    def log(self, entry: AuditEntry) -> None:
        """Append *entry* as a JSON line to the audit log file.

        Args:
            entry: The :class:`AuditEntry` to persist.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        line = entry.model_dump_json() + "\n"
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def read_entries(self) -> list[AuditEntry]:
        """Read and parse all entries from the audit log.

        Returns:
            A list of :class:`AuditEntry` objects in file order. Returns an
            empty list if the log file does not yet exist.
        """
        if not self._log_path.exists():
            return []
        entries: list[AuditEntry] = []
        with self._log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(AuditEntry(**json.loads(line)))
        return entries
