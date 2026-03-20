"""Audit logger for SafeAgent tool dispatch decisions."""

from __future__ import annotations

import collections.abc
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from safe_agent.iam.models import Decision


class AuditEntry(BaseModel):
    """A structured record of a single authorization decision.

    Attributes:
        session_id: Identifier for the session that triggered the tool call.
        timestamp: ISO-8601 UTC timestamp of the decision.
        tool_name: The fully-qualified tool name that was evaluated.
        params: The raw input parameters for the tool call. Sensitive fields
            should be redacted by the caller before constructing this entry.
        resolved_conditions: Condition values resolved by the module at
            dispatch time, passed to the policy evaluator.
        decision: The authorization decision.
        matched_statements: Sids of policy statements that contributed to the
            decision. ``None`` entries represent anonymous (no-sid) statements.
    """

    session_id: str
    timestamp: str
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    resolved_conditions: dict[str, Any] = Field(default_factory=dict)
    decision: Decision
    matched_statements: list[str | None] = Field(default_factory=list)


class AuditLogger:
    """Appends structured :class:`AuditEntry` records as JSON lines to a file.

    Thread-safe: a :class:`threading.Lock` serialises concurrent writes so
    that log lines are never interleaved under async or threaded dispatch.

    Each call to :meth:`log` serialises the entry and appends a single
    newline-delimited JSON record to the configured log file, creating the
    file (and any parent directories) if necessary.

    Timestamping is owned by :meth:`log`; callers pass a pre-constructed
    :class:`AuditEntry` with the timestamp already set (typically from the
    dispatcher which owns the logical event time).

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
                directories are created eagerly so that logging never fails
                due to a missing directory.
        """
        self._log_path = log_path
        self._lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string.

        Intended for use by callers (e.g. the dispatcher) that need to stamp
        a logical event time before constructing :class:`AuditEntry` objects.
        """
        return datetime.now(tz=UTC).isoformat()

    def log(self, entry: AuditEntry) -> None:
        """Append *entry* as a JSON line to the audit log file.

        Thread-safe. The lock ensures no two concurrent callers interleave
        partial writes.

        Args:
            entry: The :class:`AuditEntry` to persist.
        """
        line = entry.model_dump_json() + "\n"
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def read_entries(self, limit: int | None = None) -> list[AuditEntry]:
        """Read and parse entries from the audit log.

        Args:
            limit: If provided, return at most *limit* entries (from the
                start of the file). Pass ``None`` to read all entries. For
                large logs, prefer streaming via :meth:`iter_entries`.

        Returns:
            A list of :class:`AuditEntry` objects in file order. Returns an
            empty list if the log file does not yet exist.
        """
        return list(self.iter_entries(limit=limit))

    def iter_entries(
        self, limit: int | None = None
    ) -> collections.abc.Iterator[AuditEntry]:
        """Stream entries from the audit log one at a time.

        Args:
            limit: Stop after yielding *limit* entries. ``None`` means all.

        Yields:
            :class:`AuditEntry` objects in file order.
        """
        if not self._log_path.exists():
            return
        count = 0
        with self._log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield AuditEntry(**json.loads(line))
                count += 1
                if limit is not None and count >= limit:
                    return
