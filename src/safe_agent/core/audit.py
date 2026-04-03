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

"""Audit logger for SafeAgent tool dispatch decisions."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from safe_agent.access.models import Decision

_logger = logging.getLogger(__name__)

# Maximum serialised byte length for params/resolved_conditions stored in an
# audit entry. Values exceeding this limit are replaced with a truncation
# marker so the audit log cannot become a credential or bulk-data dump.
_MAX_PARAM_BYTES = 8 * 1024  # 8 KB
_TRUNCATION_MARKER = "__truncated__"


def _truncate_params(value: dict[str, Any]) -> dict[str, Any]:
    """Return *value* unchanged, or a truncation marker if it is too large.

    The entire dict is replaced (not partially redacted) to avoid creating
    records that appear complete but are missing fields.

    Args:
        value: The params or conditions dict to check.

    Returns:
        The original dict, or ``{"__truncated__": True}`` if it serialises
        to more than ``_MAX_PARAM_BYTES`` bytes.
    """
    try:
        encoded = json.dumps(value, default=str)
    except Exception:
        return {_TRUNCATION_MARKER: True}
    if len(encoded.encode()) > _MAX_PARAM_BYTES:
        return {_TRUNCATION_MARKER: True}
    return value


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Intended for use by callers (e.g. the dispatcher) that need to stamp
    a logical event time before constructing :class:`AuditEntry` objects.
    """
    return datetime.now(tz=UTC).isoformat()


class AuditEntry(BaseModel):
    """A structured record of a single authorization decision.

    Attributes:
        session_id: Identifier for the session that triggered the tool call.
        timestamp: ISO-8601 UTC timestamp of the decision.
        tool_name: The fully-qualified tool name that was evaluated.
        tool_call_id: Optional unique identifier for the tool call, used for
            provenance tracking when the LLM makes multiple identical calls.
            Propagated from :class:`~safe_agent.core.llm.ToolCall.id`.
        params: Input parameters for the tool call, capped at
            ``_MAX_PARAM_BYTES`` (8 KB). Values exceeding the cap are replaced
            with ``{"__truncated__": True}``. Callers should also redact
            known-sensitive fields (API keys, passwords, PII) before
            constructing the entry.
        resolved_conditions: Condition values resolved by the module; also
            size-capped.
        decision: The authorization decision.
        matched_statements: Sids of policy statements that contributed to the
            decision. ``None`` entries represent anonymous (no-sid) statements.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: str
    timestamp: str
    tool_name: str
    tool_call_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    resolved_conditions: dict[str, Any] = Field(default_factory=dict)
    decision: Decision
    matched_statements: list[str | None] = Field(default_factory=list)

    @field_validator("params", "resolved_conditions", mode="before")
    @classmethod
    def _cap_size(cls, v: Any) -> Any:
        """Enforce the per-field byte cap before storing."""
        if isinstance(v, dict):
            return _truncate_params(v)
        return v


class AuditLogger:
    """Appends structured :class:`AuditEntry` records as JSON lines to a file.

    Thread-safe: a :class:`threading.Lock` serialises concurrent writes so
    that log lines are never interleaved under async or threaded dispatch.

    Each call to :meth:`log` serialises the entry and appends a single
    newline-delimited JSON record to the configured log file.

    :meth:`read_entries` and :meth:`iter_entries` snapshot the file contents
    under a brief lock, then parse outside the lock, so readers never block
    writers for longer than a single file-read syscall.

    **Multi-process note:** :class:`threading.Lock` is in-process only. If
    multiple processes write to the same log file concurrently, line
    interleaving is possible. For multi-process deployments, use a dedicated
    log service or per-process log files.

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

    @property
    def log_path(self) -> Path:
        """Return the path to the audit log file."""
        return self._log_path

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string.

        Delegates to the module-level :func:`now_iso` function.
        Kept as a static method for backward compatibility with callers that
        reference ``AuditLogger.now_iso()``.
        """
        return now_iso()

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

        The file is snapshotted under a brief lock, then parsed outside the
        lock. Safe to call concurrently with :meth:`log`.

        Args:
            limit: If provided, return at most *limit* entries from the start
                of the file. ``None`` returns all entries.

        Returns:
            A list of :class:`AuditEntry` objects in file order. Returns an
            empty list if the log file does not yet exist.
        """
        return list(self.iter_entries(limit=limit))

    def iter_entries(self, limit: int | None = None) -> Iterator[AuditEntry]:
        """Iterate over entries from the audit log.

        The file contents are snapshotted under a brief lock (read + close),
        then all parsing happens outside the lock. This means readers block
        writers only for the duration of a single ``readlines()`` call — not
        for the entire parse loop — keeping contention minimal under concurrent
        dispatch.

        Args:
            limit: Stop after yielding *limit* entries. ``None`` means all.

        Yields:
            :class:`AuditEntry` objects in file order.
        """
        # Snapshot file contents under the lock. The lock is released as soon
        # as the file is closed — parsing happens entirely outside the lock.
        # FileNotFoundError is caught here to handle the case where the file
        # is deleted between the existence check (now done via try/except)
        # and the open() call, eliminating a TOCTOU race condition.
        with self._lock:
            try:
                with self._log_path.open(encoding="utf-8") as fh:
                    raw_lines = fh.readlines()
            except FileNotFoundError:
                raw_lines = []

        count = 0
        for raw in raw_lines:
            line = raw.strip()
            if not line:
                continue
            try:
                yield AuditEntry(**json.loads(line))
            except Exception:
                _logger.warning("audit: skipping unparseable log line: %.120r", line)
                continue
            count += 1
            if limit is not None and count >= limit:
                return
