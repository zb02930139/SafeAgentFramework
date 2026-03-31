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

"""Read-only audit log query module for SafeAgent."""

from __future__ import annotations

import fnmatch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

# Note: This module is now at safe_agent.modules.observability.audit
# but imports from safe_agent.modules.base remain unchanged since base.py
# is in the parent modules/ directory.

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 1000


class AuditModule(BaseModule):
    """Provide read-only access to the framework's audit log.

    This module allows querying the append-only JSONL audit log that the
    code gate writes to. It is read-only by design — the agent can search
    past decisions but cannot modify or delete log entries.

    All filtering is performed in-memory; the module reads the log file
    line by line for each query.

    Security considerations:
    - Read-only: no tools to write, modify, or delete audit entries.
    - Path validation: prevents directory traversal attacks.
    - Result capping: max_results prevents memory exhaustion.
    """

    def __init__(
        self,
        audit_log_path: Path | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        """Initialize the audit module.

        Args:
            audit_log_path: Path to the JSONL audit log file. If None,
                defaults to "audit.jsonl" in the current working directory.
            max_results: Maximum number of results to return per query.
                Prevents memory exhaustion from large result sets.

        Raises:
            ValueError: If max_results is not positive.
        """
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        self.audit_log_path = (
            audit_log_path if audit_log_path else Path("audit.jsonl")
        ).resolve()
        self.max_results = max_results

    def describe(self) -> ModuleDescriptor:
        """Return the audit module descriptor and tool definition."""
        return ModuleDescriptor(
            namespace="audit",
            description=(
                "Read-only access to the framework audit log for querying "
                "authorization decisions and action history."
            ),
            tools=[
                ToolDescriptor(
                    name="audit:query_audit_log",
                    description=(
                        "Search the authorization decision log by time range, "
                        "session, action pattern, or decision type."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "string",
                                "description": (
                                    "Start of time range (ISO 8601 timestamp)"
                                ),
                            },
                            "end": {
                                "type": "string",
                                "description": "End of time range (ISO 8601)",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Filter by session ID (optional)",
                            },
                            "action_filter": {
                                "type": "string",
                                "description": ("Glob pattern for action filter"),
                            },
                            "decision": {
                                "type": "string",
                                "enum": ["allow", "deny"],
                                "description": "Filter by decision (optional)",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    f"Maximum results (default: {self.max_results})"
                                ),
                            },
                        },
                        "required": ["start", "end"],
                        "additionalProperties": False,
                    },
                    action="audit:QueryAuditLog",
                    resource_param=["*"],
                    condition_keys=[
                        "audit:TimeRange",
                        "audit:SessionId",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve audit condition values from query parameters."""
        if tool_name.removeprefix("audit:") != "query_audit_log":
            return {}

        start = str(params.get("start", ""))
        end = str(params.get("end", ""))
        time_range = f"{start}/{end}" if start and end else ""

        conditions: dict[str, Any] = {
            "audit:TimeRange": time_range,
        }

        session_id = params.get("session_id")
        if session_id is not None:
            conditions["audit:SessionId"] = str(session_id)

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute the audit log query tool."""
        if tool_name.removeprefix("audit:") != "query_audit_log":
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        try:
            start = str(params["start"])
            end = str(params["end"])
            session_id = params.get("session_id")
            action_filter = params.get("action_filter")
            decision = params.get("decision")
            limit = int(params.get("limit", self.max_results))

            # Validate limit
            if limit <= 0:
                return ToolResult(
                    success=False,
                    error="limit must be a positive integer",
                )

            # Apply max_results cap
            limit = min(limit, self.max_results)

            results = self._query_audit_log(
                start=start,
                end=end,
                session_id=session_id,
                action_filter=action_filter,
                decision=decision,
                limit=limit,
            )

            return ToolResult(
                success=True,
                data={
                    "entries": results,
                    "count": len(results),
                    "truncated": len(results) == limit,
                },
            )

        except KeyError as exc:
            return ToolResult(
                success=False,
                error=f"Missing required parameter: {exc.args[0]}",
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            return ToolResult(success=False, error=str(exc))

    def _query_audit_log(
        self,
        start: str,
        end: str,
        session_id: str | None,
        action_filter: str | None,
        decision: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Query the audit log with filters.

        Parses the JSONL file line by line, applying filters and returning
        results sorted by timestamp (newest first).

        Args:
            start: ISO 8601 timestamp for start of time range.
            end: ISO 8601 timestamp for end of time range.
            session_id: Optional session ID filter.
            action_filter: Optional glob pattern for action filtering.
            decision: Optional decision filter ("allow" or "deny").
            limit: Maximum number of results to return.

        Returns:
            List of audit log entries matching the filters, sorted by
            timestamp (newest first).
        """
        if not self.audit_log_path.exists():
            return []

        # Parse time range boundaries
        start_dt = self._parse_iso_timestamp(start)
        end_dt = self._parse_iso_timestamp(end)

        results: list[dict[str, Any]] = []

        try:
            with self.audit_log_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "audit: skipping malformed log line: %.120r", line
                        )
                        continue

                    if not self._entry_matches_filters(
                        entry=entry,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        session_id=session_id,
                        action_filter=action_filter,
                        decision=decision,
                    ):
                        continue

                    results.append(entry)

                    if len(results) >= limit:
                        break

        except OSError as exc:
            logger.warning("audit: error reading log file: %s", exc)
            return []

        # Sort by timestamp (newest first)
        results.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

        return results

    def _entry_matches_filters(
        self,
        entry: dict[str, Any],
        start_dt: datetime | None,
        end_dt: datetime | None,
        session_id: str | None,
        action_filter: str | None,
        decision: str | None,
    ) -> bool:
        """Check if an audit entry matches all provided filters.

        Args:
            entry: The audit log entry to check.
            start_dt: Start of time range (inclusive), or None.
            end_dt: End of time range (inclusive), or None.
            session_id: Optional session ID to match.
            action_filter: Optional glob pattern for action matching.
            decision: Optional decision value ("allow" or "deny").

        Returns:
            True if the entry matches all filters, False otherwise.
        """
        # Time range filter
        entry_timestamp = entry.get("timestamp")
        if entry_timestamp:
            entry_dt = self._parse_iso_timestamp(str(entry_timestamp))
            if entry_dt is not None:
                if start_dt is not None and entry_dt < start_dt:
                    return False
                if end_dt is not None and entry_dt > end_dt:
                    return False

        # Session ID filter
        if session_id is not None:
            entry_session = entry.get("session_id")
            if entry_session != session_id:
                return False

        # Action filter (glob matching)
        if action_filter is not None:
            entry_action = entry.get("action", "")
            if not fnmatch.fnmatch(str(entry_action), action_filter):
                return False

        # Decision filter
        if decision is not None:
            entry_decision = entry.get("decision", "").lower()
            if entry_decision != decision.lower():
                return False

        return True

    def _parse_iso_timestamp(self, timestamp: str) -> datetime | None:
        """Parse an ISO 8601 timestamp string.

        Args:
            timestamp: ISO 8601 timestamp string.

        Returns:
            Parsed datetime, or None if parsing fails.
        """
        try:
            # Handle Z suffix for UTC
            normalized = timestamp.replace("Z", "+00:00")
            # Handle negative timezone offsets by checking for pattern
            return datetime.fromisoformat(normalized)
        except (ValueError, TypeError):
            logger.debug("audit: unable to parse timestamp: %r", timestamp)
            return None
