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

"""Database module for SafeAgent.

This module provides database query and execution capabilities through a pluggable
backend protocol. The framework defines the interface; a plugin provides the
concrete implementation (PostgreSQL, MySQL, SQLite, etc.).

Security Considerations:
    - SQL Injection Prevention: Backend implementations MUST use parameterized
      queries or prepared statements. The module passes raw SQL strings, so the
      backend is responsible for proper escaping and parameter binding.

    - Privileged Actions: The execute_statement tool performs write operations
      (INSERT, UPDATE, DELETE) and should be treated as a privileged action.
      Policy conditions should restrict access appropriately.

    - Query Result Exposure: Query results are returned in ToolResult.data and
      may be visible to the LLM. Avoid querying sensitive data without proper
      access controls.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)

# SQL patterns for validation
_DDL_PATTERN = re.compile(
    r"\b(CREATE|DROP|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)
_READONLY_PATTERN = re.compile(
    r"^\s*SELECT\b",
    re.IGNORECASE,
)


def _is_ddl(sql: str) -> bool:
    """Check if SQL contains DDL statements (CREATE, DROP, ALTER, TRUNCATE)."""
    return bool(_DDL_PATTERN.search(sql))


def _is_readonly(sql: str) -> bool:
    """Check if SQL is a read-only SELECT statement."""
    return bool(_READONLY_PATTERN.match(sql))


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol for database backend implementations.

    Implementations provide concrete integrations with database systems
    such as PostgreSQL, MySQL, SQLite, MongoDB, etc.

    Security Note:
        Backend implementations MUST use parameterized queries or prepared
        statements to prevent SQL injection attacks. Raw SQL strings received
        from the module should never be executed directly without proper
        parameter binding.

    Methods:
        query: Execute a read-only SQL query.
        execute_statement: Execute a write SQL statement (INSERT, UPDATE, DELETE).
    """

    async def query(self, database: str, sql: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a read-only SQL query.

        Args:
            database: The target database name or identifier.
            sql: The SQL query string (SELECT statements only).
            **kwargs: Backend-specific options (e.g., limit, timeout).

        Returns:
            Dict containing query results with at minimum:
                - "rows": List of row dicts (column -> value).
                - "row_count": Number of rows returned.

        Raises:
            Exception: If the query fails or attempts a write operation.
        """
        ...

    async def execute_statement(
        self,
        database: str,
        sql: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a write SQL statement (INSERT, UPDATE, DELETE).

        Args:
            database: The target database name or identifier.
            sql: The SQL statement string (INSERT, UPDATE, or DELETE).
            **kwargs: Backend-specific options (e.g., timeout).

        Returns:
            Dict containing execution results with at minimum:
                - "rows_affected": Number of rows modified.

        Raises:
            Exception: If the statement fails.
        """
        ...


class DatabaseModule(BaseModule):
    """Database module using the pluggable adapter pattern.

    This module exposes database tools to the SafeAgent framework while
    delegating actual database operations to an injected backend provider.

    Security Note:
        - SQL injection prevention is the backend's responsibility.
        - execute_statement is a privileged action - use policies to restrict.

    Attributes:
        _backend: The database backend implementation (private).
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        """Initialize the database module with a backend provider.

        Args:
            backend: An implementation of DatabaseBackend that handles
                actual database operations for a specific platform.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the database module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="database",
            description="Execute SQL queries and statements on databases.",
            tools=[
                ToolDescriptor(
                    name="database:query",
                    description="Run a read-only SQL query on a database.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "description": "The target database name.",
                            },
                            "sql": {
                                "type": "string",
                                "description": "The SQL query string (SELECT only).",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "Optional limit on number of rows to return."
                                ),
                            },
                        },
                        "required": ["database", "sql"],
                        "additionalProperties": False,
                    },
                    action="database:Query",
                    resource_param=["database"],
                    condition_keys=[
                        "database:DatabaseName",
                        "database:TableName",
                    ],
                ),
                ToolDescriptor(
                    name="database:execute_statement",
                    description=(
                        "Run a write statement (INSERT, UPDATE, DELETE) on a database. "
                        "This is a privileged action."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "description": "The target database name.",
                            },
                            "sql": {
                                "type": "string",
                                "description": (
                                    "The SQL statement (INSERT, UPDATE, or DELETE)."
                                ),
                            },
                        },
                        "required": ["database", "sql"],
                        "additionalProperties": False,
                    },
                    action="database:ExecuteStatement",
                    resource_param=["database"],
                    condition_keys=[
                        "database:DatabaseName",
                        "database:TableName",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve database condition values from parameters.

        Derives condition keys for policy evaluation:
            - database:DatabaseName: The target database name.
            - database:TableName: Best-effort extraction from SQL query.

        Table name extraction is best-effort and supports common patterns:
            - SELECT ... FROM table_name
            - INSERT INTO table_name
            - UPDATE table_name
            - DELETE FROM table_name

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        conditions: dict[str, Any] = {}

        database = params.get("database")
        if database is not None:
            conditions["database:DatabaseName"] = str(database)

        sql = params.get("sql")
        if sql is not None:
            table_name = self._extract_table_name(str(sql))
            if table_name is not None:
                conditions["database:TableName"] = table_name

        return conditions

    def _validate_params(
        self,
        params: dict[str, Any],
        sql_error_msg: str,
    ) -> tuple[str, str] | ToolResult[Any]:
        """Validate common parameters for database operations.

        Args:
            params: The input parameters for the tool.
            sql_error_msg: Error message for empty SQL (query vs statement).

        Returns:
            Tuple of (database, sql) if valid, or ToolResult with error.
        """
        database = params.get("database")
        if database is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: database",
            )

        sql = params.get("sql")
        if sql is None:
            return ToolResult(
                success=False,
                error="Missing required parameter: sql",
            )

        database = str(database)
        sql = str(sql)
        if not database or not database.strip():
            return ToolResult(
                success=False,
                error="Database name must not be empty",
            )
        if not sql or not sql.strip():
            return ToolResult(
                success=False,
                error=sql_error_msg,
            )

        return database, sql

    def _extract_table_name(self, sql: str) -> str | None:
        """Extract table name from SQL query using best-effort parsing.

        Supports common SQL patterns:
            - SELECT ... FROM table_name [WHERE ...]
            - SELECT ... FROM schema.table_name [WHERE ...]
            - INSERT INTO table_name [(columns)] VALUES ...
            - INSERT INTO schema.table_name ...
            - UPDATE table_name SET ...
            - UPDATE schema.table_name SET ...
            - DELETE FROM table_name [WHERE ...]
            - DELETE FROM schema.table_name [WHERE ...]

        Limitations:
            - Only extracts the first table from FROM/INTO/UPDATE clauses.
            - Does not parse JOINs, subqueries, or CTEs.
            - For complex queries with multiple tables, returns only the primary table.

        Args:
            sql: The SQL query string.

        Returns:
            The extracted table name (including schema if present),
            or None if extraction fails.
        """
        # Normalize SQL: collapse whitespace, convert to uppercase for matching
        normalized = " ".join(sql.upper().split())

        # Pattern for optional schema and table name: [SCHEMA.]TABLE
        # Schema and table can each contain letters, digits, underscores
        table_pattern = r"([A-Z_][A-Z0-9_]*(?:\.[A-Z_][A-Z0-9_]*)?)"

        # Pattern 1: SELECT ... FROM table_name or schema.table_name
        select_match = re.search(
            rf"\bFROM\s+{table_pattern}",
            normalized,
        )
        if select_match:
            return select_match.group(1).lower()

        # Pattern 2: INSERT INTO table_name or schema.table_name
        insert_match = re.search(
            rf"\bINSERT\s+INTO\s+{table_pattern}",
            normalized,
        )
        if insert_match:
            return insert_match.group(1).lower()

        # Pattern 3: UPDATE table_name SET or schema.table_name SET
        update_match = re.search(
            rf"\bUPDATE\s+{table_pattern}\s+SET\b",
            normalized,
        )
        if update_match:
            return update_match.group(1).lower()

        # Pattern 4: DELETE FROM table_name or schema.table_name
        delete_match = re.search(
            rf"\bDELETE\s+FROM\s+{table_pattern}",
            normalized,
        )
        if delete_match:
            return delete_match.group(1).lower()

        return None

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a database tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("database:")
            if normalized_tool == "query":
                return await self._query(params)
            if normalized_tool == "execute_statement":
                return await self._execute_statement(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _query(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate query execution to the backend.

        Security Note:
            Logs the database and SQL at DEBUG level. Query results are NOT logged.
            Rejects non-SELECT statements and DDL operations.
        """
        validation_result = self._validate_params(params, "SQL query must not be empty")
        if isinstance(validation_result, ToolResult):
            return validation_result

        database, sql = validation_result

        # Security: Reject non-SELECT statements in query tool
        if not _is_readonly(sql):
            return ToolResult(
                success=False,
                error=(
                    "Query tool only accepts SELECT statements. "
                    "Use execute_statement for INSERT, UPDATE, DELETE."
                ),
            )

        # Security: Block DDL even if it somehow passes the SELECT check
        if _is_ddl(sql):
            return ToolResult(
                success=False,
                error=(
                    "DDL statements (CREATE, DROP, ALTER, TRUNCATE) "
                    "are not allowed."
                ),
            )

        logger.debug("Database query: database=%s sql=%s", database, sql)

        kwargs: dict[str, Any] = {}
        if "limit" in params:
            kwargs["limit"] = params["limit"]

        result = await self._backend.query(database, sql, **kwargs)
        return ToolResult(success=True, data=result)

    async def _execute_statement(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate statement execution to the backend.

        Security Note:
            This is a privileged action that performs write operations.
            Logs the database name at INFO level for audit trail.
            SQL is logged at DEBUG level to avoid logging PII.
            Result data is NOT logged.
            Blocks DDL statements (CREATE, DROP, ALTER, TRUNCATE).
        """
        validation_result = self._validate_params(
            params,
            "SQL statement must not be empty",
        )
        if isinstance(validation_result, ToolResult):
            return validation_result

        database, sql = validation_result

        # Security: Block DDL statements
        if _is_ddl(sql):
            return ToolResult(
                success=False,
                error=(
                    "DDL statements (CREATE, DROP, ALTER, TRUNCATE) are not "
                    "allowed via execute_statement."
                ),
            )

        # Log write operations: database at INFO, SQL at DEBUG (PII risk mitigation)
        logger.info("Database execute_statement: database=%s", database)
        logger.debug("Database execute_statement SQL: %s", sql)

        result = await self._backend.execute_statement(database, sql)
        return ToolResult(success=True, data=result)
