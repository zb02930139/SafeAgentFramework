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

"""Sandboxed filesystem module for SafeAgent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiofiles

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)


class FilesystemModule(BaseModule):
    """Provide sandboxed filesystem operations rooted at a single directory."""

    DEFAULT_MAX_WRITE_SIZE = 10 * 1024 * 1024  # 10MB default
    DEFAULT_MAX_READ_SIZE = 10 * 1024 * 1024  # 10MB default
    DEFAULT_MAX_LIST_ENTRIES = 1000  # Prevent resource exhaustion

    def __init__(
        self,
        root: Path | None = None,
        max_write_size: int = DEFAULT_MAX_WRITE_SIZE,
        max_read_size: int = DEFAULT_MAX_READ_SIZE,
        max_list_entries: int = DEFAULT_MAX_LIST_ENTRIES,
    ) -> None:
        """Initialise the module with an allowed filesystem root.

        Args:
            root: The allowed filesystem root directory. Defaults to the current
                working directory if not specified.
            max_write_size: Maximum bytes allowed per file write (default 10MB).
            max_read_size: Maximum bytes allowed per file read (default 10MB).
            max_list_entries: Maximum entries returned by list_directory (default 1000).

        Raises:
            ValueError: If max_write_size, max_read_size, or max_list_entries
                is zero or negative.
        """
        if max_write_size <= 0:
            raise ValueError("max_write_size must be positive")
        if max_read_size <= 0:
            raise ValueError("max_read_size must be positive")
        if max_list_entries <= 0:
            raise ValueError("max_list_entries must be positive")
        self.root = (root or Path.cwd()).resolve()
        if root is None:
            logger.warning(
                "FilesystemModule instantiated without explicit root; "
                "defaulting to cwd: %s. This may grant broader access than intended.",
                self.root,
            )
        self.max_write_size = max_write_size
        self.max_read_size = max_read_size
        self.max_list_entries = max_list_entries

    def describe(self) -> ModuleDescriptor:
        """Return the filesystem module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="filesystem",
            description="Sandboxed filesystem operations within a configured root.",
            tools=[
                ToolDescriptor(
                    name="filesystem:read_file",
                    description="Read a file's contents.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string", "default": "utf-8"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    action="filesystem:ReadFile",
                    resource_param=["path"],
                    condition_keys=[
                        "filesystem:FileExtension",
                        "filesystem:FileSize",
                        "filesystem:IsDirectory",
                    ],
                ),
                ToolDescriptor(
                    name="filesystem:write_file",
                    description="Write content to a file.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "encoding": {"type": "string", "default": "utf-8"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                    action="filesystem:WriteFile",
                    resource_param=["path"],
                    condition_keys=["filesystem:FileExtension"],
                ),
                ToolDescriptor(
                    name="filesystem:list_directory",
                    description="List directory contents.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean", "default": False},
                            "pattern": {"type": "string"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    action="filesystem:ListDirectory",
                    resource_param=["path"],
                    condition_keys=["filesystem:IsDirectory"],
                ),
                ToolDescriptor(
                    name="filesystem:delete_file",
                    description="Delete a file.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    action="filesystem:DeleteFile",
                    resource_param=["path"],
                    condition_keys=["filesystem:FileExtension"],
                ),
                ToolDescriptor(
                    name="filesystem:move_file",
                    description="Move or rename a file.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "destination"],
                        "additionalProperties": False,
                    },
                    action="filesystem:MoveFile",
                    resource_param=["source", "destination"],
                    condition_keys=["filesystem:FileExtension"],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve filesystem condition values from the relevant path input."""
        path_value = self._condition_path_value(tool_name, params)
        if path_value is None:
            return {}

        resolved = self._resolve_path(path_value)
        suffix = resolved.suffix
        is_directory = resolved.is_dir()

        conditions: dict[str, Any] = {
            "filesystem:FileExtension": suffix,
            "filesystem:IsDirectory": "true" if is_directory else "false",
        }
        if resolved.exists() and resolved.is_file():
            conditions["filesystem:FileSize"] = resolved.stat().st_size
        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Execute a sandboxed filesystem tool invocation."""
        try:
            normalized_tool = tool_name.removeprefix("filesystem:")
            if normalized_tool == "read_file":
                return await self._read_file(params)
            if normalized_tool == "write_file":
                return await self._write_file(params)
            if normalized_tool == "list_directory":
                return await self._list_directory(params)
            if normalized_tool == "delete_file":
                return await self._delete_file(params)
            if normalized_tool == "move_file":
                return await self._move_file(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except ValueError:
            return ToolResult(success=False, error="Path outside allowed directory")
        except Exception as exc:  # pragma: no cover - defensive catch
            return ToolResult(success=False, error=str(exc))

    def _resolve_path(self, path_value: str) -> Path:
        """Resolve *path_value* against the module root and enforce sandboxing."""
        resolved = (self.root / path_value).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path outside allowed directory")
        return resolved

    def _condition_path_value(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> str | None:
        """Select the path-like parameter used for condition resolution."""
        normalized_tool = tool_name.removeprefix("filesystem:")
        if normalized_tool == "move_file":
            destination = params.get("destination")
            source = params.get("source")
            if destination is not None:
                return str(destination)
            if source is not None:
                return str(source)
            return None

        path_value = params.get("path")
        if path_value is None:
            return None
        return str(path_value)

    async def _read_file(self, params: dict[str, Any]) -> ToolResult:
        """Read file contents using async I/O."""
        path = self._resolve_path(str(params["path"]))
        encoding = str(params.get("encoding", "utf-8"))

        if not path.exists():
            return ToolResult(success=False, error="File not found")
        if path.is_dir():
            return ToolResult(success=False, error="Path is a directory")

        # Note: TOCTOU race between stat() and read() — file could grow in the
        # window. This is acceptable because: (1) the limit prevents unbounded
        # allocation, just not exact enforcement, and (2) Python's read() will
        # still be bounded by available memory; the limit provides defense-in-depth
        # rather than a hard guarantee.
        file_size = path.stat().st_size
        if file_size > self.max_read_size:
            return ToolResult(
                success=False,
                error=(
                    f"File size {file_size} bytes exceeds "
                    f"maximum {self.max_read_size} bytes"
                ),
            )

        async with aiofiles.open(path, encoding=encoding) as file_handle:
            content = await file_handle.read()

        return ToolResult(
            success=True,
            data={
                "path": str(path.relative_to(self.root)),
                "content": content,
            },
        )

    async def _write_file(self, params: dict[str, Any]) -> ToolResult:
        """Write file contents using async I/O."""
        path = self._resolve_path(str(params["path"]))
        encoding = str(params.get("encoding", "utf-8"))
        content = str(params["content"])
        content_bytes = content.encode(encoding)

        if len(content_bytes) > self.max_write_size:
            return ToolResult(
                success=False,
                error=(
                    f"File size {len(content_bytes)} bytes exceeds "
                    f"maximum {self.max_write_size} bytes"
                ),
            )

        if path.exists() and path.is_dir():
            return ToolResult(success=False, error="Path is a directory")

        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding=encoding) as file_handle:
            await file_handle.write(content)

        return ToolResult(
            success=True,
            data={
                "path": str(path.relative_to(self.root)),
                "bytes_written": len(content_bytes),
            },
        )

    async def _list_directory(self, params: dict[str, Any]) -> ToolResult:
        """List entries in a directory, optionally recursively and by pattern.

        Results are capped at max_list_entries to prevent resource exhaustion.
        If truncated, the result includes truncated=true and a warning.

        Note: Results are sorted after truncation. When truncated, the returned
        subset is non-deterministic (depends on filesystem iteration order).
        This is acceptable for the resource-exhaustion use case; callers needing
        deterministic results should use a more specific pattern or pagination.
        """
        path = self._resolve_path(str(params["path"]))
        recursive = bool(params.get("recursive", False))
        pattern = str(params.get("pattern", "*"))

        if not path.exists():
            return ToolResult(success=False, error="Directory not found")
        if not path.is_dir():
            return ToolResult(success=False, error="Path is not a directory")
        if ".." in Path(pattern).parts:
            return ToolResult(
                success=False,
                error="Pattern must not contain '..' components",
            )

        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        entries: list[dict[str, str | bool]] = []
        truncated = False

        for entry in iterator:
            if len(entries) >= self.max_list_entries:
                truncated = True
                logger.warning(
                    "list_directory truncated at %d entries for path=%s pattern=%s",
                    self.max_list_entries,
                    path,
                    pattern,
                )
                break
            entries.append(
                {
                    "path": str(entry.relative_to(self.root)),
                    "is_directory": entry.is_dir(),
                }
            )

        entries.sort(key=lambda item: str(item["path"]))

        result_data: dict[str, Any] = {"entries": entries}
        if truncated:
            result_data["truncated"] = True
            result_data["warning"] = (
                f"Results truncated at {self.max_list_entries} entries. "
                "Use a more specific pattern or non-recursive listing."
            )

        return ToolResult(success=True, data=result_data)

    async def _delete_file(self, params: dict[str, Any]) -> ToolResult:
        """Delete a file within the configured root."""
        path = self._resolve_path(str(params["path"]))

        if not path.exists():
            return ToolResult(success=False, error="File not found")
        if path.is_dir():
            return ToolResult(success=False, error="Path is a directory")

        path.unlink()
        return ToolResult(
            success=True,
            data={"path": str(path.relative_to(self.root))},
        )

    async def _move_file(self, params: dict[str, Any]) -> ToolResult:
        """Move or rename a file within the configured root."""
        source = self._resolve_path(str(params["source"]))
        destination = self._resolve_path(str(params["destination"]))

        if not source.exists():
            return ToolResult(success=False, error="File not found")
        if source.is_dir():
            return ToolResult(success=False, error="Source path is a directory")
        if destination.exists() and destination.is_dir():
            return ToolResult(success=False, error="Destination path is a directory")
        if destination.exists():
            return ToolResult(
                success=False,
                error="Destination file already exists",
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return ToolResult(
            success=True,
            data={
                "source": str(source.relative_to(self.root)),
                "destination": str(destination.relative_to(self.root)),
            },
        )
