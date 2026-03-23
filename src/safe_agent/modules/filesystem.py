"""Sandboxed filesystem module for SafeAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiofiles

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


class FilesystemModule(BaseModule):
    """Provide sandboxed filesystem operations rooted at a single directory."""

    DEFAULT_MAX_WRITE_SIZE = 10 * 1024 * 1024  # 10MB default

    def __init__(self, root: Path, max_write_size: int = DEFAULT_MAX_WRITE_SIZE) -> None:
        """Initialise the module with an allowed filesystem root.

        Args:
            root: The allowed filesystem root directory.
            max_write_size: Maximum bytes allowed per file write (default 10MB).
        """
        self.root = root.resolve()
        self.max_write_size = max_write_size

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
                error=f"File size {len(content_bytes)} bytes exceeds maximum {self.max_write_size} bytes",
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
        """List entries in a directory, optionally recursively and by pattern."""
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
        entries: list[dict[str, str | bool]] = [
            {
                "path": str(entry.relative_to(self.root)),
                "is_directory": entry.is_dir(),
            }
            for entry in iterator
        ]
        entries.sort(key=lambda item: str(item["path"]))

        return ToolResult(success=True, data={"entries": entries})

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
