"""Controlled shell execution module for SafeAgent."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


class ShellModule(BaseModule):
    """Provide controlled subprocess execution with time and output limits."""

    def __init__(
        self,
        working_directory: Path | None = None,
        default_timeout: float = 30.0,
        max_output_size: int = 1024 * 1024,
    ) -> None:
        """Initialise shell execution settings.

        Args:
            working_directory: Optional working directory for subprocesses.
            default_timeout: Timeout used when a tool call does not override it.
            max_output_size: Maximum bytes retained for each output stream.
        """
        self.working_directory = (
            working_directory.resolve() if working_directory is not None else None
        )
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size

    def describe(self) -> ModuleDescriptor:
        """Return the shell module descriptor and tool definition."""
        return ModuleDescriptor(
            namespace="shell",
            description="Controlled command execution with timeout and output limits.",
            tools=[
                ToolDescriptor(
                    name="shell:execute",
                    description=(
                        "Execute a command with optional args, timeout, and env."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "timeout": {"type": "number", "minimum": 0},
                            "env": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                    action="shell:Execute",
                    resource_param=["command"],
                    condition_keys=[
                        "shell:CommandName",
                        "shell:WorkingDirectory",
                    ],
                )
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the command name and working directory for policy checks."""
        if tool_name.removeprefix("shell:") != "execute":
            return {}

        command = str(params.get("command", ""))
        try:
            command_parts = shlex.split(command)
        except ValueError:
            command_parts = []
        command_name = command_parts[0] if command_parts else ""
        working_directory = self._working_directory_string()

        return {
            "shell:CommandName": command_name,
            "shell:WorkingDirectory": working_directory,
        }

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute a configured subprocess for the requested tool."""
        if tool_name.removeprefix("shell:") != "execute":
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        try:
            argv = self._build_argv(params)
            timeout = self._timeout_value(params)
            env = self._build_env(params)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._cwd(),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return ToolResult(success=False, error="Command not found")
        except Exception as exc:  # pragma: no cover - defensive catch
            return ToolResult(success=False, error=str(exc))

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return ToolResult(success=False, error="Command timed out")
        except Exception as exc:  # pragma: no cover - defensive catch
            process.kill()
            await process.communicate()
            return ToolResult(success=False, error=str(exc))

        stdout_text, stdout_truncated = self._decode_and_truncate(stdout_bytes)
        stderr_text, stderr_truncated = self._decode_and_truncate(stderr_bytes)

        metadata: dict[str, Any] = {}
        if stdout_truncated or stderr_truncated:
            metadata["output_truncated"] = True
            metadata["max_output_size"] = self.max_output_size
            metadata["stdout_truncated"] = stdout_truncated
            metadata["stderr_truncated"] = stderr_truncated

        return ToolResult(
            success=True,
            data={
                "stdout": stdout_text,
                "stderr": stderr_text,
                "return_code": process.returncode,
            },
            metadata=metadata,
        )

    def _build_argv(self, params: dict[str, Any]) -> list[str]:
        """Construct argv from the command string plus any explicit args."""
        command = str(params.get("command", ""))
        command_parts = shlex.split(command)
        if not command_parts:
            raise ValueError("Command must not be empty")

        raw_args = params.get("args", [])
        if not isinstance(raw_args, list):
            raise ValueError("args must be a list of strings")

        explicit_args = [str(arg) for arg in raw_args]
        return [*command_parts, *explicit_args]

    def _build_env(self, params: dict[str, Any]) -> dict[str, str]:
        """Construct the subprocess environment."""
        env = os.environ.copy()
        raw_env = params.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError("env must be an object of string key/value pairs")

        env.update({str(key): str(value) for key, value in raw_env.items()})
        return env

    def _timeout_value(self, params: dict[str, Any]) -> float:
        """Resolve the timeout to use for this invocation."""
        raw_timeout = params.get("timeout", self.default_timeout)
        timeout = float(raw_timeout)
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        return timeout

    def _cwd(self) -> str | None:
        """Return the configured working directory for subprocess execution."""
        if self.working_directory is None:
            return None
        return str(self.working_directory)

    def _working_directory_string(self) -> str:
        """Return the effective working directory as a string."""
        if self.working_directory is not None:
            return str(self.working_directory)
        return str(Path.cwd())

    def _decode_and_truncate(self, output: bytes) -> tuple[str, bool]:
        """Decode subprocess output, truncating to the configured byte limit."""
        truncated = len(output) > self.max_output_size
        if truncated:
            output = output[: self.max_output_size]
        return output.decode("utf-8", errors="replace"), truncated
