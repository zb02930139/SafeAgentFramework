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

"""Controlled shell execution module for SafeAgent."""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger(__name__)

# Default PATH for subprocesses - known-safe directories only
# Covers Linux and macOS common binary locations
_DEFAULT_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"

# Environment variables that are blocked from LLM override for security.
# These can affect process execution behavior in dangerous ways.
#
# Categories:
# - Binary/library injection: PATH, LD_*, DYLD_* (macOS)
# - Shell code injection: BASH_ENV, ENV, ENVIRONMENT
# - Interpreter code injection: PYTHON*, NODE_OPTIONS, PERL5OPT, RUBYOPT,
#   JAVA_TOOL_OPTIONS
# - Debug manipulation: LD_DEBUG
_BLOCKED_ENV_OVERRIDES: frozenset[str] = frozenset(
    {
        # Binary execution control
        "PATH",
        # Linux library injection
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",  # Equivalent to LD_PRELOAD on Linux
        "LD_DEBUG",
        # macOS library injection
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        # Shell code injection
        "BASH_ENV",
        "ENV",
        "ENVIRONMENT",
        # Python code injection
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONEXECUTABLE",
        "PYTHONSTARTUP",
        # Node.js code injection
        "NODE_OPTIONS",
        # Perl code injection
        "PERL5OPT",
        "PERL5LIB",
        # Ruby code injection
        "RUBYOPT",
        "RUBYLIB",
        # Java options injection
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "JAVA_OPTS",
    }
)


class ShellModule(BaseModule):
    """Provide controlled subprocess execution with time and output limits."""

    def __init__(
        self,
        working_directory: Path | None = None,
        default_timeout: float = 30.0,
        max_timeout: float = 300.0,
        max_output_size: int = 1024 * 1024,
        allowed_env_vars: list[str] | None = None,
    ) -> None:
        """Initialise shell execution settings.

        Args:
            working_directory: Optional working directory for subprocesses.
            default_timeout: Timeout used when a tool call does not override it.
            max_timeout: Upper bound for per-request timeouts. Prevents LLMs
                from requesting indefinitely long-running processes (e.g., 11.5 days).
                Default is 300 seconds (5 minutes).
            max_output_size: Maximum bytes retained for each output stream.
            allowed_env_vars: Whitelist of host environment variable names to
                pass through to subprocesses. If None or empty, no host env vars
                are inherited (secure default). Note: The security blocklist only
                applies to LLM-provided env overrides; operators are trusted, so
                vars in this whitelist (including PATH) will override defaults.
        """
        self.working_directory = (
            working_directory.resolve() if working_directory is not None else None
        )
        if max_timeout <= 0:
            raise ValueError("max_timeout must be > 0")
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.max_output_size = max_output_size
        self.allowed_env_vars = set(allowed_env_vars) if allowed_env_vars else set()

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
                            "timeout": {"type": "number", "exclusiveMinimum": 0},
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
            (
                stdout_text,
                stderr_text,
                stdout_truncated,
                stderr_truncated,
            ) = await self._read_output_incremental(process, timeout)
        except TimeoutError:
            self._safe_kill(process)
            await process.communicate()
            return ToolResult(success=False, error="Command timed out")
        except Exception as exc:  # pragma: no cover - defensive catch
            self._safe_kill(process)
            await process.communicate()
            return ToolResult(success=False, error=str(exc))

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
        """Construct the subprocess environment with security controls.

        Security measures:
        - Starts with minimal base environment (safe PATH only)
        - Only whitelisted host env vars are inherited
        - Blocked variables (PATH, LD_PRELOAD, etc.) cannot be overridden by LLM

        Returns:
            Sanitized environment dictionary for subprocess execution.
        """
        # Start with minimal base environment
        env: dict[str, str] = {"PATH": _DEFAULT_SAFE_PATH}

        # Pass through only whitelisted host environment variables
        for key in self.allowed_env_vars:
            if key in os.environ:
                env[key] = os.environ[key]

        # Process LLM-provided env overrides
        raw_env = params.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError("env must be an object of string key/value pairs")

        for key, value in raw_env.items():
            key_str = str(key)
            # Block dangerous environment variable overrides
            if key_str in _BLOCKED_ENV_OVERRIDES:
                # Log blocked override attempts for security auditing
                logger.warning(
                    "Blocked dangerous env var override attempt: %s",
                    key_str,
                )
                continue
            env[key_str] = str(value)

        return env

    def _timeout_value(self, params: dict[str, Any]) -> float:
        """Resolve the timeout to use for this invocation.

        The timeout is clamped to max_timeout to prevent LLMs from requesting
        indefinitely long-running processes. A timeout of 0 is rejected because
        it causes immediate timeout (surprising behavior).
        """
        raw_timeout = params.get("timeout", self.default_timeout)
        timeout = float(raw_timeout)
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        if timeout == 0:
            raise ValueError("timeout must be > 0 (immediate timeout is not allowed)")
        return min(timeout, self.max_timeout)

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

    async def _read_output_incremental(
        self,
        process: asyncio.subprocess.Process,
        timeout: float,
    ) -> tuple[str, str, bool, bool]:
        """Read stdout/stderr incrementally with byte limit.

        Returns (stdout_text, stderr_text, stdout_truncated, stderr_truncated).
        Tracks total elapsed time against timeout deadline.
        Kills process if output exceeds limit (per issue #30 requirements).
        """
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Process stdout/stderr not available")

        collected_stdout = bytearray()
        collected_stderr = bytearray()
        total_bytes = 0
        stdout_truncated = False
        stderr_truncated = False
        limits_hit = asyncio.Event()
        deadline = asyncio.get_event_loop().time() + timeout

        async def read_stream(
            stream: asyncio.StreamReader,
            collected: bytearray,
            is_stdout: bool,
            lock: asyncio.Lock,
        ) -> bool:
            """Read from stream until EOF, byte limit, or deadline.

            Returns True if truncated.
            """
            nonlocal total_bytes
            nonlocal stdout_truncated
            nonlocal stderr_truncated

            truncated = False
            while not limits_hit.is_set():
                # Check deadline before each read
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Total timeout exceeded")

                try:
                    chunk = await asyncio.wait_for(
                        stream.read(65536), timeout=remaining
                    )
                except TimeoutError:
                    raise TimeoutError("Total timeout exceeded") from None
                except Exception:
                    break

                if not chunk:
                    break

                async with lock:
                    # Check limit under lock to avoid race conditions
                    if total_bytes >= self.max_output_size:
                        limits_hit.set()
                        truncated = True
                        if is_stdout:
                            stdout_truncated = True
                        else:
                            stderr_truncated = True
                        break

                    room = self.max_output_size - total_bytes
                    if len(chunk) > room:
                        # Truncate this chunk
                        chunk = chunk[:room]
                        truncated = True
                        limits_hit.set()
                        if is_stdout:
                            stdout_truncated = True
                        else:
                            stderr_truncated = True

                    collected.extend(chunk)
                    total_bytes += len(chunk)

            return truncated

        lock = asyncio.Lock()
        stdout_task = asyncio.create_task(
            read_stream(process.stdout, collected_stdout, True, lock)
        )
        stderr_task = asyncio.create_task(
            read_stream(process.stderr, collected_stderr, False, lock)
        )

        # Wait for both readers with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task),
                timeout=timeout,
            )
        except TimeoutError:
            stdout_task.cancel()
            stderr_task.cancel()
            # Wait for tasks to cancel
            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
                    timeout=1.0,
                )
            except TimeoutError:
                pass
            raise TimeoutError("Command timed out") from None

        # If limits hit, kill the process (per issue #30 requirements)
        if limits_hit.is_set():
            self._safe_kill(process)
            # Drain any remaining data to prevent broken pipe issues
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:  # noqa: UP041
                pass
        else:
            # Wait for process to complete normally
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                self._safe_kill(process)
                raise TimeoutError("Command timed out") from None

        stdout_text = collected_stdout.decode("utf-8", errors="replace")
        stderr_text = collected_stderr.decode("utf-8", errors="replace")

        return stdout_text, stderr_text, stdout_truncated, stderr_truncated

    def _safe_kill(self, process: asyncio.subprocess.Process) -> None:
        """Kill process safely, ignoring ProcessLookupError if already dead."""
        try:
            process.kill()
        except ProcessLookupError:
            pass  # Already terminated
