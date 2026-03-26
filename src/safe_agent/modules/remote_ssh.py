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

"""Remote SSH module for SafeAgent - controlled SSH access to remote hosts."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SSHCredential:
    """SSH credentials for a host connection.

    Attributes:
        username: SSH username for authentication.
        password: Optional password for password authentication.
        key_path: Optional path to private key file for key authentication.
        key_passphrase: Optional passphrase for encrypted private key.
    """

    username: str
    password: str | None = None
    key_path: Path | str | None = None
    key_passphrase: str | None = None


class RemoteSSHModule(BaseModule):
    """Provide controlled SSH access to remote hosts with session management.

    This module manages persistent SSH sessions keyed by hostname, with
    configurable timeouts, output limits, and credential isolation. Credentials
    are provided at construction time by the operator and are never exposed
    to the LLM.

    Security features:
    - Credential isolation: passwords/keys never exposed in tool parameters or results
    - Output size limits to prevent memory exhaustion
    - Timeout enforcement on all SSH operations
    - Session idle timeout with automatic cleanup
    - Command passed directly to remote shell (no local interpretation)
    """

    def __init__(
        self,
        credentials: dict[str, SSHCredential] | None = None,
        default_timeout: float = 30.0,
        max_timeout: float = 300.0,
        max_output_size: int = 1024 * 1024,  # 1MB
        session_idle_timeout: float = 300.0,  # 5 minutes
        known_hosts_path: Path | None = None,
        max_sessions: int = 20,
    ) -> None:
        """Initialize remote SSH module with session and security settings.

        Args:
            credentials: Mapping of hostname to SSHCredential for authentication.
                Credentials are operator-provided and never exposed to LLM.
            default_timeout: Default timeout for SSH operations in seconds.
            max_timeout: Maximum allowed timeout for any operation.
            max_output_size: Maximum bytes to retain from command output.
            session_idle_timeout: Seconds before idle sessions are closed.
            known_hosts_path: Optional path to known_hosts file for host verification.
            max_sessions: Maximum number of concurrent SSH sessions allowed.
        """
        if max_timeout <= 0:
            raise ValueError("max_timeout must be > 0")
        if session_idle_timeout < 0:
            raise ValueError("session_idle_timeout must be >= 0")

        self._credentials = credentials or {}
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._max_output_size = max_output_size
        self._session_idle_timeout = session_idle_timeout
        self._known_hosts_path = known_hosts_path

        # Active sessions: hostname -> (connection, last_used_timestamp, username)
        self._sessions: dict[str, tuple[Any, float, str]] = {}
        self._session_lock = asyncio.Lock()
        self._max_sessions = max_sessions  # Limit concurrent connections

    def describe(self) -> ModuleDescriptor:
        """Return the remote SSH module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="remote_ssh",
            description=(
                "Connect to remote hosts via SSH, execute commands, "
                "and push configuration changes. Sessions persist by hostname."
            ),
            tools=[
                ToolDescriptor(
                    name="remote_ssh:connect",
                    description="Open an SSH session to a remote host.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "hostname": {"type": "string"},
                            "username": {"type": "string"},
                            "port": {"type": "integer", "default": 22},
                        },
                        "required": ["hostname", "username"],
                        "additionalProperties": False,
                    },
                    action="remote_ssh:Connect",
                    resource_param=["hostname"],
                    condition_keys=[
                        "remote_ssh:Hostname",
                        "remote_ssh:Username",
                    ],
                ),
                ToolDescriptor(
                    name="remote_ssh:execute_command",
                    description="Run a command on a connected remote host.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "hostname": {"type": "string"},
                            "command": {"type": "string"},
                            "timeout": {"type": "integer", "exclusiveMinimum": 0},
                        },
                        "required": ["hostname", "command"],
                        "additionalProperties": False,
                    },
                    action="remote_ssh:ExecuteCommand",
                    resource_param=["hostname"],
                    condition_keys=[
                        "remote_ssh:Hostname",
                        "remote_ssh:Username",
                        "remote_ssh:CommandName",
                    ],
                ),
                ToolDescriptor(
                    name="remote_ssh:push_config",
                    description=(
                        "Push a configuration change to a remote device. "
                        "Use dry_run=True to preview without applying."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "hostname": {"type": "string"},
                            "config": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["merge", "replace", "set"],
                            },
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["hostname", "config"],
                        "additionalProperties": False,
                    },
                    action="remote_ssh:PushConfig",
                    resource_param=["hostname"],
                    condition_keys=[
                        "remote_ssh:Hostname",
                        "remote_ssh:Username",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve condition keys for policy evaluation.

        Derives:
        - remote_ssh:Hostname from hostname parameter
        - remote_ssh:Username from username parameter or stored session
        - remote_ssh:CommandName from command parameter (first token)
        """
        result: dict[str, Any] = {}

        # Hostname is always available
        hostname = params.get("hostname")
        if hostname:
            result["remote_ssh:Hostname"] = str(hostname)

        # Username from params or from stored credentials/session
        username = params.get("username")
        if username:
            result["remote_ssh:Username"] = str(username)
        elif hostname:
            # Try to get username from credentials or active session
            hostname_str = str(hostname)
            if hostname_str in self._credentials:
                result["remote_ssh:Username"] = self._credentials[hostname_str].username
            else:
                # Check active session for stored username
                # Note: _session_lock access would need to be async, skip for now
                pass

        # CommandName from command parameter (first token, like ShellModule)
        command = params.get("command")
        if command:
            try:
                command_parts = shlex.split(str(command))
                if command_parts:
                    result["remote_ssh:CommandName"] = command_parts[0]
            except ValueError:
                pass  # Malformed command, skip

        return result

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute the requested SSH tool."""
        tool_suffix = tool_name.removeprefix("remote_ssh:")

        try:
            if tool_suffix == "connect":
                return await self._execute_connect(params)
            elif tool_suffix == "execute_command":
                return await self._execute_command(params)
            elif tool_suffix == "push_config":
                return await self._execute_push_config(params)
            else:
                return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

    async def _execute_connect(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Open an SSH session to a host."""
        import asyncssh

        hostname = params.get("hostname")
        username = params.get("username")
        port = params.get("port", 22)

        if not hostname:
            return ToolResult(success=False, error="hostname is required")
        if not username:
            return ToolResult(success=False, error="username is required")

        hostname_str = str(hostname)
        username_str = str(username)
        port_int = int(port) if port is not None else 22

        # Check for existing session
        async with self._session_lock:
            if hostname_str in self._sessions:
                existing_conn, _, stored_user = self._sessions[hostname_str]
                if not existing_conn.is_closed():
                    # Update last used time
                    self._sessions[hostname_str] = (
                        existing_conn,
                        time.monotonic(),
                        stored_user,
                    )
                    return ToolResult(
                        success=True,
                        data={
                            "hostname": hostname_str,
                            "status": "already_connected",
                        },
                    )

            # Check session limit BEFORE creating new connection (prevent resource leak)
            if len(self._sessions) >= self._max_sessions:
                return ToolResult(
                    success=False,
                    error=f"Maximum sessions ({self._max_sessions}) reached",
                )

        # Get credentials - prefer params username but auth from stored credentials
        cred = self._credentials.get(hostname_str)
        connect_kwargs: dict[str, Any] = {
            "host": hostname_str,
            "port": port_int,
            "username": username_str,
        }

        if cred:
            if cred.password:
                connect_kwargs["password"] = cred.password
            elif cred.key_path:
                connect_kwargs["client_keys"] = [str(cred.key_path)]
                if cred.key_passphrase:
                    connect_kwargs["passphrase"] = cred.key_passphrase
        else:
            # No stored credentials - try default SSH key auth
            # This allows agent's default SSH key to work
            pass

        # Known hosts handling
        if self._known_hosts_path:
            connect_kwargs["known_hosts"] = str(self._known_hosts_path)
        else:
            # SECURITY WARNING: known_hosts=None disables host key verification.
            # This allows MITM attacks. Operators should provide known_hosts_path
            # in production. None is allowed for testing environments only.
            logger.warning(
                "Connecting to %s without host key verification (known_hosts not set). "
                "This is insecure for production use.",
                hostname_str,
            )
            connect_kwargs["known_hosts"] = None

        timeout = self._default_timeout

        try:
            conn = await asyncio.wait_for(
                asyncssh.connect(**connect_kwargs),
                timeout=timeout,
            )
        except TimeoutError:
            return ToolResult(success=False, error="Connection timed out")
        except asyncssh.PermissionDenied:
            return ToolResult(success=False, error="Permission denied")
        except asyncssh.HostKeyNotVerifiable:
            return ToolResult(success=False, error="Host key verification failed")
        except OSError as e:
            return ToolResult(success=False, error=f"Connection failed: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"SSH error: {e}")

        # Store the new session (limit already checked earlier)
        async with self._session_lock:
            self._sessions[hostname_str] = (conn, time.monotonic(), username_str)

        return ToolResult(
            success=True,
            data={"hostname": hostname_str, "status": "connected"},
        )

    async def _execute_command(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Run a command on a connected host."""
        hostname = params.get("hostname")
        command = params.get("command")

        if not hostname:
            return ToolResult(success=False, error="hostname is required")
        if not command:
            return ToolResult(success=False, error="command is required")

        timeout = self._resolve_timeout(params)
        hostname_str = str(hostname)

        # Get or establish session
        conn = await self._get_session(hostname_str)
        if conn is None:
            return ToolResult(
                success=False,
                error=f"No session for {hostname_str}. Call connect first.",
            )

        # Execute command with timeout and output limit
        try:
            result = await asyncio.wait_for(
                conn.run(str(command), timeout=timeout),
                timeout=timeout + 1,  # Extra second for asyncssh overhead
            )
        except TimeoutError:
            return ToolResult(success=False, error="Command timed out")
        except Exception as e:
            return ToolResult(success=False, error=f"Command execution failed: {e}")

        # Apply output size limits
        stdout = result.stdout
        stderr = result.stderr
        metadata: dict[str, Any] = {}

        if len(stdout.encode("utf-8", errors="replace")) > self._max_output_size:
            stdout_bytes = stdout.encode("utf-8", errors="replace")[
                : self._max_output_size
            ]
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            metadata["output_truncated"] = True
            metadata["stdout_truncated"] = True
            metadata["max_output_size"] = self._max_output_size

        if len(stderr.encode("utf-8", errors="replace")) > self._max_output_size:
            stderr_bytes = stderr.encode("utf-8", errors="replace")[
                : self._max_output_size
            ]
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            metadata["output_truncated"] = True
            metadata["stderr_truncated"] = True
            metadata["max_output_size"] = self._max_output_size

        return ToolResult(
            success=True,
            data={
                "hostname": hostname_str,
                "stdout": stdout,
                "stderr": stderr,
                "exit_status": result.exit_status,
            },
            metadata=metadata,
        )

    async def _execute_push_config(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Push a configuration change to a remote device."""
        hostname = params.get("hostname")
        config = params.get("config")
        mode = params.get("mode", "merge")
        dry_run = params.get("dry_run", False)

        if not hostname:
            return ToolResult(success=False, error="hostname is required")
        if not config:
            return ToolResult(success=False, error="config is required")

        # Validate mode against allowed values
        valid_modes = {"merge", "replace", "set"}
        if mode not in valid_modes:
            return ToolResult(
                success=False,
                error=(
                    f"Invalid mode '{mode}'. "
                    f"Must be one of: {', '.join(sorted(valid_modes))}"
                ),
            )

        timeout = self._resolve_timeout(params)
        hostname_str = str(hostname)

        # Get or establish session
        conn = await self._get_session(hostname_str)
        if conn is None:
            return ToolResult(
                success=False,
                error=f"No session for {hostname_str}. Call connect first.",
            )

        # Build config push command based on mode
        # This is a generic implementation - real devices use vendor-specific commands
        # For network devices, this would typically use NETCONF, expect scripts, or
        # vendor-specific CLIs (Juniper: "load merge", Cisco: "copy running-config")
        config_command = self._build_config_command(config, mode, dry_run)

        try:
            result = await asyncio.wait_for(
                conn.run(config_command, timeout=timeout),
                timeout=timeout + 1,
            )
        except TimeoutError:
            return ToolResult(success=False, error="Config push timed out")
        except Exception as e:
            return ToolResult(success=False, error=f"Config push failed: {e}")

        # Apply output size limits
        stdout = result.stdout
        stderr = result.stderr
        metadata: dict[str, Any] = {"dry_run": dry_run}

        if len(stdout.encode("utf-8", errors="replace")) > self._max_output_size:
            stdout_bytes = stdout.encode("utf-8", errors="replace")[
                : self._max_output_size
            ]
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            metadata["output_truncated"] = True
            metadata["stdout_truncated"] = True

        if len(stderr.encode("utf-8", errors="replace")) > self._max_output_size:
            stderr_bytes = stderr.encode("utf-8", errors="replace")[
                : self._max_output_size
            ]
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            metadata["stderr_truncated"] = True

        return ToolResult(
            success=result.exit_status == 0,
            data={
                "hostname": hostname_str,
                "stdout": stdout,
                "stderr": stderr,
                "exit_status": result.exit_status,
                "mode": mode,
                "dry_run": dry_run,
            },
            metadata=metadata,
        )

    def _build_config_command(
        self,
        config: str,
        mode: str,
        dry_run: bool,
    ) -> str:
        """Build a configuration push command.

        Note: This is a simplified implementation. Real network device
        configuration would use vendor-specific commands or NETCONF.
        """
        # SECURITY: Use shlex.quote to prevent shell injection via config
        quoted_config = shlex.quote(config)

        # For dry_run, we echo the config without applying
        if dry_run:
            return f"echo 'DRY RUN - would apply config' && echo {quoted_config}"

        # Generic implementation - echo the config
        # In production, this would be device-specific (Juniper, Cisco, etc.)
        return f"echo {quoted_config}"

    async def _get_session(self, hostname: str) -> Any | None:
        """Get an active SSH session for a hostname, or None if not connected."""
        async with self._session_lock:
            # Clean up expired sessions first
            await self._cleanup_expired_sessions()

            if hostname not in self._sessions:
                return None

            conn, _last_used, username = self._sessions[hostname]

            # Check if connection is still valid
            if conn.is_closed():
                del self._sessions[hostname]
                return None

            # Update last used time
            self._sessions[hostname] = (conn, time.monotonic(), username)
            return conn

    async def _cleanup_expired_sessions(self) -> None:
        """Close and remove sessions that have been idle too long.

        Must be called while holding _session_lock.
        """
        if self._session_idle_timeout <= 0:
            return

        now = time.monotonic()
        expired = []

        for _hostname, (_conn, last_used, _username) in self._sessions.items():
            if now - last_used > self._session_idle_timeout:
                expired.append(_hostname)

        for hostname in expired:
            conn, _, _ = self._sessions.pop(hostname)
            try:
                conn.close()
            except Exception as e:
                logger.debug("Error closing expired session to %s: %s", hostname, e)

    def _resolve_timeout(self, params: dict[str, Any]) -> float:
        """Resolve and clamp the timeout for an operation."""
        raw_timeout = params.get("timeout", self._default_timeout)
        timeout = float(raw_timeout)
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        if timeout == 0:
            raise ValueError("timeout must be > 0 (immediate timeout is not allowed)")
        return min(timeout, self._max_timeout)

    async def close_all_sessions(self) -> None:
        """Close all active SSH sessions."""
        async with self._session_lock:
            for hostname, (conn, _, _) in self._sessions.items():
                try:
                    conn.close()
                    logger.debug("Closed session to %s", hostname)
                except Exception as e:  # nosec B110
                    # Cleanup: ignore errors closing sessions
                    logger.debug("Error closing session to %s: %s", hostname, e)
            self._sessions.clear()

    def __del__(self) -> None:
        """Clean up sessions on garbage collection."""
        # Note: async cleanup in __del__ is problematic, but we can sync close
        sessions = getattr(self, "_sessions", None)
        if sessions is None:
            return
        for _hostname, (conn, _, _) in list(sessions.items()):
            try:
                if not conn.is_closed():
                    conn.close()
            except Exception:  # nosec B110 # noqa: S110
                # Cleanup: ignore errors in destructor
                pass
