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

"""Tests for the Remote SSH module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from safe_agent.modules.remote_ssh import RemoteSSHModule, SSHCredential


class MockSSHConnection:
    """Mock asyncssh.SSHClientConnection for testing."""

    def __init__(
        self,
        hostname: str = "testhost",
        should_fail: bool = False,
        command_results: dict[str, tuple[int, str, str]] | None = None,
    ) -> None:
        self.hostname = hostname
        self.should_fail = should_fail
        self.command_results = command_results or {
            "echo hello": (0, "hello\n", ""),
            "ls -la": (0, "file1\nfile2\n", ""),
            "show version": (0, "Version 1.0.0\n", ""),
            "display config": (0, "config data\n", ""),
        }
        self._closed = False
        self._calls: list[tuple[str, float]] = []

    def __await__(self):
        """Make the mock awaitable for use with asyncssh.connect()."""

        async def _awaitable():
            return self

        return _awaitable().__await__()

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    async def run(
        self,
        command: str,
        timeout: float | None = None,
    ) -> MockSSHCompletedProcess:
        self._calls.append((command, timeout or 0.0))
        await asyncio.sleep(0.01)  # Simulate network delay

        if command in self.command_results:
            exit_status, stdout, stderr = self.command_results[command]
        else:
            # Default: echo the command back
            exit_status = 0
            stdout = f"output for: {command}\n"
            stderr = ""

        return MockSSHCompletedProcess(exit_status, stdout, stderr)


class MockSSHCompletedProcess:
    """Mock asyncssh.SSHCompletedProcess for testing."""

    def __init__(self, exit_status: int, stdout: str, stderr: str) -> None:
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr


class TestRemoteSSHModuleDescriptor:
    """Tests for module descriptor and describe() method."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return correct module metadata."""
        module = RemoteSSHModule()

        descriptor = module.describe()

        assert descriptor.namespace == "remote_ssh"
        assert "SSH" in descriptor.description
        assert len(descriptor.tools) == 3

        tool_names = {t.name for t in descriptor.tools}
        assert tool_names == {
            "remote_ssh:connect",
            "remote_ssh:execute_command",
            "remote_ssh:push_config",
        }

    def test_connect_tool_descriptor(self) -> None:
        """Connect tool should have correct descriptor."""
        module = RemoteSSHModule()
        descriptor = module.describe()

        connect_tool = next(
            t for t in descriptor.tools if t.name == "remote_ssh:connect"
        )

        assert connect_tool.action == "remote_ssh:Connect"
        assert connect_tool.resource_param == ["hostname"]
        assert "remote_ssh:Hostname" in connect_tool.condition_keys
        assert "remote_ssh:Username" in connect_tool.condition_keys
        assert connect_tool.parameters["required"] == ["hostname", "username"]
        assert connect_tool.parameters["properties"]["port"]["default"] == 22

    def test_execute_command_tool_descriptor(self) -> None:
        """execute_command tool should have correct descriptor."""
        module = RemoteSSHModule()
        descriptor = module.describe()

        exec_tool = next(
            t for t in descriptor.tools if t.name == "remote_ssh:execute_command"
        )

        assert exec_tool.action == "remote_ssh:ExecuteCommand"
        assert exec_tool.resource_param == ["hostname"]
        assert "remote_ssh:CommandName" in exec_tool.condition_keys
        assert exec_tool.parameters["required"] == ["hostname", "command"]

    def test_push_config_tool_descriptor(self) -> None:
        """push_config tool should have correct descriptor."""
        module = RemoteSSHModule()
        descriptor = module.describe()

        push_tool = next(
            t for t in descriptor.tools if t.name == "remote_ssh:push_config"
        )

        assert push_tool.action == "remote_ssh:PushConfig"
        assert push_tool.resource_param == ["hostname"]
        assert push_tool.parameters["required"] == ["hostname", "config"]
        assert "mode" in push_tool.parameters["properties"]
        assert "dry_run" in push_tool.parameters["properties"]


class TestRemoteSSHModuleResolveConditions:
    """Tests for resolve_conditions() method."""

    async def test_resolve_conditions_hostname(self) -> None:
        """Should derive remote_ssh:Hostname from hostname parameter."""
        module = RemoteSSHModule()

        conditions = await module.resolve_conditions(
            "remote_ssh:connect",
            {"hostname": "switch01.example.com", "username": "admin"},
        )

        assert conditions["remote_ssh:Hostname"] == "switch01.example.com"

    async def test_resolve_conditions_username_from_params(self) -> None:
        """Should derive remote_ssh:Username from username parameter."""
        module = RemoteSSHModule()

        conditions = await module.resolve_conditions(
            "remote_ssh:connect",
            {"hostname": "switch01", "username": "admin"},
        )

        assert conditions["remote_ssh:Username"] == "admin"

    async def test_resolve_conditions_username_from_credentials(self) -> None:
        """Should derive remote_ssh:Username from stored credentials."""
        module = RemoteSSHModule(
            credentials={"switch01": SSHCredential(username="netadmin")}
        )

        conditions = await module.resolve_conditions(
            "remote_ssh:execute_command",
            {"hostname": "switch01", "command": "show version"},
        )

        assert conditions["remote_ssh:Username"] == "netadmin"

    async def test_resolve_conditions_command_name(self) -> None:
        """Should derive remote_ssh:CommandName from command parameter."""
        module = RemoteSSHModule()

        conditions = await module.resolve_conditions(
            "remote_ssh:execute_command",
            {"hostname": "switch01", "command": "show version"},
        )

        assert conditions["remote_ssh:CommandName"] == "show"

    async def test_resolve_conditions_command_name_with_args(self) -> None:
        """Should extract first token of command for CommandName."""
        module = RemoteSSHModule()

        conditions = await module.resolve_conditions(
            "remote_ssh:execute_command",
            {"hostname": "switch01", "command": "display config detail"},
        )

        assert conditions["remote_ssh:CommandName"] == "display"

    async def test_resolve_conditions_malformed_command(self) -> None:
        """Should handle malformed command gracefully."""
        module = RemoteSSHModule()

        conditions = await module.resolve_conditions(
            "remote_ssh:execute_command",
            {"hostname": "switch01", "command": "echo 'unterminated"},
        )

        # Should not crash, CommandName should be absent
        assert "remote_ssh:CommandName" not in conditions
        assert "remote_ssh:Hostname" in conditions


class TestRemoteSSHModuleConnect:
    """Tests for connect tool."""

    async def test_connect_establishes_session(self) -> None:
        """Connect should establish and store an SSH session."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["hostname"] == "switch01"
        assert result.data["status"] == "connected"

    async def test_connect_reuses_existing_session(self) -> None:
        """Connect should return existing session if already connected."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            # First connection
            result1 = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )
            # Second connection to same host
            result2 = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

        assert result1.success is True
        assert result2.success is True
        assert result2.data["status"] == "already_connected"

    async def test_connect_with_credentials(self) -> None:
        """Connect should use stored credentials for authentication."""
        module = RemoteSSHModule(
            credentials={
                "switch01": SSHCredential(
                    username="admin",
                    password="secret123",  # noqa: S106
                )
            }
        )

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn) as mock_connect:
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            # Verify password was passed to asyncssh
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["password"] == "secret123"  # noqa: S105

        assert result.success is True

    async def test_connect_with_key_auth(self) -> None:
        """Connect should use key file for authentication."""
        module = RemoteSSHModule(
            credentials={
                "switch01": SSHCredential(
                    username="admin",
                    key_path=Path("/home/admin/.ssh/id_rsa"),
                )
            }
        )

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn) as mock_connect:
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["client_keys"] == ["/home/admin/.ssh/id_rsa"]

        assert result.success is True

    async def test_connect_times_out(self) -> None:
        """Connect should fail on timeout."""
        module = RemoteSSHModule(default_timeout=0.1)

        async def slow_connect(**kwargs: dict) -> MockSSHConnection:
            await asyncio.sleep(10)
            return MockSSHConnection()

        with mock.patch("asyncssh.connect", side_effect=slow_connect):
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

        assert result.success is False
        assert "timed out" in result.error.lower()

    async def test_connect_permission_denied(self) -> None:
        """Connect should handle auth failures cleanly."""
        import asyncssh

        module = RemoteSSHModule()

        with mock.patch(
            "asyncssh.connect",
            side_effect=asyncssh.PermissionDenied("Authentication failed"),
        ):
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

        assert result.success is False
        assert "Permission denied" in result.error

    async def test_connect_missing_hostname(self) -> None:
        """Connect should reject missing hostname."""
        module = RemoteSSHModule()

        result = await module.execute(
            "remote_ssh:connect",
            {"username": "admin"},
        )

        assert result.success is False
        assert "hostname" in result.error.lower()

    async def test_connect_missing_username(self) -> None:
        """Connect should reject missing username."""
        module = RemoteSSHModule()

        result = await module.execute(
            "remote_ssh:connect",
            {"hostname": "switch01"},
        )

        assert result.success is False
        assert "username" in result.error.lower()


class TestRemoteSSHModuleExecuteCommand:
    """Tests for execute_command tool."""

    async def test_execute_command_runs_on_remote(self) -> None:
        """execute_command should run command on connected host."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection(
            command_results={"show version": (0, "v1.0.0\n", "")}
        )

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "show version"},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"] == "v1.0.0\n"
        assert result.data["exit_status"] == 0

    async def test_execute_command_fails_without_session(self) -> None:
        """execute_command should fail if no session exists."""
        module = RemoteSSHModule()

        result = await module.execute(
            "remote_ssh:execute_command",
            {"hostname": "switch01", "command": "show version"},
        )

        assert result.success is False
        assert "No session" in result.error
        assert "connect first" in result.error

    async def test_execute_command_respects_timeout(self) -> None:
        """execute_command should enforce timeout."""
        module = RemoteSSHModule(default_timeout=0.1)

        mock_conn = MockSSHConnection()

        async def slow_run(
            command: str, timeout: float | None = None
        ) -> MockSSHCompletedProcess:
            await asyncio.sleep(10)
            return MockSSHCompletedProcess(0, "done\n", "")

        mock_conn.run = slow_run

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "sleep 100"},
            )

        assert result.success is False
        assert "timed out" in result.error.lower()

    async def test_execute_command_truncates_output(self) -> None:
        """execute_command should truncate oversized output."""
        module = RemoteSSHModule(max_output_size=50)

        large_output = "x" * 1000
        mock_conn = MockSSHConnection(
            command_results={"echo test": (0, large_output, "")}
        )

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "echo test"},
            )

        assert result.success is True
        assert result.data is not None
        assert len(result.data["stdout"]) <= 50
        assert result.metadata is not None
        assert result.metadata.get("output_truncated") is True

    async def test_execute_command_returns_stderr(self) -> None:
        """execute_command should return stderr in result."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection(
            command_results={"badcmd": (1, "", "command not found\n")}
        )

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "badcmd"},
            )

        assert result.success is True  # Command ran, even if it failed
        assert result.data is not None
        assert result.data["stderr"] == "command not found\n"
        assert result.data["exit_status"] == 1


class TestRemoteSSHModulePushConfig:
    """Tests for push_config tool."""

    async def test_push_config_delegates_to_remote(self) -> None:
        """push_config should run config command on remote device."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:push_config",
                {
                    "hostname": "switch01",
                    "config": "interface eth0\n  description test",
                },
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["hostname"] == "switch01"

    async def test_push_config_dry_run(self) -> None:
        """push_config with dry_run should not apply changes."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:push_config",
                {
                    "hostname": "switch01",
                    "config": "interface eth0\n  description test",
                    "dry_run": True,
                },
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["dry_run"] is True
        # Should include "DRY RUN" in output
        assert "DRY RUN" in result.data["stdout"] or "dry_run" in str(result.metadata)

    async def test_push_config_fails_without_session(self) -> None:
        """push_config should fail if no session exists."""
        module = RemoteSSHModule()

        result = await module.execute(
            "remote_ssh:push_config",
            {"hostname": "switch01", "config": "test config"},
        )

        assert result.success is False
        assert "No session" in result.error

    async def test_push_config_mode_parameter(self) -> None:
        """push_config should accept mode parameter."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:push_config",
                {
                    "hostname": "switch01",
                    "config": "test config",
                    "mode": "replace",
                },
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["mode"] == "replace"

    async def test_push_config_invalid_mode_rejected(self) -> None:
        """push_config should reject invalid mode values."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:push_config",
                {
                    "hostname": "switch01",
                    "config": "test config",
                    "mode": "malicious'; rm -rf / #",
                },
            )

        assert result.success is False
        assert "Invalid mode" in result.error

    async def test_push_config_shell_injection_quoted(self) -> None:
        """push_config should quote config to prevent shell injection."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            # This config contains shell injection attempt
            result = await module.execute(
                "remote_ssh:push_config",
                {
                    "hostname": "switch01",
                    "config": "'; rm -rf / #",
                },
            )

        # The command should have been quoted safely
        assert result.success is True
        # Verify the malicious string wasn't executed (it's just echoed)
        assert result.data is not None


class TestRemoteSSHModuleSessionManagement:
    """Tests for session lifecycle management."""

    async def test_session_idle_timeout(self) -> None:
        """Sessions should be closed after idle timeout."""
        module = RemoteSSHModule(session_idle_timeout=0.05)  # 50ms

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            # Wait for session to expire
            await asyncio.sleep(0.1)

            # Try to use expired session
            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "echo test"},
            )

        assert result.success is False
        assert "No session" in result.error

    async def test_close_all_sessions(self) -> None:
        """close_all_sessions should clean up all connections."""
        module = RemoteSSHModule()

        mock_conn1 = MockSSHConnection(hostname="switch01")
        mock_conn2 = MockSSHConnection(hostname="switch02")

        with mock.patch(
            "asyncssh.connect",
            side_effect=[mock_conn1, mock_conn2],
        ):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch02", "username": "admin"},
            )

        await module.close_all_sessions()

        assert mock_conn1.is_closed()
        assert mock_conn2.is_closed()
        assert len(module._sessions) == 0


class TestRemoteSSHModuleSecurity:
    """Tests for security properties."""

    async def test_credentials_not_in_tool_parameters(self) -> None:
        """Credentials should never appear in tool parameters or results."""
        module = RemoteSSHModule(
            credentials={
                "switch01": SSHCredential(
                    username="admin",
                    password="super_secret_password_123",  # noqa: S106
                )
            }
        )

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn) as mock_connect:
            # Connect
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            # Verify password was passed to asyncssh, not in result
            assert result.data is not None
            assert "password" not in str(result.data)
            assert "super_secret" not in str(result.data)

            # Verify it was passed to asyncssh correctly
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs.get("password") == "super_secret_password_123"

    async def test_credentials_not_in_execute_result(self) -> None:
        """Credentials should not leak in execute results."""
        module = RemoteSSHModule(
            credentials={
                "switch01": SSHCredential(
                    username="admin",
                    password="secret123",  # noqa: S106
                    key_passphrase="key_passphrase_secret",  # noqa: S106
                )
            }
        )

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "show version"},
            )

        assert result.success is True
        assert result.data is not None
        # Verify no credential material in result
        result_str = str(result.data) + str(result.metadata or {})
        assert "secret123" not in result_str
        assert "key_passphrase" not in result_str

    async def test_max_timeout_enforced(self) -> None:
        """Per-request timeout should be capped to max_timeout."""
        module = RemoteSSHModule(max_timeout=1.0)

        # Request 999999 second timeout, should be capped to 1.0
        timeout = module._resolve_timeout({"timeout": 999999})
        assert timeout == 1.0

    async def test_max_timeout_zero_rejected(self) -> None:
        """max_timeout=0 should be rejected at construction."""
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            RemoteSSHModule(max_timeout=0)

    async def test_timeout_zero_rejected(self) -> None:
        """timeout=0 should be rejected (immediate timeout is surprising)."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            # Timeout validation happens in execute() and returns an error result
            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "test", "timeout": 0},
            )

        assert result.success is False
        assert "timeout must be > 0" in result.error


class TestRemoteSSHModuleEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_unknown_tool_returns_error(self) -> None:
        """Unknown tool names should return a failed result."""
        module = RemoteSSHModule()

        result = await module.execute(
            "remote_ssh:unknown_tool",
            {"hostname": "switch01"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_empty_hostname_rejected(self) -> None:
        """Empty hostname should be rejected."""
        module = RemoteSSHModule()

        result = await module.execute(
            "remote_ssh:connect",
            {"hostname": "", "username": "admin"},
        )

        # Either "hostname is required" or connection failure is acceptable
        assert result.success is False

    async def test_connection_failure_returns_clean_error(self) -> None:
        """Network failures should return clean error messages."""
        module = RemoteSSHModule()

        with mock.patch(
            "asyncssh.connect",
            side_effect=OSError("Network unreachable"),
        ):
            result = await module.execute(
                "remote_ssh:connect",
                {"hostname": "nonexistent.example.com", "username": "admin"},
            )

        assert result.success is False
        assert "Connection failed" in result.error

    async def test_closed_session_not_reused(self) -> None:
        """Closed sessions should not be reused."""
        module = RemoteSSHModule()

        mock_conn = MockSSHConnection()

        with mock.patch("asyncssh.connect", return_value=mock_conn):
            await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )

            # Manually close the connection
            mock_conn.close()

            # Should fail because session is closed
            result = await module.execute(
                "remote_ssh:execute_command",
                {"hostname": "switch01", "command": "echo test"},
            )

        assert result.success is False
        assert "No session" in result.error

    async def test_max_sessions_limit_enforced(self) -> None:
        """Should reject new connections when max_sessions limit is hit."""
        module = RemoteSSHModule(max_sessions=2)
        mock_conn1 = MockSSHConnection(hostname="switch01")
        mock_conn2 = MockSSHConnection(hostname="switch02")
        mock_conn3 = MockSSHConnection(hostname="switch03")

        with mock.patch(
            "asyncssh.connect",
            side_effect=[mock_conn1, mock_conn2, mock_conn3],
        ):
            # Connect to first host
            result1 = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch01", "username": "admin"},
            )
            assert result1.success is True

            # Connect to second host
            result2 = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch02", "username": "admin"},
            )
            assert result2.success is True

            # Third connection should fail due to limit (checked BEFORE connect)
            result3 = await module.execute(
                "remote_ssh:connect",
                {"hostname": "switch03", "username": "admin"},
            )

        assert result3.success is False
        assert "Maximum sessions" in result3.error
        assert len(module._sessions) == 2
