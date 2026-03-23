"""Tests for the controlled shell execution module."""

from __future__ import annotations

import sys
from pathlib import Path

from safe_agent.modules.shell import ShellModule


class TestShellModule:
    """Tests for ShellModule descriptors, condition resolution, and execution."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        module = ShellModule()

        descriptor = module.describe()

        assert descriptor.namespace == "shell"
        assert len(descriptor.tools) == 1
        tool = descriptor.tools[0]
        assert tool.name == "shell:execute"
        assert tool.action == "shell:Execute"
        assert tool.resource_param == ["command"]
        assert tool.condition_keys == [
            "shell:CommandName",
            "shell:WorkingDirectory",
        ]

    async def test_execute_runs_simple_command(self, tmp_path: Path) -> None:
        """A basic subprocess should execute successfully."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "print('hello')"],
            },
        )

        assert result.success is True
        assert result.data == {
            "stdout": "hello\n",
            "stderr": "",
            "return_code": 0,
        }

    async def test_execute_respects_working_directory(self, tmp_path: Path) -> None:
        """Commands should run in the configured working directory."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import pathlib; print(pathlib.Path.cwd())"],
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == str(tmp_path)

    async def test_timeout_kills_long_running_command(self, tmp_path: Path) -> None:
        """Commands exceeding the timeout should fail cleanly."""
        module = ShellModule(working_directory=tmp_path, default_timeout=5.0)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(1)"],
                "timeout": 0.05,
            },
        )

        assert result.success is False
        assert result.error == "Command timed out"

    async def test_output_truncation(self, tmp_path: Path) -> None:
        """Excessive output should be truncated to the configured limit."""
        module = ShellModule(working_directory=tmp_path, max_output_size=32)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "print('x' * 100)"],
            },
        )

        assert result.success is True
        assert result.data is not None
        assert len(result.data["stdout"].encode("utf-8")) == 32
        assert result.metadata["output_truncated"] is True
        assert result.metadata["stdout_truncated"] is True
        assert result.metadata["stderr_truncated"] is False
        assert result.metadata["max_output_size"] == 32

    async def test_resolve_conditions_returns_command_name_and_working_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """resolve_conditions() should derive command name and working directory."""
        module = ShellModule(working_directory=tmp_path)

        conditions = await module.resolve_conditions(
            "shell:execute",
            {"command": "echo hello"},
        )

        assert conditions == {
            "shell:CommandName": "echo",
            "shell:WorkingDirectory": str(tmp_path),
        }

    def test_resolve_conditions_uses_current_directory_when_unset(self) -> None:
        """When unset, the effective cwd should be reported as the current cwd."""
        module = ShellModule()

        conditions = __import__("asyncio").run(
            module.resolve_conditions("shell:execute", {"command": "echo hello"})
        )

        assert conditions["shell:CommandName"] == "echo"
        assert conditions["shell:WorkingDirectory"] == str(Path.cwd())

    async def test_resolve_conditions_handles_malformed_command(self) -> None:
        """Malformed shell-like quoting should not raise from condition resolution."""
        module = ShellModule()

        conditions = await module.resolve_conditions(
            "shell:execute",
            {"command": "echo 'unterminated"},
        )

        assert conditions["shell:CommandName"] == ""
        assert conditions["shell:WorkingDirectory"] == str(Path.cwd())

    async def test_non_zero_exit_is_still_successful_execution(
        self,
        tmp_path: Path,
    ) -> None:
        """A failing command should still return success=True with its exit code."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    ("import sys; print('oops', file=sys.stderr); raise SystemExit(7)"),
                ],
            },
        )

        assert result.success is True
        assert result.data == {
            "stdout": "",
            "stderr": "oops\n",
            "return_code": 7,
        }

    async def test_execute_passes_environment_overrides(self, tmp_path: Path) -> None:
        """Explicit env values should be visible to the subprocess."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import os; print(os.environ['SAFE_AGENT_TEST'])"],
                "env": {"SAFE_AGENT_TEST": "present"},
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"] == "present\n"

    async def test_empty_command_is_rejected(self, tmp_path: Path) -> None:
        """An empty command string should return a clean validation error."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute("shell:execute", {"command": "   "})

        assert result.success is False
        assert result.error == "Command must not be empty"

    async def test_negative_timeout_is_rejected_cleanly(self, tmp_path: Path) -> None:
        """Invalid timeout values should return a failed tool result."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {"command": "echo hello", "timeout": -1},
        )

        assert result.success is False
        assert result.error == "timeout must be >= 0"

    async def test_invalid_env_is_rejected_cleanly(self, tmp_path: Path) -> None:
        """Invalid env payloads should return a failed tool result."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {"command": "echo hello", "env": "not-a-dict"},
        )

        assert result.success is False
        assert result.error == "env must be an object of string key/value pairs"

    async def test_unknown_tool_returns_failed_result(self, tmp_path: Path) -> None:
        """Unknown tool names should return a failed tool result."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute("shell:unknown", {"command": "echo hello"})

        assert result.success is False
        assert result.error == "Unknown tool: shell:unknown"

    async def test_incremental_reading_huge_output(self, tmp_path: Path) -> None:
        """Huge output should be truncated via incremental reading without OOM."""
        module = ShellModule(working_directory=tmp_path, max_output_size=1000)

        # Generate output larger than max_output_size
        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    (
                        "import sys; print('x' * 10000); "
                        "print('y' * 10000, file=sys.stderr)"
                    ),
                ],
            },
        )

        assert result.success is True
        assert result.data is not None
        # Total output should be at most max_output_size
        total_bytes = len(result.data["stdout"].encode("utf-8")) + len(
            result.data["stderr"].encode("utf-8")
        )
        assert total_bytes <= 1000
        assert result.metadata["output_truncated"] is True
        # Process should be killed when output exceeded limit
        assert result.data["return_code"] == -9  # SIGKILL

    async def test_incremental_reading_huge_output_counts_bytes_not_chars(
        self, tmp_path: Path
    ) -> None:
        """Verify byte counting, not character counting."""
        limit = 100
        module = ShellModule(working_directory=tmp_path, max_output_size=limit)

        # Emojis are 4 bytes each in UTF-8
        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    "print('🎉' * 50)  # 200 bytes total",
                ],
            },
        )

        assert result.success is True
        assert result.data is not None
        # Should be truncated since emojis are multi-byte
        total_bytes = len(result.data["stdout"].encode("utf-8"))
        assert total_bytes <= limit
