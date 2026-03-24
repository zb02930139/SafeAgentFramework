"""Tests for the controlled shell execution module."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from safe_agent.modules.shell import ShellModule


class TestShellModuleTimeoutCap:
    """Tests for max_timeout cap behavior (Issue #62)."""

    def test_max_timeout_zero_raises_valueerror(self) -> None:
        """max_timeout=0 should raise ValueError at construction."""
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            ShellModule(max_timeout=0)

    def test_max_timeout_negative_raises_valueerror(self) -> None:
        """max_timeout < 0 should raise ValueError at construction."""
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            ShellModule(max_timeout=-1)

    async def test_timeout_clamped_to_max_timeout(self, tmp_path: Path) -> None:
        """Per-request timeout exceeding max_timeout should be clamped."""
        module = ShellModule(working_directory=tmp_path, max_timeout=1.0)

        # Request 999999 seconds (~11.5 days), should be clamped to 1.0
        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(0.05); print('ok')"],
                "timeout": 999999,  # Would be ~11.5 days without cap
            },
        )

        # Should succeed because timeout is clamped to 1.0 seconds
        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "ok"

    async def test_timeout_zero_is_rejected(self, tmp_path: Path) -> None:
        """timeout=0 should be rejected (immediate timeout is surprising)."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {"command": "echo hello", "timeout": 0},
        )

        assert result.success is False
        assert result.error == "timeout must be > 0 (immediate timeout is not allowed)"

    async def test_max_timeout_configurable(self, tmp_path: Path) -> None:
        """max_timeout should be configurable at construction time."""
        module = ShellModule(working_directory=tmp_path, max_timeout=0.1)

        # Even with default_timeout=30, max_timeout=0.1 should clamp
        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(1)"],
            },
        )

        # Should timeout because clamped to 0.1 seconds
        assert result.success is False
        assert result.error == "Command timed out"

    async def test_timeout_below_max_is_not_clamped(self, tmp_path: Path) -> None:
        """Per-request timeout below max_timeout should not be modified."""
        module = ShellModule(working_directory=tmp_path, max_timeout=300.0)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "print('ok')"],
                "timeout": 5,  # Well below max_timeout
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "ok"

    async def test_default_timeout_respects_max_cap(self, tmp_path: Path) -> None:
        """default_timeout > max_timeout should be clamped when used."""
        module = ShellModule(
            working_directory=tmp_path,
            default_timeout=999999,  # Would be ~11.5 days
            max_timeout=0.1,
        )

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(1)"],
            },
        )

        # Should timeout because default is clamped to max_timeout
        assert result.success is False
        assert result.error == "Command timed out"


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


class TestShellModuleEnvSecurity:
    """Tests for secure environment variable handling (Issue #25)."""

    async def test_default_env_is_minimal(self, tmp_path: Path) -> None:
        """By default, no host env vars should be inherited except safe PATH."""
        module = ShellModule(working_directory=tmp_path)

        # Set a sensitive host env var for testing
        with mock.patch.dict(os.environ, {"SECRET_API_KEY": "secret-value-123"}):
            result = await module.execute(
                "shell:execute",
                {
                    "command": sys.executable,
                    "args": [
                        "-c",
                        "import os; print(os.environ.get('SECRET_API_KEY', 'NONE'))",
                    ],
                },
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "NONE"

    async def test_default_path_is_safe(self, tmp_path: Path) -> None:
        """Default PATH should be limited to known-safe directories."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import os; print(os.environ['PATH'])"],
            },
        )

        assert result.success is True
        assert result.data is not None
        # Should be safe path, not host PATH
        assert result.data["stdout"].strip() == "/usr/local/bin:/usr/bin:/bin"

    async def test_allowed_env_vars_pass_through(self, tmp_path: Path) -> None:
        """Whitelisted host env vars should be inherited."""
        module = ShellModule(
            working_directory=tmp_path,
            allowed_env_vars=["SAFE_AGENT_TEST_VAR"],
        )

        with mock.patch.dict(os.environ, {"SAFE_AGENT_TEST_VAR": "passed-through"}):
            result = await module.execute(
                "shell:execute",
                {
                    "command": sys.executable,
                    "args": [
                        "-c",
                        "import os; print(os.environ['SAFE_AGENT_TEST_VAR'])",
                    ],
                },
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "passed-through"

    async def test_non_whitelisted_env_vars_blocked(self, tmp_path: Path) -> None:
        """Non-whitelisted host env vars should not be inherited."""
        module = ShellModule(
            working_directory=tmp_path,
            allowed_env_vars=["SAFE_AGENT_TEST_VAR"],
        )

        with mock.patch.dict(
            os.environ,
            {"SAFE_AGENT_TEST_VAR": "allowed", "SECRET_KEY": "blocked"},
        ):
            result = await module.execute(
                "shell:execute",
                {
                    "command": sys.executable,
                    "args": [
                        "-c",
                        (
                            "import os; "
                            "print(os.environ.get('SAFE_AGENT_TEST_VAR', 'NONE'), "
                            "os.environ.get('SECRET_KEY', 'NONE'))"
                        ),
                    ],
                },
            )

        assert result.success is True
        assert result.data is not None
        assert "allowed" in result.data["stdout"]
        assert "blocked" not in result.data["stdout"]
        assert result.data["stdout"].count("NONE") == 1

    async def test_path_override_blocked(self, tmp_path: Path) -> None:
        """LLM cannot override PATH via env parameter."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import os; print(os.environ['PATH'])"],
                "env": {"PATH": "/malicious/path:/usr/bin"},
            },
        )

        assert result.success is True
        assert result.data is not None
        # Should still be safe path, not malicious one
        assert result.data["stdout"].strip() == "/usr/local/bin:/usr/bin:/bin"
        assert "/malicious" not in result.data["stdout"]

    async def test_ld_preload_override_blocked(self, tmp_path: Path) -> None:
        """LLM cannot inject LD_PRELOAD via env parameter."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    "import os; print(os.environ.get('LD_PRELOAD', 'NONE'))",
                ],
                "env": {"LD_PRELOAD": "/malicious/lib.so"},
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "NONE"

    async def test_ld_library_path_override_blocked(self, tmp_path: Path) -> None:
        """LLM cannot inject LD_LIBRARY_PATH via env parameter."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    "import os; print(os.environ.get('LD_LIBRARY_PATH', 'NONE'))",
                ],
                "env": {"LD_LIBRARY_PATH": "/malicious/lib"},
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "NONE"

    async def test_pythonpath_override_blocked(self, tmp_path: Path) -> None:
        """LLM cannot inject PYTHONPATH via env parameter."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    "import os; print(os.environ.get('PYTHONPATH', 'NONE'))",
                ],
                "env": {"PYTHONPATH": "/malicious/python"},
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "NONE"

    async def test_allowed_env_var_but_blocked_override_still_blocked(
        self, tmp_path: Path
    ) -> None:
        """Even if PATH is in allowed_env_vars, LLM cannot override it."""
        module = ShellModule(
            working_directory=tmp_path,
            allowed_env_vars=["PATH"],  # Explicitly allow PATH passthrough
        )

        # Host has a custom PATH
        with mock.patch.dict(os.environ, {"PATH": "/custom/host/path"}):
            result = await module.execute(
                "shell:execute",
                {
                    "command": sys.executable,
                    "args": ["-c", "import os; print(os.environ['PATH'])"],
                    "env": {"PATH": "/malicious/override"},
                },
            )

        assert result.success is True
        assert result.data is not None
        # Should have host PATH, not malicious override
        assert result.data["stdout"].strip() == "/custom/host/path"

    async def test_safe_env_override_works(self, tmp_path: Path) -> None:
        """Non-blocked env overrides from LLM should work."""
        module = ShellModule(working_directory=tmp_path)

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": ["-c", "import os; print(os.environ['MY_APP_CONFIG'])"],
                "env": {"MY_APP_CONFIG": "safe-value"},
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["stdout"].strip() == "safe-value"

    async def test_all_blocked_vars_simultaneously(self, tmp_path: Path) -> None:
        """Test all blocked env vars are rejected together."""
        module = ShellModule(working_directory=tmp_path)

        blocked_vars = {
            # Binary/library injection
            "PATH": "/evil/bin",
            "LD_PRELOAD": "/evil/lib.so",
            "LD_LIBRARY_PATH": "/evil/lib",
            "LD_AUDIT": "/evil/audit.so",
            "LD_DEBUG": "all",
            # macOS library injection
            "DYLD_INSERT_LIBRARIES": "/evil/dylib.so",
            "DYLD_LIBRARY_PATH": "/evil/dylib",
            "DYLD_FRAMEWORK_PATH": "/evil/framework",
            # Shell code injection
            "BASH_ENV": "/evil/bash.sh",
            "ENV": "/evil/env.sh",
            "ENVIRONMENT": "/evil/environment",
            # Python code injection
            "PYTHONPATH": "/evil/python",
            "PYTHONHOME": "/evil/pyhome",
            "PYTHONEXECUTABLE": "/evil/python",
            "PYTHONSTARTUP": "/evil/startup.py",
            # Node.js code injection
            "NODE_OPTIONS": "--require=/evil/node.js",
            # Perl code injection
            "PERL5OPT": "-I/evil/perl",
            "PERL5LIB": "/evil/perl/lib",
            # Ruby code injection
            "RUBYOPT": "-I/evil/ruby",
            "RUBYLIB": "/evil/ruby/lib",
            # Java options injection
            "JAVA_TOOL_OPTIONS": "-Devil=true",
            "_JAVA_OPTIONS": "-Devil=true",
            "JAVA_OPTS": "-Devil=true",
        }

        result = await module.execute(
            "shell:execute",
            {
                "command": sys.executable,
                "args": [
                    "-c",
                    (
                        "import os; "
                        "import json; "
                        "data = {"
                        "'PATH': os.environ.get('PATH', 'NONE'), "
                        "'LD_PRELOAD': os.environ.get('LD_PRELOAD', 'NONE'), "
                        "'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', 'NONE'), "
                        "'LD_AUDIT': os.environ.get('LD_AUDIT', 'NONE'), "
                        "'LD_DEBUG': os.environ.get('LD_DEBUG', 'NONE'), "
                        "'DYLD_INSERT_LIBRARIES': os.environ.get("
                        "'DYLD_INSERT_LIBRARIES', 'NONE'), "
                        "'DYLD_LIBRARY_PATH': os.environ.get("
                        "'DYLD_LIBRARY_PATH', 'NONE'), "
                        "'DYLD_FRAMEWORK_PATH': os.environ.get("
                        "'DYLD_FRAMEWORK_PATH', 'NONE'), "
                        "'BASH_ENV': os.environ.get('BASH_ENV', 'NONE'), "
                        "'ENV': os.environ.get('ENV', 'NONE'), "
                        "'ENVIRONMENT': os.environ.get('ENVIRONMENT', 'NONE'), "
                        "'PYTHONPATH': os.environ.get('PYTHONPATH', 'NONE'), "
                        "'PYTHONHOME': os.environ.get('PYTHONHOME', 'NONE'), "
                        "'PYTHONEXECUTABLE': os.environ.get("
                        "'PYTHONEXECUTABLE', 'NONE'), "
                        "'PYTHONSTARTUP': os.environ.get('PYTHONSTARTUP', 'NONE'), "
                        "'NODE_OPTIONS': os.environ.get('NODE_OPTIONS', 'NONE'), "
                        "'PERL5OPT': os.environ.get('PERL5OPT', 'NONE'), "
                        "'PERL5LIB': os.environ.get('PERL5LIB', 'NONE'), "
                        "'RUBYOPT': os.environ.get('RUBYOPT', 'NONE'), "
                        "'RUBYLIB': os.environ.get('RUBYLIB', 'NONE'), "
                        "'JAVA_TOOL_OPTIONS': os.environ.get("
                        "'JAVA_TOOL_OPTIONS', 'NONE'), "
                        "'_JAVA_OPTIONS': os.environ.get('_JAVA_OPTIONS', 'NONE'), "
                        "'JAVA_OPTS': os.environ.get('JAVA_OPTS', 'NONE')}; "
                        "print(json.dumps(data))"
                    ),
                ],
                "env": blocked_vars,
            },
        )

        assert result.success is True
        assert result.data is not None
        import json

        values = json.loads(result.data["stdout"].strip())
        # PATH should be default safe path, not malicious override
        assert values["PATH"] == "/usr/local/bin:/usr/bin:/bin"
        # All other blocked vars should be NONE
        assert values["LD_PRELOAD"] == "NONE"
        assert values["LD_LIBRARY_PATH"] == "NONE"
        assert values["LD_AUDIT"] == "NONE"
        assert values["LD_DEBUG"] == "NONE"
        # macOS library injection
        assert values["DYLD_INSERT_LIBRARIES"] == "NONE"
        assert values["DYLD_LIBRARY_PATH"] == "NONE"
        assert values["DYLD_FRAMEWORK_PATH"] == "NONE"
        # Shell code injection
        assert values["BASH_ENV"] == "NONE"
        assert values["ENV"] == "NONE"
        assert values["ENVIRONMENT"] == "NONE"
        # Python code injection
        assert values["PYTHONPATH"] == "NONE"
        assert values["PYTHONHOME"] == "NONE"
        assert values["PYTHONEXECUTABLE"] == "NONE"
        assert values["PYTHONSTARTUP"] == "NONE"
        # Other interpreter injection
        assert values["NODE_OPTIONS"] == "NONE"
        assert values["PERL5OPT"] == "NONE"
        assert values["PERL5LIB"] == "NONE"
        assert values["RUBYOPT"] == "NONE"
        assert values["RUBYLIB"] == "NONE"
        assert values["JAVA_TOOL_OPTIONS"] == "NONE"
        assert values["_JAVA_OPTIONS"] == "NONE"
        assert values["JAVA_OPTS"] == "NONE"

    async def test_blocked_override_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Blocked override attempts should emit a warning for security auditing."""
        import logging

        module = ShellModule(working_directory=tmp_path)

        with caplog.at_level(logging.WARNING):
            result = await module.execute(
                "shell:execute",
                {
                    "command": sys.executable,
                    "args": ["-c", "print('ok')"],
                    "env": {"LD_PRELOAD": "/evil/lib.so"},
                },
            )

        assert result.success is True
        # Should have logged a warning about the blocked override
        assert any(
            "Blocked dangerous env var override attempt: LD_PRELOAD" in record.message
            for record in caplog.records
        )
