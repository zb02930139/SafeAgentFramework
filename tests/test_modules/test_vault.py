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

"""Tests for the vault module."""

import logging
from typing import Any

from safe_agent.modules.vault import VaultBackend, VaultModule


class MockVaultBackend:
    """Mock backend for testing VaultModule."""

    def __init__(self) -> None:
        """Initialize mock backend with storage for secrets."""
        self.secrets: dict[str, dict[str, Any]] = {}
        self.last_get_kwargs: dict[str, Any] = {}
        self.access_log: list[str] = []

    async def get_secret(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Mock get_secret that returns stored secret or raises error."""
        self.access_log.append(path)
        self.last_get_kwargs = kwargs

        if path not in self.secrets:
            raise RuntimeError(f"Secret not found: {path}")

        return self.secrets[path]


class TestVaultModule:
    """Tests for VaultModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "vault"
        assert len(descriptor.tools) == 1
        tool = descriptor.tools[0]
        assert tool.name == "vault:get_secret"
        assert tool.action == "vault:GetSecret"
        assert tool.resource_param == ["path"]
        assert tool.condition_keys == [
            "vault:SecretPath",
            "vault:SecretEngine",
        ]

    def test_tool_descriptor_has_correct_parameters(self) -> None:
        """Tool descriptor should have correct parameter schema."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        descriptor = module.describe()
        tool = descriptor.tools[0]

        assert tool.parameters["type"] == "object"
        assert "path" in tool.parameters["properties"]
        assert "version" in tool.parameters["properties"]
        assert tool.parameters["properties"]["path"]["type"] == "string"
        assert tool.parameters["properties"]["version"]["type"] == "integer"
        assert tool.parameters["required"] == ["path"]
        assert tool.parameters["additionalProperties"] is False

    async def test_resolve_conditions_derives_secret_path(self) -> None:
        """resolve_conditions should derive vault:SecretPath from params."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        conditions = await module.resolve_conditions(
            "vault:get_secret",
            {"path": "kv/database/credentials"},
        )

        assert conditions["vault:SecretPath"] == "kv/database/credentials"

    async def test_resolve_conditions_derives_secret_engine_from_prefix(
        self,
    ) -> None:
        """resolve_conditions should derive vault:SecretEngine from path prefix."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        # Test kv/ prefix
        conditions = await module.resolve_conditions(
            "vault:get_secret",
            {"path": "kv/database/credentials"},
        )
        assert conditions["vault:SecretEngine"] == "kv"

        # Test secret/ prefix
        conditions = await module.resolve_conditions(
            "vault:get_secret",
            {"path": "secret/myapp/config"},
        )
        assert conditions["vault:SecretEngine"] == "secret"

        # Test aws/ prefix
        conditions = await module.resolve_conditions(
            "vault:get_secret",
            {"path": "aws/creds/my-role"},
        )
        assert conditions["vault:SecretEngine"] == "aws"

    async def test_resolve_conditions_handles_path_without_slash(self) -> None:
        """resolve_conditions should handle paths without '/'."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        conditions = await module.resolve_conditions(
            "vault:get_secret",
            {"path": "simple-secret"},
        )

        # When no "/" is present, the whole path becomes the engine
        assert conditions == {
            "vault:SecretPath": "simple-secret",
            "vault:SecretEngine": "simple-secret",
        }

    async def test_resolve_conditions_returns_empty_for_missing_path(
        self,
    ) -> None:
        """resolve_conditions should return empty dict if path is missing."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        conditions = await module.resolve_conditions(
            "vault:get_secret",
            {},
        )

        assert conditions == {}

    async def test_execute_delegates_to_backend(self) -> None:
        """Execute should delegate to backend and return secret data."""
        backend = MockVaultBackend()
        backend.secrets["kv/database/credentials"] = {
            "username": "admin",
            "password": "secret123",
        }
        module = VaultModule(backend)

        result = await module.execute(
            "vault:get_secret",
            {"path": "kv/database/credentials"},
        )

        assert result.success is True
        assert result.data == {
            "username": "admin",
            "password": "secret123",
        }
        assert "kv/database/credentials" in backend.access_log

    async def test_execute_passes_version_to_backend(self) -> None:
        """Execute should pass version parameter to backend."""
        backend = MockVaultBackend()
        backend.secrets["kv/app/config"] = {"key": "value"}
        module = VaultModule(backend)

        result = await module.execute(
            "vault:get_secret",
            {"path": "kv/app/config", "version": 2},
        )

        assert result.success is True
        assert backend.last_get_kwargs.get("version") == 2

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """Execute should return error for unknown tool names."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        result = await module.execute(
            "vault:unknown_tool",
            {"path": "kv/test"},
        )

        assert result.success is False
        assert result.error == "Unknown tool: vault:unknown_tool"

    async def test_backend_error_propagates_as_tool_result_error(self) -> None:
        """Backend errors should be caught and returned as ToolResult errors."""
        backend = MockVaultBackend()
        # Don't add any secrets - will cause RuntimeError
        module = VaultModule(backend)

        result = await module.execute(
            "vault:get_secret",
            {"path": "kv/nonexistent"},
        )

        assert result.success is False
        assert result.error is not None
        assert "Secret not found" in result.error

    def test_backend_protocol_check(self) -> None:
        """MockVaultBackend should satisfy the VaultBackend protocol."""
        backend = MockVaultBackend()
        assert isinstance(backend, VaultBackend)

    async def test_secret_value_never_logged(self, caplog: Any) -> None:
        """Secret values should never appear in logs."""
        backend = MockVaultBackend()
        backend.secrets["kv/test"] = {"password": "super_secret_value"}
        module = VaultModule(backend)

        with caplog.at_level(logging.DEBUG):
            await module.execute("vault:get_secret", {"path": "kv/test"})

        assert "kv/test" in caplog.text  # Path should be logged
        assert "super_secret_value" not in caplog.text  # Value should NEVER appear

    async def test_get_secret_returns_error_for_missing_path(self) -> None:
        """_get_secret should return error when path parameter is missing."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        result = await module.execute("vault:get_secret", {})

        assert result.success is False
        assert result.error == "Missing required parameter: path"

    async def test_get_secret_returns_error_for_empty_path(self) -> None:
        """_get_secret should return error when path is empty or whitespace."""
        backend = MockVaultBackend()
        module = VaultModule(backend)

        # Test empty string
        result = await module.execute("vault:get_secret", {"path": ""})
        assert result.success is False
        assert result.error == "Secret path must not be empty"

        # Test whitespace-only string
        result = await module.execute("vault:get_secret", {"path": "   "})
        assert result.success is False
        assert result.error == "Secret path must not be empty"
