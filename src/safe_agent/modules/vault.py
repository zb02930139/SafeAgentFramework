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

"""Vault module for secret management in SafeAgent.

This module provides secure secret access through a pluggable backend protocol.
The framework defines the interface; a plugin provides the concrete implementation
(HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class VaultBackend(Protocol):
    """Protocol for vault backend implementations.

    Implementations provide concrete integrations with secret management
    systems such as HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, etc.

    Methods:
        get_secret: Retrieve a secret by its path.
    """

    async def get_secret(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Retrieve a secret by its path.

        Args:
            path: The secret path (e.g., "kv/database/credentials").
            **kwargs: Backend-specific options (e.g., version).

        Returns:
            Dict containing the secret data.

        Raises:
            Exception: If the secret cannot be retrieved.
        """
        ...


class VaultModule(BaseModule):
    """Vault module using the pluggable adapter pattern.

    This module exposes secret access tools to the SafeAgent framework while
    delegating actual secret operations to an injected backend provider.

    Security Note:
        Secret access is logged at INFO level with the path ONLY.
        Secret values are NEVER logged. However, secret values WILL be
        visible in ToolResult.data and thus may be exposed to the LLM.

    Attributes:
        _backend: The vault backend implementation (private).
    """

    def __init__(self, backend: VaultBackend) -> None:
        """Initialize the vault module with a backend provider.

        Args:
            backend: An implementation of VaultBackend that handles
                actual secret operations for a specific platform.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the vault module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="vault",
            description="Secure secret access through a vault backend.",
            tools=[
                ToolDescriptor(
                    name="vault:get_secret",
                    description="Retrieve a secret by path.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The secret path.",
                            },
                            "version": {
                                "type": "integer",
                                "description": "Optional secret version number.",
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    action="vault:GetSecret",
                    resource_param=["path"],
                    condition_keys=[
                        "vault:SecretPath",
                        "vault:SecretEngine",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve vault condition values from the path parameter.

        Derives condition keys for policy evaluation:
            - vault:SecretPath: The full secret path.
            - vault:SecretEngine: The secret engine prefix (e.g., "kv/" -> "kv").

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        path = params.get("path")
        if path is None:
            return {}

        path_str = str(path)
        conditions: dict[str, Any] = {
            "vault:SecretPath": path_str,
        }

        # Extract secret engine from path prefix (e.g., "kv/database/creds" -> "kv")
        # Look for the first path segment before a "/"
        if "/" in path_str:
            engine = path_str.split("/", 1)[0]
            conditions["vault:SecretEngine"] = engine
        else:
            # Path without "/" - use the whole path as engine
            conditions["vault:SecretEngine"] = path_str

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a vault tool invocation by delegating to the backend.

        Security Note:
            Secret access is logged at INFO level with the path ONLY.
            Secret values are NEVER logged.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("vault:")
            if normalized_tool == "get_secret":
                return await self._get_secret(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _get_secret(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate secret retrieval to the backend.

        Security Note:
            Logs the secret path at INFO level. NEVER logs the secret value.
        """
        path = params.get("path")
        if path is None:
            return ToolResult(success=False, error="Missing required parameter: path")

        path = str(path)
        if not path or not path.strip():
            return ToolResult(success=False, error="Secret path must not be empty")

        # Log secret access at INFO level - path ONLY, NEVER the value
        logger.info("Secret access: path=%s", path)

        kwargs: dict[str, Any] = {}
        if "version" in params:
            kwargs["version"] = params["version"]

        secret = await self._backend.get_secret(path, **kwargs)
        return ToolResult(success=True, data=secret)
