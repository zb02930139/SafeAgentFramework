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

"""Pluggable messaging module for SafeAgent.

This module provides a messaging interface that delegates to an injected backend
provider. The framework defines the interface; a plugin provides the concrete
implementation (Slack, Microsoft Teams, Discord, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


@runtime_checkable
class MessagingBackend(Protocol):
    """Protocol for messaging backend implementations.

    Implementations provide concrete integrations with messaging platforms
    such as Slack, Microsoft Teams, Discord, etc.

    Methods:
        send_message: Send a message to a channel or user.
        read_messages: Read recent messages from a channel.
    """

    async def send_message(
        self,
        channel: str,
        text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a message to a channel or user.

        Args:
            channel: Target channel or user identifier.
            text: Message content to send.
            **kwargs: Backend-specific options (e.g., thread_id, reply_to).

        Returns:
            Dict containing message metadata (id, timestamp, etc.).
        """
        ...

    async def read_messages(
        self,
        channel: str,
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Read recent messages from a channel.

        Args:
            channel: Source channel identifier.
            limit: Maximum number of messages to return.
            **kwargs: Backend-specific filters (e.g., since timestamp).

        Returns:
            List of message dicts with content, sender, timestamp, etc.
        """
        ...


class MessagingModule(BaseModule):
    """Messaging module using the pluggable adapter pattern.

    This module exposes messaging tools to the SafeAgent framework while
    delegating actual message operations to an injected backend provider.

    Attributes:
        backend: The messaging backend implementation.
    """

    def __init__(self, backend: MessagingBackend) -> None:
        """Initialize the messaging module with a backend provider.

        Args:
            backend: An implementation of MessagingBackend that handles
                actual message operations for a specific platform.
        """
        self.backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the messaging module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="messaging",
            description="Send and receive messages on team communication platforms.",
            tools=[
                ToolDescriptor(
                    name="messaging:send_message",
                    description="Send a message to a channel or user.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string"},
                            "text": {"type": "string"},
                            "thread_id": {"type": "string"},
                        },
                        "required": ["channel", "text"],
                        "additionalProperties": False,
                    },
                    action="messaging:SendMessage",
                    resource_param=["channel"],
                    condition_keys=[
                        "messaging:Channel",
                        "messaging:Recipient",
                    ],
                ),
                ToolDescriptor(
                    name="messaging:read_messages",
                    description="Read recent messages from a channel.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string"},
                            "limit": {"type": "integer"},
                            "since": {"type": "string"},
                        },
                        "required": ["channel", "limit"],
                        "additionalProperties": False,
                    },
                    action="messaging:ReadMessages",
                    resource_param=["channel"],
                    condition_keys=[
                        "messaging:Channel",
                        "messaging:Recipient",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve messaging condition values from the channel parameter.

        Derives condition keys for policy evaluation:
            - messaging:Channel: The target channel identifier.
            - messaging:Recipient: Inferred recipient for DM channels.

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        channel = params.get("channel")
        if channel is None:
            return {}

        channel_str = str(channel)
        conditions: dict[str, Any] = {
            "messaging:Channel": channel_str,
        }

        # Infer recipient for DM-style channels (e.g., "@username", "D12345")
        # This is a heuristic; backends may override with more precise logic.
        if channel_str.startswith("@") or channel_str.upper().startswith("D"):
            conditions["messaging:Recipient"] = channel_str.lstrip("@")

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a messaging tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("messaging:")
            if normalized_tool == "send_message":
                return await self._send_message(params)
            if normalized_tool == "read_messages":
                return await self._read_messages(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _send_message(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate message sending to the backend."""
        channel = str(params["channel"])
        text = str(params["text"])
        kwargs: dict[str, Any] = {}
        if "thread_id" in params:
            kwargs["thread_id"] = params["thread_id"]

        result = await self.backend.send_message(channel, text, **kwargs)
        return ToolResult(success=True, data=result)

    async def _read_messages(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate message reading to the backend."""
        channel = str(params["channel"])
        limit = int(params["limit"])
        kwargs: dict[str, Any] = {}
        if "since" in params:
            kwargs["since"] = params["since"]

        messages = await self.backend.read_messages(channel, limit, **kwargs)
        return ToolResult(success=True, data={"messages": messages})
