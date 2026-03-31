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

"""Tests for the pluggable messaging module."""

from typing import Any

from safe_agent.modules.communication.messaging import MessagingBackend, MessagingModule


class MockMessagingBackend:
    """Mock backend for testing MessagingModule."""

    def __init__(self) -> None:
        """Initialize mock backend with storage for sent messages."""
        self.sent_messages: list[dict[str, Any]] = []
        self.stored_messages: dict[str, list[dict[str, Any]]] = {}
        # Track kwargs passed to read_messages
        self.last_read_kwargs: dict[str, Any] = {}

    async def send_message(
        self,
        channel: str,
        text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock send_message that records the call."""
        message = {
            "id": f"msg-{len(self.sent_messages) + 1}",
            "channel": channel,
            "text": text,
            **kwargs,
        }
        self.sent_messages.append(message)
        return message

    async def read_messages(
        self,
        channel: str,
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Mock read_messages that returns stored messages for the channel."""
        self.last_read_kwargs = kwargs  # Track what kwargs were passed
        messages = self.stored_messages.get(channel, [])
        return messages[:limit]


class TestMessagingModule:
    """Tests for MessagingModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "messaging"
        assert len(descriptor.tools) == 2
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "messaging:send_message",
            "messaging:read_messages",
        }

    def test_describe_tools_have_correct_actions_and_resource_params(
        self,
    ) -> None:
        """Tools should have correct action names and resource params."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        descriptor = module.describe()

        send_tool = next(
            tool for tool in descriptor.tools if tool.name == "messaging:send_message"
        )
        read_tool = next(
            tool for tool in descriptor.tools if tool.name == "messaging:read_messages"
        )

        assert send_tool.action == "messaging:SendMessage"
        assert send_tool.resource_param == ["channel"]
        assert send_tool.condition_keys == ["messaging:Channel", "messaging:Recipient"]

        assert read_tool.action == "messaging:ReadMessages"
        assert read_tool.resource_param == ["channel"]
        # read_messages should NOT have messaging:Recipient (semantically wrong)
        assert read_tool.condition_keys == ["messaging:Channel"]

    async def test_send_message_delegates_to_backend(self) -> None:
        """send_message should delegate to the backend and return success."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        result = await module.execute(
            "messaging:send_message",
            {"channel": "#general", "text": "Hello, world!"},
        )

        assert result.success is True
        assert result.data == {
            "id": "msg-1",
            "channel": "#general",
            "text": "Hello, world!",
        }
        assert len(backend.sent_messages) == 1

    async def test_send_message_with_thread_id(self) -> None:
        """send_message should pass through optional thread_id."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        result = await module.execute(
            "messaging:send_message",
            {
                "channel": "#general",
                "text": "Replying to thread",
                "thread_id": "thread-123",
            },
        )

        assert result.success is True
        assert result.data == {
            "id": "msg-1",
            "channel": "#general",
            "text": "Replying to thread",
            "thread_id": "thread-123",
        }

    async def test_read_messages_delegates_to_backend(self) -> None:
        """read_messages should delegate to backend and return messages."""
        backend = MockMessagingBackend()
        backend.stored_messages["#general"] = [
            {"id": "m1", "text": "First"},
            {"id": "m2", "text": "Second"},
            {"id": "m3", "text": "Third"},
        ]
        module = MessagingModule(backend)

        result = await module.execute(
            "messaging:read_messages",
            {"channel": "#general", "limit": 2},
        )

        assert result.success is True
        assert result.data == {
            "messages": [
                {"id": "m1", "text": "First"},
                {"id": "m2", "text": "Second"},
            ]
        }

    async def test_read_messages_with_since_filter(self) -> None:
        """read_messages should pass through optional since filter."""
        backend = MockMessagingBackend()
        backend.stored_messages["#general"] = [
            {"id": "m1", "text": "Message 1"},
        ]
        module = MessagingModule(backend)

        result = await module.execute(
            "messaging:read_messages",
            {
                "channel": "#general",
                "limit": 10,
                "since": "2026-03-26T00:00:00Z",
            },
        )

        assert result.success is True
        assert result.data == {"messages": [{"id": "m1", "text": "Message 1"}]}
        # Verify the since kwarg was actually forwarded to the backend
        assert backend.last_read_kwargs.get("since") == "2026-03-26T00:00:00Z"

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """Execute should return error for unknown tool names."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        result = await module.execute(
            "messaging:unknown_tool",
            {"channel": "#general"},
        )

        assert result.success is False
        assert result.error == "Unknown tool: messaging:unknown_tool"

    async def test_backend_error_propagates_as_tool_result_error(self) -> None:
        """Backend errors should be caught and returned as ToolResult errors."""

        class FailingBackend(MockMessagingBackend):
            async def send_message(
                self, channel: str, text: str, **kwargs: Any
            ) -> dict[str, Any]:
                raise RuntimeError("Network error")

        backend = FailingBackend()
        module = MessagingModule(backend)

        result = await module.execute(
            "messaging:send_message",
            {"channel": "#general", "text": "test"},
        )

        assert result.success is False
        assert result.error == "Network error"

    async def test_resolve_conditions_derives_channel(self) -> None:
        """resolve_conditions should derive messaging:Channel from params."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"channel": "#general", "text": "hello"},
        )

        assert conditions == {
            "messaging:Channel": "#general",
        }

    async def test_resolve_conditions_infers_recipient_for_dm(self) -> None:
        """resolve_conditions should infer messaging:Recipient for DM channels."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        # Test @username style DM
        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"channel": "@alice", "text": "hello"},
        )

        assert conditions == {
            "messaging:Channel": "@alice",
            "messaging:Recipient": "alice",
        }

        # Test D-style DM (Slack convention: D followed by digits)
        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"channel": "D12345", "text": "hello"},
        )

        assert conditions == {
            "messaging:Channel": "D12345",
            "messaging:Recipient": "D12345",
        }

    async def test_resolve_conditions_no_recipient_for_read_messages(
        self,
    ) -> None:
        """read_messages should not get messaging:Recipient condition key."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        conditions = await module.resolve_conditions(
            "messaging:read_messages",
            {"channel": "D12345", "limit": 10},
        )

        assert conditions == {
            "messaging:Channel": "D12345",
        }

    async def test_resolve_conditions_no_false_positive_recipient(self) -> None:
        """resolve_conditions should not infer Recipient for #dev, #design, etc."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        # Test that #dev does NOT get Recipient (D prefix on its own)
        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"channel": "#dev", "text": "hello"},
        )

        assert conditions == {
            "messaging:Channel": "#dev",
        }

        # Test that #design does NOT get Recipient
        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"channel": "#design", "text": "hello"},
        )

        assert conditions == {
            "messaging:Channel": "#design",
        }

        # Test that deployment-alerts does NOT get Recipient
        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"channel": "deployment-alerts", "text": "hello"},
        )

        assert conditions == {
            "messaging:Channel": "deployment-alerts",
        }

    async def test_resolve_conditions_returns_empty_for_missing_channel(
        self,
    ) -> None:
        """resolve_conditions should return empty dict if channel is missing."""
        backend = MockMessagingBackend()
        module = MessagingModule(backend)

        conditions = await module.resolve_conditions(
            "messaging:send_message",
            {"text": "hello"},
        )

        assert conditions == {}

    def test_backend_protocol_check(self) -> None:
        """MockMessagingBackend should satisfy the MessagingBackend protocol."""
        backend = MockMessagingBackend()
        assert isinstance(backend, MessagingBackend)
