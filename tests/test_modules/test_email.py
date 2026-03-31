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


"""Tests for the email interface module using pluggable adapter pattern."""

from __future__ import annotations

from typing import Any

from safe_agent.modules.base import ModuleDescriptor
from safe_agent.modules.communication.email import EmailBackend, EmailModule


class MockEmailBackend:
    """Mock implementation of EmailBackend for testing."""

    def __init__(self) -> None:
        """Initialize mock backend with tracking state."""
        self.sent_messages: list[dict[str, Any]] = []
        self.inbox_messages: list[dict[str, Any]] = [
            {
                "message_id": "msg-001",
                "sender": "alice@example.com",
                "subject": "Test Email",
                "date": "2025-01-15T10:30:00Z",
            },
            {
                "message_id": "msg-002",
                "sender": "bob@example.com",
                "subject": "Another Test",
                "date": "2025-01-16T14:20:00Z",
            },
        ]
        self.parsed_messages: dict[str, dict[str, Any]] = {
            "msg-001": {
                "message_id": "msg-001",
                "body": "Hello, this is a test email.",
                "headers": {
                    "From": "alice@example.com",
                    "Subject": "Test Email",
                },
            },
        }

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock send: record the message and return a result."""
        message = {
            "to": to,
            "subject": subject,
            "body": body,
            **kwargs,
        }
        self.sent_messages.append(message)
        return {
            "message_id": f"sent-{len(self.sent_messages)}",
            "status": "delivered",
        }

    async def read_inbox(
        self,
        folder: str,
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Mock read_inbox: return stored messages up to limit."""
        return self.inbox_messages[:limit]

    async def parse(
        self,
        message_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock parse: return stored parsed message or empty dict."""
        return self.parsed_messages.get(message_id, {})


class FailingEmailBackend:
    """Backend that raises exceptions for testing error propagation."""

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Raise an error to simulate send failure."""
        raise RuntimeError("SendGrid API error: 401 Unauthorized")

    async def read_inbox(
        self,
        folder: str,
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Raise an error to simulate read failure."""
        raise RuntimeError("IMAP connection failed")

    async def parse(
        self,
        message_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Raise an error to simulate parse failure."""
        raise RuntimeError("Message not found")


class TestEmailModuleDescriptor:
    """Tests for EmailModule.describe() behavior."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return expected module metadata."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()

        assert isinstance(descriptor, ModuleDescriptor)
        assert descriptor.namespace == "email"
        assert "email" in descriptor.description.lower()

    def test_describe_returns_three_tools(self) -> None:
        """describe() should return exactly three tools."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()

        assert len(descriptor.tools) == 3

    def test_describe_tool_names(self) -> None:
        """describe() should return tools with correct names."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        tool_names = {tool.name for tool in descriptor.tools}

        assert tool_names == {
            "email:send_email",
            "email:read_inbox",
            "email:parse_email",
        }

    def test_describe_tool_actions(self) -> None:
        """describe() should return tools with correct action identifiers."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        tool_actions = {tool.action for tool in descriptor.tools}

        assert tool_actions == {
            "email:SendEmail",
            "email:ReadInbox",
            "email:ParseEmail",
        }

    def test_describe_send_email_parameters(self) -> None:
        """send_email tool should have correct parameter schema."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        send_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:send_email"
        )

        assert send_tool.parameters["type"] == "object"
        props = send_tool.parameters["properties"]
        assert "to" in props
        assert "subject" in props
        assert "body" in props
        assert "cc" in props
        assert "attachments" in props
        assert send_tool.parameters["required"] == ["to", "subject", "body"]
        assert send_tool.parameters["additionalProperties"] is False

    def test_describe_send_email_resource_param(self) -> None:
        """send_email tool should declare 'to' as resource param."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        send_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:send_email"
        )

        assert send_tool.resource_param == ["to"]

    def test_describe_send_email_condition_keys(self) -> None:
        """send_email tool should declare correct condition keys."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        send_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:send_email"
        )

        assert set(send_tool.condition_keys) == {
            "email:Recipient",
            "email:Sender",
            "email:Subject",
        }

    def test_describe_read_inbox_parameters(self) -> None:
        """read_inbox tool should have correct parameter schema."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        read_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:read_inbox"
        )

        assert read_tool.parameters["type"] == "object"
        props = read_tool.parameters["properties"]
        assert "folder" in props
        assert "limit" in props
        assert "filter" in props
        assert read_tool.parameters["required"] == ["limit"]
        assert read_tool.parameters["additionalProperties"] is False

    def test_describe_read_inbox_defaults(self) -> None:
        """read_inbox tool should have default for folder."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        read_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:read_inbox"
        )

        assert read_tool.parameters["properties"]["folder"]["default"] == "inbox"

    def test_describe_read_inbox_condition_keys(self) -> None:
        """read_inbox tool should declare email:Folder condition key."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        read_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:read_inbox"
        )

        assert read_tool.condition_keys == ["email:Folder"]

    def test_describe_parse_email_parameters(self) -> None:
        """parse_email tool should have correct parameter schema."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        descriptor = module.describe()
        parse_tool = next(
            tool for tool in descriptor.tools if tool.name == "email:parse_email"
        )

        assert parse_tool.parameters["type"] == "object"
        props = parse_tool.parameters["properties"]
        assert "message_id" in props
        assert "extract_fields" in props
        assert parse_tool.parameters["required"] == ["message_id"]
        assert parse_tool.parameters["additionalProperties"] is False


class TestEmailModuleResolveConditions:
    """Tests for EmailModule.resolve_conditions() behavior."""

    async def test_resolve_conditions_send_email(self) -> None:
        """resolve_conditions for send_email should derive Recipient and Subject."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        conditions = await module.resolve_conditions(
            "email:send_email",
            {"to": "user@example.com", "subject": "Hello"},
        )

        assert conditions == {
            "email:Recipient": "user@example.com",
            "email:Subject": "Hello",
        }

    async def test_resolve_conditions_send_email_missing_params(self) -> None:
        """resolve_conditions should handle missing params gracefully."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        conditions = await module.resolve_conditions("email:send_email", {})

        assert conditions == {
            "email:Recipient": "",
            "email:Subject": "",
        }

    async def test_resolve_conditions_read_inbox(self) -> None:
        """resolve_conditions for read_inbox should derive Folder."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        conditions = await module.resolve_conditions(
            "email:read_inbox",
            {"folder": "archive", "limit": 10},
        )

        assert conditions == {"email:Folder": "archive"}

    async def test_resolve_conditions_read_inbox_default_folder(self) -> None:
        """resolve_conditions for read_inbox should use default folder."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        conditions = await module.resolve_conditions(
            "email:read_inbox",
            {"limit": 10},
        )

        assert conditions == {"email:Folder": "inbox"}

    async def test_resolve_conditions_parse_email(self) -> None:
        """resolve_conditions for parse_email should return empty dict."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        conditions = await module.resolve_conditions(
            "email:parse_email",
            {"message_id": "msg-001"},
        )

        assert conditions == {}


class TestEmailModuleExecute:
    """Tests for EmailModule.execute() behavior."""

    async def test_execute_send_email_success(self) -> None:
        """Execute send_email should delegate to backend and return success."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:send_email",
            {
                "to": "user@example.com",
                "subject": "Test Subject",
                "body": "Test body content",
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["message_id"] == "sent-1"
        assert result.data["status"] == "delivered"
        assert len(backend.sent_messages) == 1
        assert backend.sent_messages[0]["to"] == "user@example.com"

    async def test_execute_send_email_with_optional_params(self) -> None:
        """Execute send_email should pass cc and attachments to backend."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:send_email",
            {
                "to": "user@example.com",
                "subject": "Test",
                "body": "Body",
                "cc": "cc@example.com",
                "attachments": ["file1.pdf", "file2.png"],
            },
        )

        assert result.success is True
        assert backend.sent_messages[0]["cc"] == "cc@example.com"
        assert backend.sent_messages[0]["attachments"] == ["file1.pdf", "file2.png"]

    async def test_execute_read_inbox_success(self) -> None:
        """Execute read_inbox should delegate to backend and return messages."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:read_inbox",
            {"folder": "inbox", "limit": 10},
        )

        assert result.success is True
        assert result.data is not None
        assert "messages" in result.data
        assert len(result.data["messages"]) == 2

    async def test_execute_read_inbox_respects_limit(self) -> None:
        """Execute read_inbox should pass limit to backend correctly."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:read_inbox",
            {"folder": "inbox", "limit": 1},
        )

        assert result.success is True
        assert len(result.data["messages"]) == 1

    async def test_execute_read_inbox_default_folder(self) -> None:
        """Execute read_inbox should use default folder if not specified."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:read_inbox",
            {"limit": 5},
        )

        assert result.success is True

    async def test_execute_read_inbox_default_limit(self) -> None:
        """Execute read_inbox should use default limit of 10 if not specified."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:read_inbox",
            {"folder": "inbox"},
        )

        assert result.success is True
        # Backend has 2 messages, default limit 10 should return both
        assert len(result.data["messages"]) == 2

    async def test_execute_parse_email_success(self) -> None:
        """Execute parse_email should delegate to backend and return data."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:parse_email",
            {"message_id": "msg-001"},
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["message_id"] == "msg-001"
        assert "body" in result.data

    async def test_execute_parse_email_not_found(self) -> None:
        """Execute parse_email should return empty data for unknown message."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:parse_email",
            {"message_id": "unknown-id"},
        )

        assert result.success is True
        assert result.data == {}

    async def test_execute_unknown_tool(self) -> None:
        """Execute should return error for unknown tool name."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute("email:unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_execute_send_email_backend_error(self) -> None:
        """Execute should propagate backend errors as failed ToolResult."""
        backend = FailingEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:send_email",
            {"to": "user@example.com", "subject": "Test", "body": "Body"},
        )

        assert result.success is False
        assert "401 Unauthorized" in result.error

    async def test_execute_read_inbox_backend_error(self) -> None:
        """Execute should propagate read_inbox backend errors."""
        backend = FailingEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:read_inbox",
            {"limit": 10},
        )

        assert result.success is False
        assert "IMAP connection failed" in result.error

    async def test_execute_parse_email_backend_error(self) -> None:
        """Execute should propagate parse backend errors."""
        backend = FailingEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:parse_email",
            {"message_id": "msg-001"},
        )

        assert result.success is False
        assert "Message not found" in result.error

    async def test_execute_read_inbox_invalid_limit(self) -> None:
        """Execute read_inbox should return clear error for non-integer limit."""
        backend = MockEmailBackend()
        module = EmailModule(backend)

        result = await module.execute(
            "email:read_inbox",
            {"limit": "not-a-number"},
        )

        assert result.success is False
        assert "limit must be an integer" in result.error


class TestEmailBackendProtocol:
    """Tests to verify EmailBackend protocol contract."""

    def test_mock_backend_satisfies_protocol(self) -> None:
        """MockEmailBackend should satisfy EmailBackend protocol."""
        backend: EmailBackend = MockEmailBackend()
        assert hasattr(backend, "send")
        assert hasattr(backend, "read_inbox")
        assert hasattr(backend, "parse")

    def test_failing_backend_satisfies_protocol(self) -> None:
        """FailingEmailBackend should satisfy EmailBackend protocol."""
        backend: EmailBackend = FailingEmailBackend()
        assert hasattr(backend, "send")
        assert hasattr(backend, "read_inbox")
        assert hasattr(backend, "parse")

    def test_mock_backend_is_runtime_checkable(self) -> None:
        """MockEmailBackend should pass isinstance check with runtime_checkable."""
        backend = MockEmailBackend()
        assert isinstance(backend, EmailBackend)

    def test_failing_backend_is_runtime_checkable(self) -> None:
        """FailingEmailBackend should pass isinstance check with runtime_checkable."""
        backend = FailingEmailBackend()
        assert isinstance(backend, EmailBackend)

    def test_non_backend_fails_isinstance(self) -> None:
        """A class without protocol methods should fail isinstance check."""
        assert not isinstance(object(), EmailBackend)
