"""Email interface module for SafeAgent using pluggable adapter pattern."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


@runtime_checkable
class EmailBackend(Protocol):
    """Protocol for provider-specific email operations.

    Implementations provide concrete email sending, reading, and parsing
    capabilities. The framework's EmailModule delegates execute() calls to
    an instance of this protocol.

    All methods accept **kwargs for provider-specific options that don't
    belong in the core interface but may be needed for specific backends
    (e.g., SendGrid template IDs, SES configuration sets, etc.).
    """

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send an email to one or more recipients.

        Args:
            to: Recipient email address(es). For multiple recipients,
                implementations may accept comma-separated or handle
                as appropriate for the provider.
            subject: Email subject line.
            body: Email body content (plain text or HTML, per implementation).
            **kwargs: Provider-specific options (cc, bcc, attachments, etc.).

        Returns:
            Provider-specific result dict, typically including message_id
            and/or status information.
        """
        ...

    async def read_inbox(
        self,
        folder: str,
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Read messages from a mailbox folder.

        Args:
            folder: Mailbox folder name (e.g., "inbox", "sent", "archive").
            limit: Maximum number of messages to return.
            **kwargs: Provider-specific filter options.

        Returns:
            List of message metadata dicts, each typically containing
            message_id, sender, subject, date, and other summary fields.
        """
        ...

    async def parse(
        self,
        message_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract structured data from an email body.

        Args:
            message_id: Unique identifier for the email message.
            **kwargs: Extraction options (e.g., extract_fields list).

        Returns:
            Structured data extracted from the email, typically including
            body text, headers, and any extracted fields requested.
        """
        ...


class EmailModule(BaseModule):
    """Email interface module using pluggable adapter pattern.

    The framework defines the IAM surface (namespace, tool descriptors,
    condition keys); a plugin provides the concrete EmailBackend implementation.
    This allows administrators to write policies against stable identifiers
    like "email:SendEmail" regardless of whether the backend is SendGrid,
    Amazon SES, SMTP relay, etc.

    Example usage:
        # In adapter package
        class SendGridBackend:
            async def send(self, to, subject, body, **kwargs):
                # SendGrid-specific implementation
                ...

        def create_email_module(config):
            return EmailModule(backend=SendGridBackend(config))
    """

    def __init__(self, backend: EmailBackend) -> None:
        """Initialize the email module with a provider-specific backend.

        Args:
            backend: An implementation of EmailBackend protocol that handles
                the actual email operations for a specific provider.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the email module descriptor with three tools.

        Tools:
            email:send_email: Send an email to one or more recipients.
            email:read_inbox: Read messages from a mailbox folder.
            email:parse_email: Extract structured data from an email body.
        """
        return ModuleDescriptor(
            namespace="email",
            description="Send and receive emails via pluggable backend adapter.",
            tools=[
                ToolDescriptor(
                    name="email:send_email",
                    description="Send an email to one or more recipients.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                            "cc": {"type": "string"},
                            "attachments": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["to", "subject", "body"],
                        "additionalProperties": False,
                    },
                    action="email:SendEmail",
                    resource_param=["to"],
                    condition_keys=[
                        "email:Recipient",
                        "email:Sender",
                        "email:Subject",
                    ],
                ),
                ToolDescriptor(
                    name="email:read_inbox",
                    description="Read messages from a mailbox folder.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "folder": {"type": "string", "default": "inbox"},
                            "limit": {"type": "integer"},
                            "filter": {"type": "string"},
                        },
                        "required": ["limit"],
                        "additionalProperties": False,
                    },
                    action="email:ReadInbox",
                    resource_param=["folder"],
                    condition_keys=["email:Folder"],
                ),
                ToolDescriptor(
                    name="email:parse_email",
                    description="Extract structured data from an email body.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string"},
                            "extract_fields": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["message_id"],
                        "additionalProperties": False,
                    },
                    action="email:ParseEmail",
                    resource_param=["message_id"],
                    condition_keys=[],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve email condition values for policy evaluation.

        Condition keys resolved:
            email:Recipient: The 'to' parameter (recipient address(es)).
            email:Sender: The sender address from backend context (if available).
            email:Subject: The 'subject' parameter.
            email:Folder: The 'folder' parameter for read_inbox.

        Note: email:Sender requires backend context and is not resolved here
        for send_email tool, as the sender is determined by the backend
        configuration (e.g., the authenticated account), not by parameters.
        For read operations, the sender would come from message metadata.
        """
        normalized_tool = tool_name.removeprefix("email:")

        if normalized_tool == "send_email":
            conditions: dict[str, Any] = {
                "email:Recipient": params.get("to", ""),
                "email:Subject": params.get("subject", ""),
            }
            # Sender is determined by backend configuration, not params
            # Backends may inject this via context if needed
            return conditions

        if normalized_tool == "read_inbox":
            return {"email:Folder": params.get("folder", "inbox")}

        # parse_email doesn't resolve conditions from params
        return {}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute an email tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute (e.g., "email:send_email").
            params: The input parameters for the tool.

        Returns:
            ToolResult with success status and data from backend, or error.
        """
        normalized_tool = tool_name.removeprefix("email:")

        try:
            if normalized_tool == "send_email":
                return await self._execute_send(params)
            if normalized_tool == "read_inbox":
                return await self._execute_read_inbox(params)
            if normalized_tool == "parse_email":
                return await self._execute_parse(params)

            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        except Exception as exc:
            # Propagate backend errors as failed ToolResult
            return ToolResult(success=False, error=str(exc))

    async def _execute_send(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Execute email:send_email by delegating to backend."""
        to = str(params.get("to", ""))
        subject = str(params.get("subject", ""))
        body = str(params.get("body", ""))

        # Build kwargs for optional parameters
        kwargs: dict[str, Any] = {}
        if "cc" in params:
            kwargs["cc"] = str(params["cc"])
        if "attachments" in params:
            attachments = params["attachments"]
            if isinstance(attachments, list):
                kwargs["attachments"] = [str(a) for a in attachments]

        result = await self._backend.send(
            to=to,
            subject=subject,
            body=body,
            **kwargs,
        )
        return ToolResult(success=True, data=result)

    async def _execute_read_inbox(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Execute email:read_inbox by delegating to backend."""
        folder = str(params.get("folder", "inbox"))
        raw_limit = params.get("limit", 10)
        if not isinstance(raw_limit, int):
            try:
                limit = int(raw_limit)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    error=f"limit must be an integer, got {type(raw_limit).__name__}",
                )
        else:
            limit = raw_limit

        kwargs: dict[str, Any] = {}
        if "filter" in params:
            kwargs["filter"] = str(params["filter"])

        messages = await self._backend.read_inbox(
            folder=folder,
            limit=limit,
            **kwargs,
        )
        return ToolResult(success=True, data={"messages": messages})

    async def _execute_parse(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Execute email:parse_email by delegating to backend."""
        message_id = str(params.get("message_id", ""))

        kwargs: dict[str, Any] = {}
        if "extract_fields" in params:
            fields = params["extract_fields"]
            if isinstance(fields, list):
                kwargs["extract_fields"] = [str(f) for f in fields]

        result = await self._backend.parse(
            message_id=message_id,
            **kwargs,
        )
        return ToolResult(success=True, data=result)
