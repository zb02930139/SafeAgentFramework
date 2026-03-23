"""LLM protocol types for SafeAgent core."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

#: Separator used in SafeAgent tool names (e.g. ``"filesystem:read_file"``).
_TOOL_NAME_SEPARATOR = ":"
#: Replacement used when a provider forbids colons in function names.
_TOOL_NAME_REPLACEMENT = "__"


def sanitize_tool_name(name: str) -> str:
    """Replace the first colon in *name* with ``__`` for provider compatibility.

    Some LLM providers (e.g. OpenAI) require function names to match
    ``^[a-zA-Z0-9_-]+$`` and therefore reject the colon namespace separator
    used by SafeAgent tool names (``"filesystem:read_file"``).

    Only the **first** colon is replaced, matching the ``namespace:tool`` convention
    enforced by :class:`~safe_agent.modules.base.ToolDescriptor`.  Tool names
    containing ``__`` are rejected at registration time, making the one-colon
    assumption safe by construction.

    The :class:`EventLoop` calls this automatically before passing tool
    definitions and message history to the :class:`LLMClient`, so
    client implementations do **not** need to handle the translation
    themselves.

    Args:
        name: A SafeAgent tool name such as ``"filesystem:read_file"``.

    Returns:
        The sanitized name, e.g. ``"filesystem__read_file"``.
    """
    return name.replace(_TOOL_NAME_SEPARATOR, _TOOL_NAME_REPLACEMENT, 1)


def restore_tool_name(name: str) -> str:
    """Reverse :func:`sanitize_tool_name` — restore the first ``__`` to ``:``.

    Args:
        name: A sanitized tool name such as ``"filesystem__read_file"``.

    Returns:
        The original SafeAgent tool name, e.g. ``"filesystem:read_file"``.
    """
    return name.replace(_TOOL_NAME_REPLACEMENT, _TOOL_NAME_SEPARATOR, 1)


class ToolCall(BaseModel):
    """Represents a model-requested tool invocation.

    Attributes:
        name: The fully-qualified tool name (e.g. ``"fs:ReadFile"``).
        params: Input parameters for the tool call.
        id: Optional unique identifier for the tool call, used for provenance
            tracking when the LLM makes multiple identical calls in one turn.
            Most LLM APIs (OpenAI, Anthropic) include this field; it propagates
            through the event loop, dispatcher, and audit log.
    """

    name: str
    params: dict[str, Any]
    id: str | None = None


class LLMResponse(BaseModel):
    """Represents a single LLM turn response."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMClient(Protocol):
    """Protocol for chat-capable LLM clients used by the event loop."""

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMResponse:
        """Return the next model response for the provided conversation.

        Tool names in *tools* and any prior ``tool_calls`` message entries
        will already be sanitized (colons replaced with ``__``) by the
        :class:`EventLoop` before this method is called.  Returned
        :class:`ToolCall` names must use the same sanitized form; the event
        loop restores the original colon-style names before dispatching.
        """
