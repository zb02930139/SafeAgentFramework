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

"""Session-scoped event loop for LLM and tool orchestration."""

from __future__ import annotations

import asyncio
import json
import logging

from safe_agent.core.dispatcher import ToolDispatcher
from safe_agent.core.llm import LLMClient, restore_tool_name, sanitize_tool_name
from safe_agent.core.session import Session
from safe_agent.modules.registry import ModuleRegistry

logger = logging.getLogger(__name__)


def _trim_messages_preserve_pairs(session: Session) -> None:
    """Trim session messages to max_messages, preserving tool call/result pairs.

    When trimming would split an assistant message with tool_calls from its
    corresponding tool result messages, we keep the entire group to maintain
    conversation integrity and avoid API errors.
    """
    max_msgs = session.max_messages
    if len(session.messages) <= max_msgs:
        return

    # Find the safe trim point that doesn't split tool call/result pairs
    # Look for the earliest index where we can cut without orphaning tool results
    excess = len(session.messages) - max_msgs
    trim_index = excess

    # Walk forward from the trim point to find a safe boundary
    # A safe boundary is after any tool result that follows an assistant tool_calls
    while trim_index < len(session.messages):
        msg = session.messages[trim_index]
        # If this message is a tool result, we need to keep the previous assistant's
        # tool_calls, so back up to include that assistant message
        if msg.get("role") == "tool" and trim_index > 0:
            # Find the assistant message with tool_calls that precedes this tool
            prev_idx = trim_index - 1
            while (
                prev_idx >= 0 and session.messages[prev_idx].get("role") != "assistant"
            ):
                prev_idx -= 1
            if prev_idx >= 0 and session.messages[prev_idx].get("tool_calls"):
                # This tool belongs to an assistant tool_calls, include the assistant
                trim_index = prev_idx
                break
        # If this is a user message, it's a safe boundary
        if msg.get("role") == "user":
            break
        # Otherwise continue to next message
        trim_index += 1

    # Ensure we don't trim more than we need to
    if trim_index > excess:
        trim_index = excess

    # Actually trim the messages
    if trim_index > 0:
        del session.messages[:trim_index]


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of *messages* with tool names sanitized for LLM providers.

    Rewrites ``tool_calls[].name`` in assistant messages and ``name`` in tool
    result messages so all tool references use ``__`` instead of ``:`` as the
    namespace separator.

    **Precondition:** tool_calls entries must use the flat SafeAgent format
    ``{"name": ..., "params": ...}``, not the OpenAI-native
    ``{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}``
    structure.  The session transcript always stores the SafeAgent format;
    LLM clients are responsible for any further provider-specific adaptation.

    This function performs a *shallow* copy of each message dict and each
    tool_call dict.  Nested values (e.g. ``params``) are **not** deep-copied;
    the caller must not mutate them.
    """
    sanitized = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            sanitized_calls = [
                {**tc, "name": sanitize_tool_name(tc["name"]) if "name" in tc else tc}
                for tc in msg["tool_calls"]
                if "name" in tc
            ]
            sanitized.append({**msg, "tool_calls": sanitized_calls})
        elif msg.get("role") == "tool" and msg.get("name"):
            sanitized.append({**msg, "name": sanitize_tool_name(msg["name"])})
        else:
            sanitized.append(msg)
    return sanitized


class EventLoop:
    """Coordinates LLM responses and tool execution for a session.

    Callers should invoke :meth:`release_session` when a session is done so
    per-session locks do not accumulate indefinitely.
    """

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        llm_client: LLMClient,
        registry: ModuleRegistry,
        max_turns: int = 10,
    ) -> None:
        """Initialise the event loop dependencies and turn limit."""
        self._dispatcher = dispatcher
        self._llm_client = llm_client
        self._registry = registry
        self._max_turns = max_turns
        self._session_locks: dict[str, asyncio.Lock] = {}

    @property
    def max_turns(self) -> int:
        """Return the maximum turn limit for tool-call loops."""
        return self._max_turns

    def release_session(self, session_id: str) -> None:
        """Release any per-session resources once a session is finished."""
        self._session_locks.pop(session_id, None)

    async def process_turn(self, session: Session, user_message: str) -> str:
        """Process one user turn until the model returns final text.

        The loop serializes work per session, appends messages to the session
        transcript, dispatches any model-requested tools, and enforces a hard
        turn limit by making a final text-only model call when necessary.
        """
        lock = self._session_locks.setdefault(session.id, asyncio.Lock())

        async with lock:
            session.messages.append({"role": "user", "content": user_message})

            # Build tool definitions with sanitized names for LLM provider
            # compatibility (some providers forbid colons in function names).
            tool_definitions = [
                {**descriptor.model_dump(), "name": sanitize_tool_name(descriptor.name)}
                for descriptor in self._registry.get_all_tool_descriptors()
            ]

            turn_count = 0
            while True:
                # Pass sanitized tool names and a sanitized view of any prior
                # tool_calls entries in the session history.
                sanitized_messages = _sanitize_messages(session.messages)
                response = await self._llm_client.chat(
                    sanitized_messages,
                    tool_definitions,
                )

                if response.tool_calls:
                    # Process tool calls first. Also record content if present,
                    # since LLM can return both content and tool_calls.
                    if response.content is not None:
                        # Store content first, then append tool_calls to preserve
                        # the full assistant response in message history
                        session.messages.append(
                            {"role": "assistant", "content": response.content}
                        )
                    # Restore canonical colon-style names before storing in the
                    # session transcript and dispatching.
                    restored_calls = [
                        tool_call.model_copy(
                            update={"name": restore_tool_name(tool_call.name)}
                        )
                        for tool_call in response.tool_calls
                    ]
                    session.messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                tool_call.model_dump() for tool_call in restored_calls
                            ],
                        }
                    )
                    for tool_call in restored_calls:
                        try:
                            result = await self._dispatcher.dispatch(
                                tool_call.name,
                                tool_call.params,
                                session.id,
                                tool_call.id,
                            )
                            content = json.dumps(result.model_dump())
                        except Exception:
                            logger.exception(
                                "Tool dispatch failed: %s (call_id=%s)",
                                tool_call.name,
                                tool_call.id,
                            )
                            content = json.dumps(
                                {"error": f"Tool execution failed: {tool_call.name}"}
                            )

                        tool_msg: dict = {
                            "role": "tool",
                            "name": tool_call.name,
                            "content": content,
                        }
                        if tool_call.id is not None:
                            tool_msg["tool_call_id"] = tool_call.id
                        session.messages.append(tool_msg)

                elif response.content is not None:
                    # Content-only response (no tool_calls) - append and break loop
                    session.messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                    break

                turn_count += 1
                if turn_count >= self._max_turns:
                    final_response = await self._llm_client.chat(
                        _sanitize_messages(session.messages), []
                    )
                    final_text = final_response.content or ""
                    session.messages.append(
                        {"role": "assistant", "content": final_text}
                    )
                    break

            # Apply trimming once at the end of the turn, preserving
            # tool call/result pairs
            _trim_messages_preserve_pairs(session)

            # Return the last assistant message content
            last_msg = session.messages[-1]
            return str(last_msg.get("content", ""))
