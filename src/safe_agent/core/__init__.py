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

"""SafeAgent core primitives and orchestration types."""

from safe_agent.core.audit import AuditEntry, AuditLogger
from safe_agent.core.dispatcher import ToolDispatcher
from safe_agent.core.event_loop import MAX_TURNS_LIMIT, EventLoop, validate_max_turns
from safe_agent.core.llm import (
    LLMClient,
    LLMResponse,
    ToolCall,
    restore_tool_name,
    sanitize_tool_name,
)
from safe_agent.core.session import Session, SessionManager

__all__ = [
    "MAX_TURNS_LIMIT",
    "AuditEntry",
    "AuditLogger",
    "EventLoop",
    "LLMClient",
    "LLMResponse",
    "Session",
    "SessionManager",
    "ToolCall",
    "ToolDispatcher",
    "restore_tool_name",
    "sanitize_tool_name",
    "validate_max_turns",
]
