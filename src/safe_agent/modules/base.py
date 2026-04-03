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

"""Base module protocol types and abstract base class for SafeAgent modules."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolDescriptor(BaseModel):
    """Describes a single tool exposed by a module.

    Attributes:
        name: Unique tool name within the module namespace.
        description: Human-readable description of the tool.
        parameters: JSON Schema dict describing the tool's input parameters.
        action: Action identifier string, e.g. "filesystem:ReadFile".
        resource_param: Name(s) of parameter(s) that identify the resource.
            Always stored as a list; a bare string is coerced automatically.
        condition_keys: List of condition keys used for policy evaluation.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    action: str
    resource_param: list[str] = Field(default_factory=list)
    condition_keys: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        """Reject names containing ``__`` (double underscore).

        SafeAgent uses ``__`` as the LLM-boundary sanitization replacement for
        the ``:`` namespace separator (see
        :func:`safe_agent.core.llm.sanitize_tool_name`).
        A tool name that already contains ``__`` would make the colon-restore
        round-trip ambiguous and silently corrupt dispatch.
        """
        if not v or not v.strip():
            raise ValueError("tool name must not be empty")
        if "__" in v:
            msg = (
                f"Tool name {v!r} must not contain '__' (double underscore); "
                "that sequence is reserved for LLM provider name sanitization."
            )
            raise ValueError(msg)
        return v

    @field_validator("resource_param", mode="before")
    @classmethod
    def _coerce_resource_param(cls, v: Any) -> list[str]:
        """Coerce a bare string into a single-element list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class ModuleDescriptor(BaseModel):
    """Describes a module and all the tools it provides.

    Attributes:
        namespace: Unique namespace string for this module.
        description: Human-readable description of the module.
        tools: List of tool descriptors exposed by this module.
    """

    namespace: str
    description: str
    tools: list[ToolDescriptor] = Field(default_factory=list)


class ToolResult[ToolDataT = Any](BaseModel):
    """Represents the result of a tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        data: Optional result data returned by the tool.
        error: Optional error message if execution failed.
        metadata: Additional metadata about the execution.

    Notes:
        ``ToolResult`` is generic so module authors can describe the expected
        shape of ``data`` more precisely than ``Any``. Consumers must still
        validate untrusted tool output before sending it to sensitive sinks
        such as shell commands, SQL, template engines, or dynamic execution.
    """

    success: bool
    data: ToolDataT | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ToolResult[ToolDataT]":
        """Enforce consistency between success, data, and error.

        - success=True implies error is None
        - success=False implies data is None
        """
        if self.success and self.error is not None:
            raise ValueError(
                "ToolResult with success=True must not have an error message"
            )
        if not self.success and self.data is not None:
            raise ValueError("ToolResult with success=False must not have data")
        return self


class BaseModule(ABC):
    """Abstract base class for all SafeAgent modules.

    Subclass this to implement a module that exposes tools to the SafeAgent
    framework. Each module declares its namespace and tools via ``describe()``,
    resolves runtime policy conditions via ``resolve_conditions()``, and
    executes tool invocations via ``execute()``.

    Note: ``describe()`` is synchronous by design. Modules must not perform
    async work inside ``describe()`` — use ``__init__`` for eager sync
    initialisation or add a dedicated async ``setup()`` lifecycle method in
    your concrete subclass if async setup is required. Descriptors are expected
    to be stable and inexpensive to produce.
    """

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation including the module namespace."""
        try:
            ns = self.describe().namespace
        except Exception:
            ns = "<unknown>"
        return f"<{type(self).__name__} namespace={ns!r}>"

    @abstractmethod
    def describe(self) -> ModuleDescriptor:
        """Return the descriptor for this module including all its tools.

        Returns:
            A ``ModuleDescriptor`` with the module's namespace, description,
            and the list of ``ToolDescriptor`` objects it exposes.
        """

    @abstractmethod
    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve runtime condition values for policy evaluation.

        Args:
            tool_name: The name of the tool being invoked.
            params: The raw input parameters for the invocation.

        Returns:
            A dict mapping condition key names to their resolved runtime values.
        """

    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a tool invocation and return the result.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ``ToolResult`` indicating success or failure, plus any data.
        """
