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


"""Tests for safe_agent.modules.base — models and BaseModule ABC."""

from typing import Any

import pytest
from pydantic import ValidationError

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteModule(BaseModule):
    """Minimal concrete subclass of BaseModule used for testing."""

    def describe(self) -> ModuleDescriptor:
        """Return a simple ModuleDescriptor for testing."""
        return ModuleDescriptor(
            namespace="test",
            description="A test module.",
            tools=[
                ToolDescriptor(
                    name="test:greet",
                    description="Says hello.",
                    parameters={"type": "object", "properties": {}},
                    action="test:Greet",
                    resource_param="name",
                    condition_keys=["time_of_day"],
                )
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a stub condition map."""
        return {"time_of_day": "morning"}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[str]:
        """Return a successful ToolResult with greeting data."""
        return ToolResult[str](
            success=True,
            data=f"Hello, {params.get('name', 'world')}!",
        )


# ---------------------------------------------------------------------------
# ToolDescriptor
# ---------------------------------------------------------------------------


class TestToolDescriptor:
    """Tests for ToolDescriptor model validation and defaults."""

    def test_required_fields(self) -> None:
        """ToolDescriptor should accept all required fields."""
        td = ToolDescriptor(
            name="fs:ReadFile",
            description="Read a file.",
            action="filesystem:ReadFile",
        )
        assert td.name == "fs:ReadFile"
        assert td.action == "filesystem:ReadFile"

    def test_defaults(self) -> None:
        """ToolDescriptor optional fields should default to empty collections."""
        td = ToolDescriptor(
            name="x:y",
            description="desc",
            action="x:Y",
        )
        assert td.parameters == {}
        assert td.resource_param == []
        assert td.condition_keys == []

    def test_resource_param_list(self) -> None:
        """resource_param should accept a list of strings."""
        td = ToolDescriptor(
            name="x:y",
            description="d",
            action="x:Y",
            resource_param=["bucket", "key"],
        )
        assert td.resource_param == ["bucket", "key"]

    def test_resource_param_string_coerced_to_list(self) -> None:
        """A bare string resource_param should be coerced to a single-element list."""
        td = ToolDescriptor(
            name="x:y",
            description="d",
            action="x:Y",
            resource_param="path",
        )
        assert td.resource_param == ["path"]

    def test_resource_param_always_list(self) -> None:
        """resource_param should always be a list, never a bare string."""
        td = ToolDescriptor(
            name="x:y",
            description="d",
            action="x:Y",
            resource_param="single",
        )
        assert isinstance(td.resource_param, list)

    def test_parameters_schema(self) -> None:
        """Parameters should accept a JSON-Schema-like dict."""
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        td = ToolDescriptor(
            name="x:y",
            description="d",
            action="x:Y",
            parameters=schema,
        )
        assert td.parameters == schema

    def test_resource_param_none_returns_empty_list(self) -> None:
        """resource_param=None should result in [] (not TypeError)."""
        td = ToolDescriptor(
            name="x:y",
            description="d",
            action="x:Y",
            resource_param=None,
        )
        assert td.resource_param == []


# ---------------------------------------------------------------------------
# ModuleDescriptor
# ---------------------------------------------------------------------------


class TestModuleDescriptor:
    """Tests for ModuleDescriptor model validation and defaults."""

    def test_basic_construction(self) -> None:
        """ModuleDescriptor should store namespace, description, and tools."""
        md = ModuleDescriptor(
            namespace="fs",
            description="Filesystem module.",
            tools=[],
        )
        assert md.namespace == "fs"
        assert md.tools == []

    def test_default_tools(self) -> None:
        """ModuleDescriptor tools should default to an empty list."""
        md = ModuleDescriptor(namespace="x", description="d")
        assert md.tools == []

    def test_tools_list(self) -> None:
        """ModuleDescriptor should hold a list of ToolDescriptors."""
        td = ToolDescriptor(name="x:y", description="d", action="x:Y")
        md = ModuleDescriptor(namespace="x", description="d", tools=[td])
        assert len(md.tools) == 1
        assert md.tools[0].name == "x:y"


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    """Tests for ToolResult model validation and defaults."""

    def test_success_with_data(self) -> None:
        """ToolResult should store success flag and data."""
        r = ToolResult[dict[str, int]](success=True, data={"value": 42})
        assert r.success is True
        assert r.data == {"value": 42}
        assert r.error is None

    def test_generic_string_payload(self) -> None:
        """ToolResult should support typed scalar payloads via generics."""
        r = ToolResult[str](success=True, data="hello")
        assert r.data == "hello"

    def test_generic_payload_validation_rejects_wrong_type(self) -> None:
        """Parameterized ToolResult should validate the payload type at runtime."""
        with pytest.raises(ValidationError):
            ToolResult[int](success=True, data="not-an-int")

    def test_failure_with_error(self) -> None:
        """ToolResult should store error message on failure."""
        r = ToolResult(success=False, error="Not found")
        assert r.success is False
        assert r.error == "Not found"
        assert r.data is None

    def test_metadata_default(self) -> None:
        """ToolResult metadata should default to an empty dict."""
        r = ToolResult(success=True)
        assert r.metadata == {}

    def test_metadata_populated(self) -> None:
        """ToolResult should accept arbitrary metadata."""
        r = ToolResult(success=True, metadata={"latency_ms": 12})
        assert r.metadata["latency_ms"] == 12

    def test_success_true_with_error_rejected(self) -> None:
        """ToolResult(success=True, error='msg') should raise ValidationError."""
        with pytest.raises(ValidationError):
            ToolResult(success=True, error="msg")

    def test_success_false_with_data_rejected(self) -> None:
        """ToolResult(success=False, data='some data') should raise ValidationError."""
        with pytest.raises(ValidationError):
            ToolResult(success=False, data="some data")

    def test_success_true_with_none_error_allowed(self) -> None:
        """ToolResult(success=True, error=None, data='ok') should be allowed."""
        r = ToolResult(success=True, error=None, data="ok")
        assert r.success is True
        assert r.error is None
        assert r.data == "ok"

    def test_success_false_with_none_data_allowed(self) -> None:
        """ToolResult(success=False, data=None, error='fail') should be allowed."""
        r = ToolResult(success=False, data=None, error="fail")
        assert r.success is False
        assert r.data is None
        assert r.error == "fail"


# ---------------------------------------------------------------------------
# BaseModule
# ---------------------------------------------------------------------------


class TestBaseModule:
    """Tests for the BaseModule abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        """BaseModule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModule()  # type: ignore[abstract]

    def test_concrete_subclass_describe(self) -> None:
        """ConcreteModule.describe() returns a valid ModuleDescriptor."""
        m = ConcreteModule()
        desc = m.describe()
        assert isinstance(desc, ModuleDescriptor)
        assert desc.namespace == "test"

    async def test_resolve_conditions(self) -> None:
        """ConcreteModule.resolve_conditions() returns a dict."""
        m = ConcreteModule()
        conds = await m.resolve_conditions("test:greet", {"name": "Alice"})
        assert isinstance(conds, dict)
        assert conds["time_of_day"] == "morning"

    async def test_execute_returns_tool_result(self) -> None:
        """ConcreteModule.execute() returns a ToolResult."""
        m = ConcreteModule()
        result = await m.execute("test:greet", {"name": "Alice"})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == "Hello, Alice!"


class TestToolDescriptorNameValidation:
    """Tests for the ToolDescriptor name validator."""

    def test_valid_name_accepted(self) -> None:
        """A normal namespace:tool name is accepted."""
        td = ToolDescriptor(
            name="filesystem:read_file",
            description="Read a file.",
            parameters={},
            action="filesystem:ReadFile",
        )
        assert td.name == "filesystem:read_file"

    def test_double_underscore_rejected(self) -> None:
        """A name containing __ must be rejected to prevent sanitization ambiguity."""
        with pytest.raises(ValidationError, match="__"):
            ToolDescriptor(
                name="my_module:list__items",
                description="Bad name.",
                parameters={},
                action="my_module:ListItems",
            )

    def test_single_underscore_allowed(self) -> None:
        """Single underscores in tool names are fine."""
        td = ToolDescriptor(
            name="shell:run_command",
            description="Run a command.",
            parameters={},
            action="shell:RunCommand",
        )
        assert td.name == "shell:run_command"

    def test_empty_name_rejected(self) -> None:
        """An empty name should raise ValidationError with 'empty' message."""
        with pytest.raises(ValidationError, match="empty"):
            ToolDescriptor(
                name="",
                description="Empty name test.",
                parameters={},
                action="test:Empty",
            )

    def test_whitespace_only_name_rejected(self) -> None:
        """A whitespace-only name should raise ValidationError with 'empty' message."""
        with pytest.raises(ValidationError, match="empty"):
            ToolDescriptor(
                name="   ",
                description="Whitespace name test.",
                parameters={},
                action="test:Whitespace",
            )
