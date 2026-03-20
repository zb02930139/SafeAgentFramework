"""Tests for safe_agent.modules.base — models and BaseModule ABC."""

from typing import Any

import pytest

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
    ) -> ToolResult:
        """Return a successful ToolResult with greeting data."""
        return ToolResult(success=True, data=f"Hello, {params.get('name', 'world')}!")


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
        r = ToolResult(success=True, data={"value": 42})
        assert r.success is True
        assert r.data == {"value": 42}
        assert r.error is None

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
