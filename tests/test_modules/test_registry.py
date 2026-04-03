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


"""Tests for safe_agent.modules.registry — ModuleRegistry."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(
    namespace: str,
    tool_names: list[str],
) -> BaseModule:
    """Create a minimal concrete BaseModule for testing.

    Args:
        namespace: The namespace to assign to the module.
        tool_names: List of tool names the module should expose.

    Returns:
        A concrete ``BaseModule`` instance.
    """

    class _Module(BaseModule):
        """Dynamically created test module."""

        def describe(self) -> ModuleDescriptor:
            """Return a descriptor with the configured namespace and tools."""
            return ModuleDescriptor(
                namespace=namespace,
                description=f"Module {namespace}.",
                tools=[
                    ToolDescriptor(
                        name=t,
                        description=f"Tool {t}.",
                        action=t,
                    )
                    for t in tool_names
                ],
            )

        async def resolve_conditions(
            self,
            tool_name: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            """Return empty conditions."""
            return {}

        async def execute(
            self,
            tool_name: str,
            params: dict[str, Any],
        ) -> ToolResult[Any]:
            """Return a default success result."""
            return ToolResult(success=True)

    return _Module()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModuleRegistry:
    """Tests for ModuleRegistry registration and lookup."""

    def test_register_and_get_tool(self) -> None:
        """A registered tool should be retrievable by name."""
        registry = ModuleRegistry()
        module = _make_module("alpha", ["alpha:Read", "alpha:Write"])
        registry.register(module)

        result = registry.get_tool("alpha:Read")
        assert result is not None
        found_module, descriptor = result
        assert found_module is module
        assert descriptor.name == "alpha:Read"

    def test_get_tool_returns_none_for_unknown(self) -> None:
        """get_tool() should return None for an unregistered tool name."""
        registry = ModuleRegistry()
        assert registry.get_tool("nonexistent:tool") is None

    def test_get_all_tool_descriptors_empty(self) -> None:
        """get_all_tool_descriptors() should return [] when no modules registered."""
        registry = ModuleRegistry()
        assert registry.get_all_tool_descriptors() == []

    def test_get_all_tool_descriptors(self) -> None:
        """get_all_tool_descriptors() should return all registered tools."""
        registry = ModuleRegistry()
        registry.register(_make_module("a", ["a:T1", "a:T2"]))
        registry.register(_make_module("b", ["b:T3"]))

        descriptors = registry.get_all_tool_descriptors()
        names = {d.name for d in descriptors}
        assert names == {"a:T1", "a:T2", "b:T3"}

    def test_namespace_collision_raises(self) -> None:
        """Registering two modules with the same namespace raises ValueError."""
        registry = ModuleRegistry()
        registry.register(_make_module("clash", ["clash:A"]))
        second = _make_module("clash", ["clash:B"])
        with pytest.raises(ValueError, match="Namespace collision"):
            registry.register(second)

    def test_tool_name_collision_raises(self) -> None:
        """Registering a tool whose name is already taken raises ValueError.

        This prevents silent shadowing of tools across modules.
        """
        registry = ModuleRegistry()
        registry.register(_make_module("a", ["shared:Tool"]))
        second = _make_module("b", ["shared:Tool"])
        with pytest.raises(ValueError, match="Tool name collision"):
            registry.register(second)

    def test_intra_module_duplicate_tool_name_raises(self) -> None:
        """A module whose describe() returns duplicate tool names raises ValueError.

        Previously the second duplicate silently overwrote the first in
        _tool_map. This test fixes issue #136.
        """
        registry = ModuleRegistry()
        duplicate = _make_module("ns", ["ns:tool", "ns:tool"])
        with pytest.raises(ValueError, match="Duplicate tool name 'ns:tool'"):
            registry.register(duplicate)

    def test_intra_module_duplicate_is_atomic(self) -> None:
        """A duplicate within a module must not partially register the namespace."""
        registry = ModuleRegistry()
        duplicate = _make_module("dup", ["dup:A", "dup:B", "dup:A"])
        with pytest.raises(ValueError, match="Duplicate tool name"):
            registry.register(duplicate)
        # Nothing should have been committed.
        assert registry.get_module("dup") is None
        assert registry.get_tool("dup:A") is None
        assert registry.get_tool("dup:B") is None

    def test_tool_name_collision_is_atomic(self) -> None:
        """If a tool name collision occurs, no partial state is committed."""
        registry = ModuleRegistry()
        registry.register(_make_module("existing", ["existing:A"]))
        # Module with two tools: first is unique, second collides.
        colliding = _make_module("new", ["new:Unique", "existing:A"])
        with pytest.raises(ValueError, match="Tool name collision"):
            registry.register(colliding)
        # Neither the namespace nor the unique tool should have been registered.
        assert registry.get_module("new") is None
        assert registry.get_tool("new:Unique") is None

    def test_idempotent_registration(self) -> None:
        """Registering the same module instance twice should not raise."""
        registry = ModuleRegistry()
        module = _make_module("idem", ["idem:X"])
        registry.register(module)
        registry.register(module)  # should be a no-op
        descriptors = registry.get_all_tool_descriptors()
        assert len(descriptors) == 1

    def test_multiple_tools_from_one_module(self) -> None:
        """All tools of a module should be accessible after registration."""
        registry = ModuleRegistry()
        tools = ["ns:T1", "ns:T2", "ns:T3"]
        registry.register(_make_module("ns", tools))
        for t in tools:
            assert registry.get_tool(t) is not None

    def test_get_tool_returns_correct_descriptor(self) -> None:
        """get_tool() descriptor should match the registered ToolDescriptor."""
        registry = ModuleRegistry()
        registry.register(_make_module("z", ["z:Foo"]))
        result = registry.get_tool("z:Foo")
        assert result is not None
        _, descriptor = result
        assert descriptor.action == "z:Foo"
        assert descriptor.description == "Tool z:Foo."

    def test_get_module_by_namespace(self) -> None:
        """get_module() should return the registered module for a known namespace."""
        registry = ModuleRegistry()
        module = _make_module("myns", ["myns:Op"])
        registry.register(module)
        assert registry.get_module("myns") is module

    def test_get_module_returns_none_for_unknown(self) -> None:
        """get_module() should return None for an unregistered namespace."""
        registry = ModuleRegistry()
        assert registry.get_module("ghost") is None

    def test_get_all_modules(self) -> None:
        """get_all_modules() should return all registered module instances."""
        registry = ModuleRegistry()
        m1 = _make_module("x", ["x:A"])
        m2 = _make_module("y", ["y:B"])
        registry.register(m1)
        registry.register(m2)
        modules = registry.get_all_modules()
        assert set(modules) == {m1, m2}

    def test_get_all_modules_empty(self) -> None:
        """get_all_modules() should return [] when nothing is registered."""
        registry = ModuleRegistry()
        assert registry.get_all_modules() == []

    # ---------------------------------------------------------------------------
    # discover() tests
    # ---------------------------------------------------------------------------

    def test_discover_raises_on_second_call(self) -> None:
        """discover() should raise RuntimeError if called more than once."""
        registry = ModuleRegistry()
        with patch("safe_agent.modules.registry.entry_points", return_value=[]):
            registry.discover()
            with pytest.raises(
                RuntimeError, match=r"discover\(\) has already been called"
            ):
                registry.discover()

    def test_discover_rejects_non_basemodule_class(self) -> None:
        """discover() should raise TypeError for a non-BaseModule subclass."""

        class NotAModule:
            """Not a module."""

        mock_ep = MagicMock()
        mock_ep.name = "bad_ep"
        mock_ep.value = "bad_pkg:NotAModule"
        mock_ep.load.return_value = NotAModule

        registry = ModuleRegistry()
        with patch("safe_agent.modules.registry.entry_points", return_value=[mock_ep]):
            with pytest.raises(TypeError, match="not a subclass of BaseModule"):
                registry.discover()

    def test_discover_rejects_non_class_entry_point(self) -> None:
        """discover() should raise TypeError if entry point loads a non-class."""
        mock_ep = MagicMock()
        mock_ep.name = "bad_ep"
        mock_ep.value = "bad_pkg:some_function"
        mock_ep.load.return_value = lambda: None  # a function, not a class

        registry = ModuleRegistry()
        with patch("safe_agent.modules.registry.entry_points", return_value=[mock_ep]):
            with pytest.raises(TypeError, match="not a subclass of BaseModule"):
                registry.discover()

    def test_discover_registers_valid_module(self) -> None:
        """discover() should register a valid BaseModule subclass."""
        module_instance = _make_module("discovered", ["discovered:Op"])
        ModuleClass = type(module_instance)

        mock_ep = MagicMock()
        mock_ep.name = "discovered_ep"
        mock_ep.value = "some_pkg:DiscoveredModule"
        mock_ep.load.return_value = ModuleClass

        registry = ModuleRegistry()
        with patch("safe_agent.modules.registry.entry_points", return_value=[mock_ep]):
            registry.discover()

        assert registry.get_module("discovered") is not None
        assert registry.get_tool("discovered:Op") is not None

    def test_discover_does_not_set_discovered_flag_on_failure(self) -> None:
        """If discover() fails mid-loop, _discovered must not be set True.

        This ensures the registry is not left in a poisoned state where it
        appears discovered but is incomplete.
        """

        class NotAModule:
            """Not a module."""

        mock_ep = MagicMock()
        mock_ep.name = "bad_ep"
        mock_ep.value = "bad_pkg:NotAModule"
        mock_ep.load.return_value = NotAModule

        registry = ModuleRegistry()
        with patch("safe_agent.modules.registry.entry_points", return_value=[mock_ep]):
            with pytest.raises(TypeError):
                registry.discover()

        # Registry should NOT be marked as discovered after failure.
        assert registry._discovered is False

    def test_discover_is_atomic_no_partial_state_on_type_error(self) -> None:
        """discover() must leave registry unchanged when it fails mid-loop.

        First entry point loads a valid module; second raises TypeError.
        Neither should appear in the registry after the failure.
        Fixes issue #11.
        """
        good_instance = _make_module("good", ["good:Op"])
        GoodClass = type(good_instance)

        class NotAModule:
            """Not a module."""

        good_ep = MagicMock()
        good_ep.name = "good_ep"
        good_ep.value = "pkg:GoodModule"
        good_ep.load.return_value = GoodClass

        bad_ep = MagicMock()
        bad_ep.name = "bad_ep"
        bad_ep.value = "pkg:NotAModule"
        bad_ep.load.return_value = NotAModule

        registry = ModuleRegistry()
        with patch(
            "safe_agent.modules.registry.entry_points",
            return_value=[good_ep, bad_ep],
        ):
            with pytest.raises(TypeError):
                registry.discover()

        # Both the valid entry point and the failed one must be absent.
        assert registry.get_module("good") is None
        assert registry.get_tool("good:Op") is None
        assert registry._discovered is False

    def test_discover_is_atomic_no_partial_state_on_collision(self) -> None:
        """discover() must leave registry unchanged when a collision occurs.

        First entry point loads fine; second collides on tool name.
        Neither should pollute the live registry. Fixes issue #11.
        """
        first_instance = _make_module("first", ["shared:Tool"])
        FirstClass = type(first_instance)

        second_instance = _make_module("second", ["shared:Tool"])
        SecondClass = type(second_instance)

        ep1 = MagicMock()
        ep1.name = "ep1"
        ep1.value = "pkg:First"
        ep1.load.return_value = FirstClass

        ep2 = MagicMock()
        ep2.name = "ep2"
        ep2.value = "pkg:Second"
        ep2.load.return_value = SecondClass

        registry = ModuleRegistry()
        with patch(
            "safe_agent.modules.registry.entry_points",
            return_value=[ep1, ep2],
        ):
            with pytest.raises(ValueError, match="Tool name collision"):
                registry.discover()

        assert registry.get_module("first") is None
        assert registry.get_module("second") is None
        assert registry.get_tool("shared:Tool") is None
        assert registry._discovered is False

    def test_discover_atomic_preserves_pre_existing_manual_registrations(
        self,
    ) -> None:
        """A failed discover() must not disturb manually registered modules.

        If register() was called before discover(), and discover() then fails,
        the manually registered module must still be present.
        """
        manual = _make_module("manual", ["manual:Op"])

        class NotAModule:
            """Not a module."""

        bad_ep = MagicMock()
        bad_ep.name = "bad_ep"
        bad_ep.value = "pkg:Bad"
        bad_ep.load.return_value = NotAModule

        registry = ModuleRegistry()
        registry.register(manual)

        with patch(
            "safe_agent.modules.registry.entry_points",
            return_value=[bad_ep],
        ):
            with pytest.raises(TypeError):
                registry.discover()

        # Manual registration must be intact.
        assert registry.get_module("manual") is manual
        assert registry.get_tool("manual:Op") is not None
        assert registry._discovered is False

    def test_discover_can_be_retried_after_failure(self) -> None:
        """discover() should be retryable after a failed attempt.

        A failure must leave ``_discovered`` false so a later clean retry can
        succeed on the same registry instance.
        """

        class NotAModule:
            """Not a module."""

        bad_ep = MagicMock()
        bad_ep.name = "bad_ep"
        bad_ep.value = "pkg:Bad"
        bad_ep.load.return_value = NotAModule

        good_instance = _make_module("retry", ["retry:Op"])
        GoodClass = type(good_instance)

        good_ep = MagicMock()
        good_ep.name = "good_ep"
        good_ep.value = "pkg:RetryModule"
        good_ep.load.return_value = GoodClass

        registry = ModuleRegistry()
        with patch(
            "safe_agent.modules.registry.entry_points",
            return_value=[bad_ep],
        ):
            with pytest.raises(TypeError):
                registry.discover()

        assert registry._discovered is False

        with patch(
            "safe_agent.modules.registry.entry_points",
            return_value=[good_ep],
        ):
            registry.discover()

        assert registry._discovered is True
        assert registry.get_module("retry") is not None
        assert registry.get_tool("retry:Op") is not None

    def test_discover_rejects_namespace_collision_with_manual_registration(
        self,
    ) -> None:
        """discover() should reject namespace collisions against live state.

        Pre-existing manual registrations must be included in the staging copy,
        and a colliding discovered module must fail without changing live state.
        """
        manual = _make_module("shared", ["shared:Manual"])
        discovered = _make_module("shared", ["shared:Discovered"])
        DiscoveredClass = type(discovered)

        ep = MagicMock()
        ep.name = "shared_ep"
        ep.value = "pkg:SharedModule"
        ep.load.return_value = DiscoveredClass

        registry = ModuleRegistry()
        registry.register(manual)

        with patch(
            "safe_agent.modules.registry.entry_points",
            return_value=[ep],
        ):
            with pytest.raises(ValueError, match="Namespace collision"):
                registry.discover()

        assert registry.get_module("shared") is manual
        assert registry.get_tool("shared:Manual") is not None
        assert registry.get_tool("shared:Discovered") is None
        assert registry._discovered is False

    def test_discover_intra_module_duplicate_tool_name_raises(self) -> None:
        """discover() should raise ValueError for a module with duplicate tool names.

        Covers issue #136 for the discover() path.
        """
        dup_instance = _make_module("dupmod", ["dupmod:tool", "dupmod:tool"])
        DupClass = type(dup_instance)

        mock_ep = MagicMock()
        mock_ep.name = "dup_ep"
        mock_ep.value = "pkg:DupModule"
        mock_ep.load.return_value = DupClass

        registry = ModuleRegistry()
        with patch("safe_agent.modules.registry.entry_points", return_value=[mock_ep]):
            with pytest.raises(ValueError, match="Duplicate tool name"):
                registry.discover()

        # Registry must remain clean.
        assert registry.get_module("dupmod") is None
        assert registry.get_tool("dupmod:tool") is None
        assert registry._discovered is False

    def test_discover_loads_builtin_modules_from_real_entry_points(self) -> None:
        """discover() should load FilesystemModule and ShellModule from package.

        Integration test verifying entry points in pyproject.toml resolve.
        Dogfood test for issue #28.
        """
        registry = ModuleRegistry()
        registry.discover()

        # Both built-in modules should be registered
        assert registry.get_module("filesystem") is not None
        assert registry.get_module("shell") is not None

        # Check filesystem tools are present
        fs_tools = {
            "filesystem:read_file",
            "filesystem:write_file",
            "filesystem:list_directory",
            "filesystem:delete_file",
            "filesystem:move_file",
        }
        assert fs_tools.issubset({t.name for t in registry.get_all_tool_descriptors()})

        # Check shell tools are present
        shell_tools = {"shell:execute"}
        assert shell_tools.issubset(
            {t.name for t in registry.get_all_tool_descriptors()}
        )
