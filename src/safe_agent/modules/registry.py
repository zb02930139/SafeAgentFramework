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

"""Module registry for discovering and managing SafeAgent modules."""

import logging
from importlib.metadata import entry_points

from safe_agent.modules.base import BaseModule, ToolDescriptor

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Registry for SafeAgent modules supporting discovery and lookup.

    Modules can be registered manually via ``register()`` or discovered
    automatically from installed packages via ``discover()``.

    Collision rules:

    - Namespace collisions between *different* module instances raise
      ``ValueError``.
    - Tool name collisions across modules raise ``ValueError`` (no silent
      shadowing).
    - Registering the *same* instance twice is idempotent and a no-op.

    ``discover()`` may be retried after a failure. After a successful call,
    subsequent calls raise ``RuntimeError``.

    This class is not thread-safe. External synchronization is required for
    concurrent access.

    Security note: ``discover()`` loads and instantiates code from installed
    packages that declare a ``safe_agent.modules`` entry point. This is only
    safe when the Python environment is fully controlled and no untrusted
    packages are installed. The ``issubclass(…, BaseModule)`` guard is a
    structural sanity check, not a security boundary. Do not use ``discover()``
    in environments where third-party packages cannot be fully trusted.

    Security note: ``dispatch()`` has been removed. Tool execution must go
    through ``ToolDispatcher``, which enforces policy evaluation and audit
    logging. Directly calling ``module.execute()`` bypasses these safeguards
    and is a security violation.

    Example::

        registry = ModuleRegistry()
        registry.register(my_module)
        tool = registry.get_tool("my_namespace:my_tool")
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._tool_map: dict[str, tuple[BaseModule, ToolDescriptor]] = {}
        self._namespace_map: dict[str, BaseModule] = {}
        self._discovered: bool = False

    def register(self, module: BaseModule) -> None:
        """Register a module instance into the registry.

        Args:
            module: A concrete ``BaseModule`` instance to register.

        Raises:
            ValueError: If the module's namespace is already registered by a
                different module instance, or if any of the module's tool names
                collide with an already-registered tool.

        Note:
            ``register()`` is atomic with respect to its own state mutations:
            all validation is performed before any writes, so a ``ValueError``
            leaves the registry unchanged. However, if ``describe()`` itself
            raises, the registry is unaffected. Callers should discard the
            registry after any unexpected exception to be safe.
        """
        descriptor = module.describe()
        namespace = descriptor.namespace
        # Snapshot tools once so both validation and write passes use the same
        # list, guarding against mutation between the two iterations.
        tools = list(descriptor.tools)

        # Namespace collision check.
        if namespace in self._namespace_map:
            existing = self._namespace_map[namespace]
            if existing is not module:
                raise ValueError(
                    f"Namespace collision: '{namespace}' is already registered "
                    f"by {existing!r}."
                )
            # Same module registered twice — idempotent, just return.
            return

        # Pre-validate tool name uniqueness before touching any state.
        # This makes register() atomic: either all tools are committed or none.
        seen: set[str] = set()
        for tool in tools:
            if tool.name in seen:
                raise ValueError(
                    f"Duplicate tool name '{tool.name}' within module '{namespace}'. "
                    f"Tool names must be unique within a module."
                )
            if tool.name in self._tool_map:
                existing_module, _ = self._tool_map[tool.name]
                raise ValueError(
                    f"Tool name collision: '{tool.name}' is already registered "
                    f"by {existing_module!r}. "
                    f"Tool names must be unique across all modules."
                )
            seen.add(tool.name)

        # All validation passed — commit namespace and tools together.
        self._namespace_map[namespace] = module
        for tool in tools:
            self._tool_map[tool.name] = (module, tool)

    def discover(self) -> None:
        """Discover and register modules from installed package entry points.

        Scans the ``safe_agent.modules`` entry-point group, instantiates each
        advertised class, and calls ``register()`` on it.

        Only call this method once. Subsequent calls raise ``RuntimeError`` to
        prevent double-instantiation of entry point classes.

        Each loaded entry point is logged at INFO level (name, value) to provide
        an audit trail of what was loaded and from where.

        .. warning::
            This method loads and executes third-party code. Only use it in
            environments where all installed packages are fully trusted. See
            the class-level security note for details.

        Raises:
            RuntimeError: If ``discover()`` has already been called on this
                registry.
            TypeError: If an entry point loads a value that is not a subclass
                of ``BaseModule``.
            ValueError: If namespace or tool name collisions occur during
                registration.

        Note:
            ``discover()`` is atomic: all validation and instantiation run
            against staging dicts before any live state is modified. If an
            error occurs at any point, the registry is left in its
            pre-discover state — no partial registration occurs. Fixes
            issue #11.
        """
        if self._discovered:
            raise RuntimeError(
                "discover() has already been called on this registry. "
                "Create a new ModuleRegistry instance to re-discover."
            )

        eps = entry_points(group="safe_agent.modules")

        # Use staging dicts so that discover() is all-or-nothing. If anything
        # fails mid-loop the live maps are never touched and the registry
        # remains in its pre-discover state.
        staging_namespace_map: dict[str, BaseModule] = dict(self._namespace_map)
        staging_tool_map: dict[str, tuple[BaseModule, ToolDescriptor]] = dict(
            self._tool_map
        )

        import inspect

        for ep in eps:
            module_class = ep.load()
            logger.info(
                "safe_agent.modules: loading entry point '%s' (%s)",
                ep.name,
                ep.value,
            )
            if not (
                isinstance(module_class, type) and issubclass(module_class, BaseModule)
            ):
                raise TypeError(
                    f"Entry point '{ep.name}' ({ep.value}) loaded "
                    f"{module_class!r}, which is not a subclass of "
                    f"BaseModule. Only trusted BaseModule subclasses "
                    f"may be registered."
                )

            # Skip modules that require constructor arguments (can't auto-instantiate)
            sig = inspect.signature(module_class.__init__)
            params = list(sig.parameters.values())
            var_kinds = (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            required = [
                p
                for p in params
                if (
                    p.default is inspect.Parameter.empty
                    and p.name != "self"
                    and p.kind not in var_kinds
                )
            ]
            if required:
                logger.info(
                    "safe_agent.modules: skipping '%s' (requires constructor args: %s)",
                    ep.name,
                    [p.name for p in required],
                )
                continue

            instance: BaseModule = module_class()
            descriptor = instance.describe()
            namespace = descriptor.namespace
            tools = list(descriptor.tools)

            # Namespace collision check against staging state.
            if namespace in staging_namespace_map:
                existing = staging_namespace_map[namespace]
                raise ValueError(
                    f"Namespace collision: '{namespace}' is already "
                    f"registered by {existing!r}."
                )

            # Tool collision check: intra-module duplicates and cross-module
            # collisions against staging state.
            seen_in_module: set[str] = set()
            for tool in tools:
                if tool.name in seen_in_module:
                    raise ValueError(
                        f"Duplicate tool name '{tool.name}' within module "
                        f"'{namespace}'. Tool names must be unique within a module."
                    )
                if tool.name in staging_tool_map:
                    existing_module, _ = staging_tool_map[tool.name]
                    raise ValueError(
                        f"Tool name collision: '{tool.name}' is already "
                        f"registered by {existing_module!r}. "
                        f"Tool names must be unique across all modules."
                    )
                seen_in_module.add(tool.name)

            # All checks passed — write to staging only. If anything below
            # raises, these local dicts are discarded and live state remains
            # untouched.
            staging_namespace_map[namespace] = instance
            for tool in tools:
                staging_tool_map[tool.name] = (instance, tool)

            logger.info(
                "safe_agent.modules: registered '%s' from entry point '%s'",
                instance,
                ep.name,
            )

        # All entry points processed successfully — atomic swap.
        self._namespace_map = staging_namespace_map
        self._tool_map = staging_tool_map
        self._discovered = True

    def get_tool(self, tool_name: str) -> tuple[BaseModule, ToolDescriptor] | None:
        """Look up a tool by its fully-qualified name.

        Args:
            tool_name: The tool name as registered (e.g. ``"fs:ReadFile"``).

        Returns:
            A ``(BaseModule, ToolDescriptor)`` tuple if found, else ``None``.
        """
        return self._tool_map.get(tool_name)

    def get_module(self, namespace: str) -> BaseModule | None:
        """Look up a registered module by its namespace.

        Args:
            namespace: The namespace string (e.g. ``"fs"``).

        Returns:
            The ``BaseModule`` instance if found, else ``None``.
        """
        return self._namespace_map.get(namespace)

    def get_all_modules(self) -> list[BaseModule]:
        """Return all registered module instances in insertion order.

        Note: ordering reflects registration order and is stable within a
        single registry lifetime, but should not be relied upon across
        restarts or re-registration sequences.

        Returns:
            A list of all ``BaseModule`` instances.
        """
        return list(self._namespace_map.values())

    def get_all_tool_descriptors(self) -> list[ToolDescriptor]:
        """Return descriptors for every registered tool.

        Returns:
            A list of ``ToolDescriptor`` objects across all registered modules.
        """
        return [td for _, td in self._tool_map.values()]
