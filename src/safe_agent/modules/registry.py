"""Module registry for discovering and dispatching SafeAgent modules."""

import logging
from importlib.metadata import entry_points

from safe_agent.modules.base import BaseModule, ToolDescriptor, ToolResult

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Registry for SafeAgent modules supporting discovery and dispatch.

    Modules can be registered manually via ``register()`` or discovered
    automatically from installed packages via ``discover()``.

    Collision rules:

    - Namespace collisions between *different* module instances raise
      ``ValueError``.
    - Tool name collisions across modules raise ``ValueError`` (no silent
      shadowing).
    - Registering the *same* instance twice is idempotent and a no-op.

    ``discover()`` is call-once; subsequent calls raise ``RuntimeError``.

    Security note: ``discover()`` loads and instantiates code from installed
    packages that declare a ``safe_agent.modules`` entry point. This is only
    safe when the Python environment is fully controlled and no untrusted
    packages are installed. The ``issubclass(…, BaseModule)`` guard is a
    structural sanity check, not a security boundary. Do not use ``discover()``
    in environments where third-party packages cannot be fully trusted.

    Example::

        registry = ModuleRegistry()
        registry.register(my_module)
        result = registry.dispatch("my_namespace:my_tool", params={})
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
        """
        descriptor = module.describe()
        namespace = descriptor.namespace

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

        # Tool name collision check — must happen before any mutations.
        for tool in descriptor.tools:
            if tool.name in self._tool_map:
                existing_module, _ = self._tool_map[tool.name]
                raise ValueError(
                    f"Tool name collision: '{tool.name}' is already registered "
                    f"by {existing_module!r}. "
                    f"Tool names must be unique across all modules."
                )

        self._namespace_map[namespace] = module
        for tool in descriptor.tools:
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
        """
        if self._discovered:
            raise RuntimeError(
                "discover() has already been called on this registry. "
                "Create a new ModuleRegistry instance to re-discover."
            )

        eps = entry_points(group="safe_agent.modules")
        try:
            for ep in eps:
                module_class = ep.load()
                logger.info(
                    "safe_agent.modules: loading entry point '%s' (%s)",
                    ep.name,
                    ep.value,
                )
                if not (
                    isinstance(module_class, type)
                    and issubclass(module_class, BaseModule)
                ):
                    raise TypeError(
                        f"Entry point '{ep.name}' ({ep.value}) loaded "
                        f"{module_class!r}, which is not a subclass of "
                        f"BaseModule. Only trusted BaseModule subclasses "
                        f"may be registered."
                    )
                instance: BaseModule = module_class()
                self.register(instance)
                logger.info(
                    "safe_agent.modules: registered '%s' from entry point '%s'",
                    instance,
                    ep.name,
                )
        except Exception:
            # Discovery failed — do not mark as completed so callers can
            # diagnose and retry with a fresh registry.
            raise
        else:
            # Only mark discovered after the loop completes successfully.
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

    async def dispatch(
        self,
        tool_name: str,
        params: dict,
    ) -> ToolResult:
        """Look up and execute a tool by name, validating it exists first.

        This is the preferred dispatch path. It ensures tool lookup goes
        through the registry (the enforced choke point) rather than callers
        invoking ``module.execute()`` directly.

        Args:
            tool_name: The fully-qualified tool name (e.g. ``"fs:ReadFile"``).
            params: Input parameters for the tool.

        Returns:
            A ``ToolResult`` from the module's ``execute()`` method.

        Raises:
            KeyError: If ``tool_name`` is not registered in this registry.
        """
        result = self.get_tool(tool_name)
        if result is None:
            raise KeyError(f"Tool '{tool_name}' is not registered in this registry.")
        module, _descriptor = result
        return await module.execute(tool_name, params)
