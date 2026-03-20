"""Module registry for discovering and dispatching SafeAgent modules."""

from importlib.metadata import entry_points

from safe_agent.modules.base import BaseModule, ModuleDescriptor, ToolDescriptor


class ModuleRegistry:
    """Registry for SafeAgent modules supporting discovery and dispatch.

    Modules can be registered manually via ``register()`` or discovered
    automatically from installed packages via ``discover()``.

    Collision rules:
    - Namespace collisions between *different* module instances raise ``ValueError``.
    - Tool name collisions across any modules raise ``ValueError`` (no silent shadowing).
    - Registering the *same* instance twice is idempotent and a no-op.

    ``discover()`` is call-once; subsequent calls raise ``RuntimeError``.

    Example:
        >>> registry = ModuleRegistry()
        >>> registry.register(my_module)
        >>> result = registry.get_tool("my_namespace:my_tool")
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
                    f"by {existing_module!r}. Tool names must be unique across all modules."
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

        Raises:
            RuntimeError: If ``discover()`` has already been called on this registry.
            TypeError: If an entry point loads a class that is not a subclass of
                ``BaseModule``.
            ValueError: If namespace or tool name collisions occur during registration.
        """
        if self._discovered:
            raise RuntimeError(
                "discover() has already been called on this registry. "
                "Create a new ModuleRegistry instance to re-discover."
            )
        self._discovered = True

        eps = entry_points(group="safe_agent.modules")
        for ep in eps:
            module_class = ep.load()
            if not (isinstance(module_class, type) and issubclass(module_class, BaseModule)):
                raise TypeError(
                    f"Entry point '{ep.name}' ({ep.value}) loaded {module_class!r}, "
                    f"which is not a subclass of BaseModule. "
                    f"Only trusted BaseModule subclasses may be registered."
                )
            instance: BaseModule = module_class()
            self.register(instance)

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
        """Return all registered module instances.

        Returns:
            A list of all ``BaseModule`` instances in registration order.
        """
        return list(self._namespace_map.values())

    def get_all_tool_descriptors(self) -> list[ToolDescriptor]:
        """Return descriptors for every registered tool.

        Returns:
            A list of ``ToolDescriptor`` objects across all registered modules.
        """
        return [td for _, td in self._tool_map.values()]
