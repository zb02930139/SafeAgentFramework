"""Top-level SafeAgent runtime wiring."""

from __future__ import annotations

from pathlib import Path

from safe_agent.core.audit import AuditLogger
from safe_agent.core.dispatcher import ToolDispatcher
from safe_agent.core.event_loop import EventLoop
from safe_agent.core.gateway import Gateway
from safe_agent.core.llm import LLMClient
from safe_agent.core.session import SessionManager
from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.policy import PolicyStore
from safe_agent.modules.base import BaseModule
from safe_agent.modules.registry import ModuleRegistry


class Agent:
    """Convenience runtime that wires policies, modules, sessions, and chat."""

    def __init__(
        self,
        policy_dir: Path,
        llm_client: LLMClient,
        modules: list[BaseModule] | None = None,
        audit_log_path: Path | None = None,
        max_turns: int = 10,
    ) -> None:
        """Initialise the complete SafeAgent runtime stack.

        Args:
            policy_dir: Directory containing policy files to load.
            llm_client: LLM client for generating responses.
            modules: List of modules to register. If None, auto-discovers built-in
                modules (filesystem, shell) via entry points. Discovered modules
                use default configs (e.g., FilesystemModule roots at cwd()).
            audit_log_path: Path for audit log file. Defaults to "audit.jsonl".
            max_turns: Maximum turns per session before eviction.
        """
        store = PolicyStore()
        store.load(policy_dir)
        store.freeze()

        evaluator = PolicyEvaluator(store)

        registry = ModuleRegistry()
        if modules is None:
            registry.discover()
        else:
            for module in modules:
                registry.register(module)

        audit_logger = AuditLogger(audit_log_path or Path("audit.jsonl"))
        dispatcher = ToolDispatcher(registry, evaluator, audit_logger)
        session_manager = SessionManager()
        event_loop = EventLoop(dispatcher, llm_client, registry, max_turns)
        gateway = Gateway(session_manager, event_loop)

        self.policy_store = store
        self.policy_evaluator = evaluator
        self.registry = registry
        self.audit_logger = audit_logger
        self.dispatcher = dispatcher
        self.session_manager = session_manager
        self.event_loop = event_loop
        self.gateway = gateway

    async def chat(
        self, message: str, session_id: str | None = None
    ) -> tuple[str, str]:
        """Process a chat message via the gateway.

        Returns:
            A ``(response, session_id)`` tuple. Pass ``session_id`` back on the
            next call to continue the same conversation.
        """
        return await self.gateway.submit(message, session_id)
