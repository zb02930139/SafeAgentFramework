"""Gateway entrypoint for session-aware agent chat."""

from __future__ import annotations

from safe_agent.core.event_loop import EventLoop
from safe_agent.core.session import SessionManager


class Gateway:
    """Routes incoming chat messages to the correct session event loop."""

    def __init__(
        self,
        session_manager: SessionManager,
        event_loop: EventLoop,
    ) -> None:
        """Initialise the gateway dependencies."""
        self._session_manager = session_manager
        self._event_loop = event_loop

    async def submit(self, message: str, session_id: str | None = None) -> str:
        """Submit a user message to a session and return the assistant response.

        Args:
            message: The user message to process.
            session_id: Existing session identifier, or ``None`` to create one.

        Returns:
            The final response text returned by the event loop.

        Raises:
            KeyError: If *session_id* is provided but no tracked session exists.
        """
        session = self._session_manager.create()
        if session_id is not None:
            existing_session = self._session_manager.get(session_id)
            if existing_session is None:
                raise KeyError(session_id)
            session = existing_session

        return await self._event_loop.process_turn(session, message)
