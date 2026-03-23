"""Unit tests for Gateway."""

from __future__ import annotations

import pytest

from safe_agent.core.event_loop import EventLoop
from safe_agent.core.gateway import Gateway
from safe_agent.core.session import Session, SessionManager


class MockEventLoop:
    """Test double for EventLoop."""
    
    def __init__(self) -> None:
        self.calls: list[tuple[Session, str]] = []
        self.responses: list[str] = []
    
    async def process_turn(self, session: Session, message: str) -> str:
        self.calls.append((session, message))
        if not self.responses:
            return "default response"
        return self.responses.pop(0)


class MockSessionManager:
    """Test double for SessionManager."""
    
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.create_calls = 0
    
    def create(self) -> Session:
        session = Session(id=f"session-{self.create_calls}")
        self.sessions[session.id] = session
        self.create_calls += 1
        return session
    
    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


@pytest.mark.asyncio
async def test_gateway_submit_new_session() -> None:
    """Test submit() with a new session ID (auto-creation)."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)
    
    response, session_id = await gateway.submit("test message")
    
    assert response == "default response"
    assert session_id is not None
    assert session_id in session_manager.sessions
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][0].id == session_id
    assert event_loop.calls[0][1] == "test message"


@pytest.mark.asyncio
async def test_gateway_submit_existing_session() -> None:
    """Test submit() with an existing session ID."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)
    
    # Create a session first
    session = session_manager.create()
    
    response, session_id = await gateway.submit("test message", session.id)
    
    assert response == "default response"
    assert session_id == session.id
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][0].id == session.id
    assert event_loop.calls[0][1] == "test message"


@pytest.mark.asyncio
async def test_gateway_submit_unknown_session_raises_key_error() -> None:
    """Test submit() with an invalid/unknown session ID raises KeyError."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)
    
    with pytest.raises(KeyError, match="nonexistent"):
        await gateway.submit("test message", "nonexistent")


@pytest.mark.asyncio
async def test_gateway_submit_empty_message() -> None:
    """Test submit() with an empty message."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)
    
    response, session_id = await gateway.submit("")
    
    assert response == "default response"
    assert session_id is not None
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][1] == ""


@pytest.mark.asyncio
async def test_gateway_submit_with_custom_response() -> None:
    """Test submit() with a custom response from event loop."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    event_loop.responses = ["custom response"]
    gateway = Gateway(session_manager, event_loop)
    
    response, session_id = await gateway.submit("test message")
    
    assert response == "custom response"
    assert session_id is not None
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][1] == "test message"


@pytest.mark.asyncio
async def test_gateway_submit_multiple_sessions() -> None:
    """Test submit() with multiple sessions."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)
    
    # First session
    response1, session_id1 = await gateway.submit("message 1")
    # Second session
    response2, session_id2 = await gateway.submit("message 2")
    
    assert session_id1 != session_id2
    assert len(session_manager.sessions) == 2
    assert len(event_loop.calls) == 2
    assert event_loop.calls[0][0].id == session_id1
    assert event_loop.calls[0][1] == "message 1"
    assert event_loop.calls[1][0].id == session_id2
    assert event_loop.calls[1][1] == "message 2"