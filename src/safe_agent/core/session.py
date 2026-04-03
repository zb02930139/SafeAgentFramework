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

"""Session models and in-memory session tracking with TTL and eviction."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Session(BaseModel):
    """Represents a single agent conversation session.

    Attributes:
        id: Unique session identifier.
        messages: Conversation history for the session.
        max_messages: Maximum messages to retain (oldest trimmed first).
        metadata: Arbitrary session-scoped metadata.
        created_at: Timestamp when the session was created (datetime in UTC).
        last_accessed: Monotonic timestamp when the session was last accessed (float).
            This is a monotonic clock value (seconds since an arbitrary point), not a
            wall-clock datetime. Use time.monotonic() for comparisons.

    Thread safety
    -------------
    All mutations of ``messages`` should be guarded by ``_message_lock``, a
    per-session ``threading.RLock``.  Both :class:`SessionManager` (via
    :meth:`~SessionManager.add_message`) and :class:`EventLoop` (via
    :meth:`~EventLoop.process_turn`) acquire this lock before reading or
    writing ``messages``, ensuring correct cross-thread behaviour.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[dict] = Field(default_factory=list)
    max_messages: int = Field(default=1000, gt=0)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: float = Field(default_factory=time.monotonic)

    #: Per-session reentrant lock protecting ``messages``.
    #: Excluded from serialisation; created fresh for each Session instance.
    _message_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)


class SessionManager:
    """Manages active in-memory sessions with TTL and LRU eviction.

    This implementation provides:
    - Configurable session TTL (time-to-live) with automatic cleanup of idle sessions
    - Maximum session count with LRU (least recently used) eviction
    - Maximum messages per session with oldest-first trimming
    - Thread-safe operations using a reentrant lock
    - Lazy cleanup on access (no background thread required)

    Attributes:
        session_ttl: Maximum idle time before a session is evicted (default 1 hour).
        max_sessions: Maximum number of concurrent sessions (default 1000).
        max_messages: Maximum messages per session before trimming (default 1000).

    Note:
        Session.last_accessed is a float (monotonic time) for efficient TTL
        comparison, not a datetime. Use time.monotonic() values for comparisons.
    """

    def __init__(
        self,
        *,
        session_ttl: timedelta | float | None = None,
        max_sessions: int | None = None,
        max_messages: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize session manager with optional limits.

        Args:
            session_ttl: Maximum idle time before eviction. Can be a timedelta
                or float (interpreted as seconds). Default is 1 hour (3600 seconds).
            max_sessions: Maximum concurrent sessions. Default is 1000.
            max_messages: Maximum messages per session. Default is 1000.
            clock: Optional clock function returning monotonic time as float.
                Defaults to time.monotonic. Used for testing with fake clocks.
        """
        # Store clock (for testing)
        self._clock = clock or time.monotonic

        # Normalize TTL to float seconds for comparison
        if session_ttl is None:
            self._session_ttl_seconds = 3600.0  # 1 hour default
        elif isinstance(session_ttl, timedelta):
            self._session_ttl_seconds = session_ttl.total_seconds()
        else:
            self._session_ttl_seconds = float(session_ttl)

        # Also store as timedelta for backward compatibility
        self.session_ttl = timedelta(seconds=self._session_ttl_seconds)

        # Validate TTL is non-negative
        if self._session_ttl_seconds < 0:
            raise ValueError("session_ttl must be non-negative")

        # Apply defaults
        self.max_sessions = max_sessions if max_sessions is not None else 1000
        self.max_messages = max_messages if max_messages is not None else 1000

        # Validate limits are positive
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if self.max_messages <= 0:
            raise ValueError("max_messages must be positive")

        # OrderedDict maintains access order for LRU eviction
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.RLock()

        # Optional callbacks for eviction events
        self._on_evict: Callable[[Session], None] | None = None

    def create(self) -> Session:
        """Create and track a new session.

        If adding this session would exceed max_sessions, the least recently
        used session is evicted first.

        Returns:
            The newly created Session instance.
        """
        with self._lock:
            self._cleanup_expired()

            # Evict LRU if at capacity
            while len(self._sessions) >= self.max_sessions:
                self._evict_lru()

            session = Session(max_messages=self.max_messages)
            # Use the manager's clock for last_accessed (important for testing)
            session.last_accessed = self._clock()
            self._sessions[session.id] = session
            # Move to end (most recently used)
            self._sessions.move_to_end(session.id)
            return session

    def get(self, session_id: str) -> Session | None:
        """Return a tracked session by ID, if present and not expired.

        Updates the session's last_accessed timestamp and moves it to the
        most recently used position.

        Args:
            session_id: The unique identifier of the session.

        Returns:
            The Session instance if found and valid, None otherwise.
        """
        with self._lock:
            self._cleanup_expired()

            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Update access time and move to end (most recently used)
            session.last_accessed = self._clock()
            self._sessions.move_to_end(session_id)
            return session

    def close(self, session_id: str) -> None:
        """Remove a session from active tracking if present.

        This triggers the eviction callback (if set) to release any associated
        resources, such as EventLoop session locks.

        Note: This triggers lazy cleanup of expired sessions as a side effect.

        Args:
            session_id: The unique identifier of the session to close.
        """
        with self._lock:
            self._cleanup_expired()
            self._evict_session(session_id, reason="close")

    def list_active(self) -> list[str]:
        """Return active session IDs in LRU order (oldest first).

        Returns:
            List of session IDs currently being tracked.
        """
        with self._lock:
            self._cleanup_expired()
            return list(self._sessions.keys())

    def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
        *,
        trim: bool = True,
    ) -> Session | None:
        """Add a message to a session with optional trimming.

        This method is **thread-safe**: it may be called from any OS thread
        concurrently with :meth:`EventLoop.process_turn`.  The session-level
        ``threading.RLock`` (``Session._message_lock``) ensures that message
        mutations are serialised with respect to the event-loop thread as well
        as other threads calling this method.

        Args:
            session_id: The unique identifier of the session.
            message: The message dict to append.
            trim: Whether to trim messages to max_messages (default True).

        Returns:
            The Session instance if found, None otherwise.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            # Update access time and LRU position while holding the manager lock.
            session.last_accessed = self._clock()
            self._sessions.move_to_end(session_id)

        # Protect the actual message mutation with the per-session lock so that
        # concurrent EventLoop turns (which also acquire session._message_lock)
        # cannot interleave with this write.
        with session._message_lock:
            session.messages.append(message)

            if trim and len(session.messages) > session.max_messages:
                # Trim oldest messages in-place (keep most recent max_messages)
                excess = len(session.messages) - session.max_messages
                del session.messages[:excess]

        return session

    def count(self) -> int:
        """Return the current number of active sessions.

        Note: This triggers lazy cleanup of expired sessions first.

        Returns:
            Number of sessions currently being tracked.
        """
        with self._lock:
            self._cleanup_expired()
            return len(self._sessions)

    def set_eviction_callback(self, callback: Callable[[Session], None] | None) -> None:
        """Set a callback to be invoked when a session is evicted.

        Args:
            callback: A function that receives the evicted Session, or None to disable.
        """
        with self._lock:
            self._on_evict = callback

    def _cleanup_expired(self) -> int:
        """Remove all sessions that have exceeded their TTL.

        This is called lazily on access operations.

        Returns:
            Number of sessions evicted.
        """
        now = self._clock()
        expired_ids = []

        for session_id, session in self._sessions.items():
            idle_time = now - session.last_accessed
            if idle_time > self._session_ttl_seconds:
                expired_ids.append(session_id)

        for session_id in expired_ids:
            self._evict_session(session_id, reason="ttl")

        return len(expired_ids)

    def _evict_lru(self) -> Session | None:
        """Evict the least recently used session.

        Returns:
            The evicted Session, or None if no sessions exist.
        """
        if not self._sessions:
            return None

        # OrderedDict: first item is LRU
        lru_id = next(iter(self._sessions))
        return self._evict_session(lru_id, reason="lru")

    def _evict_session(
        self, session_id: str, reason: str = "unknown"
    ) -> Session | None:
        """Evict a specific session by ID.

        Args:
            session_id: The ID of the session to evict.
            reason: Reason for eviction (for logging/callback purposes).

        Returns:
            The evicted Session, or None if not found.
        """
        session = self._sessions.pop(session_id, None)
        if session is not None and self._on_evict is not None:
            # Suppress exceptions from user-provided callback
            with contextlib.suppress(Exception):
                self._on_evict(session)
        return session
