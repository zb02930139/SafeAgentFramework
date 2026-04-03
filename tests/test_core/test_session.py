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


"""Tests for Session and SessionManager with TTL and eviction."""

import threading
from datetime import timedelta
from uuid import UUID

from safe_agent.core import SessionManager


class FakeClock:
    """A fake clock for deterministic testing of TTL behavior.

    Implements the clock protocol (Callable[[], float]) used by SessionManager.
    Time is controlled manually via advance() instead of relying on real time.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        """Advance the fake clock by the given number of seconds."""
        self._time += seconds


class TestSessionBasics:
    """Basic session creation and retrieval tests."""

    def test_create_session(self) -> None:
        manager = SessionManager()

        session_one = manager.create()
        session_two = manager.create()

        assert session_one.id != session_two.id
        assert UUID(session_one.id)
        assert UUID(session_two.id)

    def test_get_session(self) -> None:
        manager = SessionManager()
        session = manager.create()

        assert manager.get(session.id) is session

    def test_close_session(self) -> None:
        manager = SessionManager()
        session = manager.create()

        manager.close(session.id)

        assert manager.get(session.id) is None

    def test_list_active(self) -> None:
        manager = SessionManager()
        session_one = manager.create()
        session_two = manager.create()

        assert manager.list_active() == [session_one.id, session_two.id]

    def test_session_isolation(self) -> None:
        manager = SessionManager()
        session_one = manager.create()
        session_two = manager.create()

        session_one.messages.append({"role": "user", "content": "one"})
        session_two.messages.append({"role": "user", "content": "two"})

        assert session_one.messages == [{"role": "user", "content": "one"}]
        assert session_two.messages == [{"role": "user", "content": "two"}]

    def test_count(self) -> None:
        manager = SessionManager()
        assert manager.count() == 0

        manager.create()
        assert manager.count() == 1

        manager.create()
        assert manager.count() == 2


class TestSessionTTL:
    """Tests for session TTL (time-to-live) functionality."""

    def test_default_ttl_is_one_hour(self) -> None:
        manager = SessionManager()
        assert manager.session_ttl == timedelta(seconds=3600)

    def test_custom_ttl_timedelta(self) -> None:
        manager = SessionManager(session_ttl=timedelta(minutes=30))
        assert manager.session_ttl == timedelta(minutes=30)

    def test_custom_ttl_seconds(self) -> None:
        manager = SessionManager(session_ttl=1800)
        assert manager.session_ttl == timedelta(seconds=1800)

    def test_expired_session_not_returned(self) -> None:
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=0.1, clock=fake_time)  # 100ms TTL

        session = manager.create()
        session_id = session.id

        # Session should be present initially
        assert manager.get(session_id) is not None

        # Advance time past TTL
        fake_time.advance(0.15)  # 150ms

        # Expired session should be cleaned up on access
        assert manager.get(session_id) is None

    def test_expired_sessions_removed_on_list_active(self) -> None:
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=0.1, clock=fake_time)  # 100ms TTL

        session = manager.create()

        # Session should be in list initially
        assert session.id in manager.list_active()

        # Advance time past TTL
        fake_time.advance(0.15)  # 150ms

        # Should be cleaned up on list_active
        assert session.id not in manager.list_active()

    def test_accessed_session_not_expired(self) -> None:
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=0.2, clock=fake_time)  # 200ms TTL

        session = manager.create()
        session_id = session.id

        # Keep accessing the session to keep it alive
        for _ in range(3):
            fake_time.advance(0.1)  # Advance 100ms each time
            assert manager.get(session_id) is not None

        # Should still exist after 300ms total because we kept accessing
        assert manager.get(session_id) is not None

    def test_last_accessed_updated_on_get(self) -> None:
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=3600, clock=fake_time)

        session = manager.create()
        initial_access = session.last_accessed

        # Advance time
        fake_time.advance(0.01)

        manager.get(session.id)
        assert session.last_accessed > initial_access


class TestMaxSessions:
    """Tests for maximum session count and LRU eviction."""

    def test_max_sessions_default(self) -> None:
        manager = SessionManager()
        assert manager.max_sessions == 1000

    def test_custom_max_sessions(self) -> None:
        manager = SessionManager(max_sessions=10)
        assert manager.max_sessions == 10

    def test_lru_eviction_on_create(self) -> None:
        manager = SessionManager(max_sessions=2, session_ttl=3600)

        session1 = manager.create()
        session2 = manager.create()

        # Both should exist
        assert manager.get(session1.id) is not None
        assert manager.get(session2.id) is not None

        # Creating third should evict session1 (LRU)
        session3 = manager.create()

        assert manager.get(session1.id) is None  # Evicted
        assert manager.get(session2.id) is not None
        assert manager.get(session3.id) is not None

    def test_access_updates_lru_order(self) -> None:
        manager = SessionManager(max_sessions=2, session_ttl=3600)

        session1 = manager.create()
        session2 = manager.create()

        # Access session1 to make it most recently used
        manager.get(session1.id)

        # Now session2 is LRU
        session3 = manager.create()

        # session2 should be evicted (was LRU after we accessed session1)
        assert manager.get(session1.id) is not None
        assert manager.get(session2.id) is None  # Evicted
        assert manager.get(session3.id) is not None

    def test_count_respects_max(self) -> None:
        manager = SessionManager(max_sessions=3)

        for _ in range(5):
            manager.create()

        # Should still be at max_sessions
        assert manager.count() == 3


class TestMaxMessages:
    """Tests for maximum messages per session and trimming."""

    def test_max_messages_default(self) -> None:
        manager = SessionManager()
        assert manager.max_messages == 1000

    def test_custom_max_messages(self) -> None:
        manager = SessionManager(max_messages=50)
        assert manager.max_messages == 50

    def test_add_message(self) -> None:
        manager = SessionManager()
        session = manager.create()

        manager.add_message(session.id, {"role": "user", "content": "hello"})

        assert len(session.messages) == 1
        assert session.messages[0] == {"role": "user", "content": "hello"}

    def test_message_trimming(self) -> None:
        manager = SessionManager(max_messages=3)
        session = manager.create()

        # Add 5 messages
        for i in range(5):
            manager.add_message(session.id, {"role": "user", "content": str(i)})

        # Should keep only last 3
        assert len(session.messages) == 3
        assert session.messages[0]["content"] == "2"
        assert session.messages[1]["content"] == "3"
        assert session.messages[2]["content"] == "4"

    def test_add_message_no_trim(self) -> None:
        manager = SessionManager(max_messages=3)
        session = manager.create()

        # Add 5 messages without trimming
        for i in range(5):
            manager.add_message(
                session.id, {"role": "user", "content": str(i)}, trim=False
            )

        # Should keep all
        assert len(session.messages) == 5

    def test_add_message_returns_none_for_unknown_session(self) -> None:
        manager = SessionManager()

        result = manager.add_message("nonexistent", {"role": "user", "content": "test"})

        assert result is None


class TestEvictionCallback:
    """Tests for eviction callback functionality."""

    def test_eviction_callback_on_ttl_expiry(self) -> None:
        evicted = []
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=0.1, clock=fake_time)
        manager.set_eviction_callback(lambda s: evicted.append(s.id))

        session = manager.create()

        # Advance time past TTL
        fake_time.advance(0.15)  # 150ms

        # Trigger cleanup via get
        manager.get(session.id)

        assert session.id in evicted

    def test_eviction_callback_on_lru_eviction(self) -> None:
        evicted = []
        manager = SessionManager(max_sessions=1)
        manager.set_eviction_callback(lambda s: evicted.append(s.id))

        session1 = manager.create()
        _ = manager.create()  # Evicts session1

        assert session1.id in evicted

    def test_callback_none_disables(self) -> None:
        evicted = []
        manager = SessionManager(max_sessions=1)
        manager.set_eviction_callback(lambda s: evicted.append(s.id))
        manager.set_eviction_callback(None)  # Disable

        manager.create()
        manager.create()

        # Callback was disabled, should not have been called
        assert len(evicted) == 0

    def test_callback_exception_does_not_break_eviction(self) -> None:
        def bad_callback(s: object) -> None:
            raise RuntimeError("Callback error")

        manager = SessionManager(max_sessions=1, session_ttl=3600)
        manager.set_eviction_callback(bad_callback)

        session1 = manager.create()
        _ = manager.create()

        # Session1 should still be evicted despite callback error
        assert manager.get(session1.id) is None

    def test_eviction_callback_on_close(self) -> None:
        """close() should trigger the eviction callback to release resources."""
        evicted = []
        manager = SessionManager()
        manager.set_eviction_callback(lambda s: evicted.append(s.id))

        session = manager.create()
        manager.close(session.id)

        assert session.id in evicted


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_creates(self) -> None:
        manager = SessionManager(max_sessions=1000)
        session_ids_lock = threading.Lock()
        session_ids: list[str] = []
        errors = []

        def create_sessions() -> None:
            try:
                for _ in range(100):
                    session = manager.create()
                    with session_ids_lock:
                        session_ids.append(session.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager.count() <= 1000  # Should respect max_sessions

    def test_concurrent_get_and_close(self) -> None:
        manager = SessionManager()
        sessions = [manager.create() for _ in range(100)]
        errors = []

        def get_and_close(start: int) -> None:
            try:
                for i in range(start, min(start + 50, len(sessions))):
                    manager.get(sessions[i].id)
                    if i % 2 == 0:
                        manager.close(sessions[i].id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=get_and_close, args=(i * 10,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_add_message(self) -> None:
        """Concurrent threads calling add_message must not corrupt messages.

        This test exercises the shared ``threading.RLock`` path that protects
        ``session.messages`` from concurrent OS-thread access — the scenario
        that previously relied on the incorrect asyncio.Lock approach.
        """
        manager = SessionManager(max_messages=1000)
        session = manager.create()
        errors: list[Exception] = []

        def add_messages() -> None:
            try:
                for i in range(100):
                    manager.add_message(session.id, {"role": "user", "content": str(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 8 threads * 100 messages each = 800 total, all within max_messages=1000
        assert len(session.messages) == 800

    def test_concurrent_add_message_and_event_loop_interleave(self) -> None:
        """Threads calling add_message must not race with EventLoop message reads.

        Simulates the real scenario: one thread is an "EventLoop" reading
        session.messages while other threads inject messages via add_message.
        Both paths must acquire session._message_lock; no corruption allowed.
        """
        import asyncio

        manager = SessionManager(max_messages=10_000)
        session = manager.create()
        errors: list[Exception] = []

        def thread_add_messages() -> None:
            """Simulate a background thread calling add_message."""
            try:
                for i in range(500):
                    manager.add_message(session.id, {"role": "tool", "content": str(i)})
            except Exception as e:
                errors.append(e)

        async def simulate_event_loop_reads() -> None:
            """Simulate EventLoop acquiring session._message_lock to read messages."""
            for _ in range(500):
                with session._message_lock:  # type: ignore[attr-defined]
                    _ = list(session.messages)  # snapshot under lock
                await asyncio.sleep(0)  # yield to event loop

        def run_event_loop_sim() -> None:
            try:
                asyncio.run(simulate_event_loop_reads())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=thread_add_messages) for _ in range(4)] + [
            threading.Thread(target=run_event_loop_sim)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All 4 * 500 = 2000 thread messages must be present
        tool_messages = [m for m in session.messages if m["role"] == "tool"]
        assert len(tool_messages) == 2000


class TestInputValidation:
    """Tests for constructor input validation."""

    def test_max_sessions_zero_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_sessions must be positive"):
            SessionManager(max_sessions=0)

    def test_max_sessions_negative_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_sessions must be positive"):
            SessionManager(max_sessions=-1)

    def test_max_messages_zero_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_messages must be positive"):
            SessionManager(max_messages=0)

    def test_max_messages_negative_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_messages must be positive"):
            SessionManager(max_messages=-1)

    def test_negative_ttl_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="session_ttl must be non-negative"):
            SessionManager(session_ttl=-10)

    def test_negative_timedelta_ttl_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="session_ttl must be non-negative"):
            SessionManager(session_ttl=timedelta(seconds=-5))

    def test_zero_ttl_allowed(self) -> None:
        # Zero TTL is allowed (sessions expire immediately)
        manager = SessionManager(session_ttl=0)
        assert manager.session_ttl == timedelta(0)


class TestCleanupConsistency:
    """Tests for cleanup behavior across all methods."""

    def test_count_triggers_cleanup(self) -> None:
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=0.1, clock=fake_time)

        manager.create()

        # Advance time past TTL
        fake_time.advance(0.15)  # 150ms

        # count() should trigger cleanup
        assert manager.count() == 0

    def test_close_triggers_cleanup(self) -> None:
        fake_time = FakeClock(start=0.0)
        manager = SessionManager(session_ttl=0.1, clock=fake_time)

        session1 = manager.create()
        _ = manager.create()

        # Advance time past TTL
        fake_time.advance(0.15)  # 150ms

        # close() should trigger cleanup
        manager.close(session1.id)

        # Both sessions should be gone (session1 closed, session2 expired)
        assert manager.count() == 0
