"""AgentOS routes must not run sync database calls on the event loop.

A sync driver blocks the thread it runs on. On the event loop that stalls every
concurrent request for the length of the call; in the threadpool it stalls only
the request that made it. Each test slows one sync db call down, fires a
``/health`` while that call is in flight, and requires the health check to
come back before the slow call returns. On the event loop it cannot: the loop
is blocked until the call returns.
"""

import asyncio
import contextvars
import threading
import time
from typing import Any, Callable, Optional

import httpx
import pytest
from fastapi import FastAPI

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.os.utils import db_call

SLOW_S = 0.5


@pytest.fixture
def db(tmp_path) -> SqliteDb:
    return SqliteDb(db_file=str(tmp_path / "agno.db"))


def _app(db: SqliteDb) -> FastAPI:
    return AgentOS(agents=[Agent(id="code-agent", name="Code Agent", db=db)], db=db).get_app()


class _SlowCall:
    """Wraps a sync callable so it blocks for ``SLOW_S`` and records when it returned."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.started = threading.Event()
        self.returned_at: Optional[float] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.started.set()
        time.sleep(SLOW_S)
        self.returned_at = time.perf_counter()
        return self.fn(*args, **kwargs)


async def _health_while_in_flight(app: FastAPI, slow: _SlowCall, method: str, url: str, **kwargs: Any):
    """Send ``method url`` and, once its slow db call has started, a ``/health``.

    Returns the slow response and the time the health check completed.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def health() -> float:
            deadline = time.perf_counter() + 4 * SLOW_S
            while not slow.started.is_set():
                if time.perf_counter() > deadline:
                    raise AssertionError(f"{method} {url} never reached the slowed db call")
                await asyncio.sleep(0.005)
            response = await client.get("/health")
            assert response.status_code == 200
            return time.perf_counter()

        slow_response, health_done_at = await asyncio.gather(client.request(method, url, **kwargs), health())
    return slow_response, health_done_at


def _assert_health_beat_the_slow_call(health_done_at: float, slow: _SlowCall, what: str) -> None:
    assert slow.returned_at is not None
    assert health_done_at < slow.returned_at, (
        f"/health finished {health_done_at - slow.returned_at:.3f}s after the sync {what} returned"
    )


ROUTES = [
    # (id, db method, http method, url, expected status)
    ("sessions", "delete_session", "DELETE", "/sessions/s1?type=agent", 204),
    ("memory", "get_user_memories", "GET", "/memories", 200),
    ("traces", "get_traces", "GET", "/traces", 200),
    ("schedules", "get_schedules", "GET", "/schedules", 200),
    ("approvals", "get_approvals", "GET", "/approvals", 200),
    ("learnings", "list_learnings", "GET", "/learnings", 200),
    ("evals", "get_eval_runs", "GET", "/eval-runs", 200),
    ("components", "list_components", "GET", "/components", 200),
]


@pytest.mark.parametrize("db_method,http_method,url,status", [r[1:] for r in ROUTES], ids=[r[0] for r in ROUTES])
async def test_router_db_call_leaves_the_loop_responsive(db, monkeypatch, db_method, http_method, url, status):
    app = _app(db)
    slow = _SlowCall(getattr(db, db_method))
    monkeypatch.setattr(db, db_method, slow)

    response, health_done_at = await _health_while_in_flight(app, slow, http_method, url)

    assert response.status_code == status, response.text
    _assert_health_beat_the_slow_call(health_done_at, slow, db_method)


async def test_run_route_loads_stored_agent_off_the_loop(db, monkeypatch):
    """A run against a db-backed agent rehydrates it in the threadpool."""
    import agno.agent.agent as agent_module

    app = _app(db)
    slow = _SlowCall(lambda **kwargs: None)
    monkeypatch.setattr(agent_module, "get_agent_by_id", slow)

    response, health_done_at = await _health_while_in_flight(
        app, slow, "POST", "/agents/stored-agent/runs", data={"message": "hi", "stream": "false"}
    )

    assert response.status_code == 404, response.text
    _assert_health_beat_the_slow_call(health_done_at, slow, "stored-agent load")


async def test_agent_list_loads_stored_agents_off_the_loop(db, monkeypatch):
    import agno.agent.agent as agent_module

    app = _app(db)
    slow = _SlowCall(lambda **kwargs: [])
    monkeypatch.setattr(agent_module, "get_agents", slow)

    response, health_done_at = await _health_while_in_flight(app, slow, "GET", "/agents")

    assert response.status_code == 200, response.text
    _assert_health_beat_the_slow_call(health_done_at, slow, "stored-agent listing")


_request_marker: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_marker", default=None)


async def test_db_call_carries_the_callers_contextvars_into_the_thread():
    seen = {}

    def sync_method() -> None:
        seen["thread"] = threading.current_thread()
        seen["marker"] = _request_marker.get()

    _request_marker.set("request-1")
    await db_call(sync_method)

    assert seen["thread"] is not threading.main_thread()
    assert seen["marker"] == "request-1"


async def test_request_user_scope_reaches_the_db_call(db, monkeypatch):
    """The user id a route scopes a query to is the one the threaded db call sees.

    The scope is resolved from ``request.state`` on the loop and passed as an
    argument; a middleware-set contextvar must survive the hop as well.
    """
    app = _app(db)

    @app.middleware("http")
    async def mark(request, call_next):
        _request_marker.set(request.headers.get("x-marker"))
        return await call_next(request)

    seen = {}
    original = db.get_user_memories

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["thread"] = threading.current_thread()
        seen["user_id"] = kwargs.get("user_id")
        seen["marker"] = _request_marker.get()
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "get_user_memories", spy)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/memories", params={"user_id": "user-a"}, headers={"x-marker": "m-1"})

    assert response.status_code == 200, response.text
    assert seen["thread"] is not threading.main_thread()
    assert seen["user_id"] == "user-a"
    assert seen["marker"] == "m-1"


async def test_sqlite_db_works_from_worker_threads(db):
    """A file-backed SqliteDb (QueuePool, check_same_thread off) is usable from
    threadpool workers: writes from concurrent db_calls are visible to each other
    and to the loop thread."""
    from agno.db.schemas import UserMemory

    async def write(i: int) -> None:
        await db_call(db.upsert_user_memory, memory=UserMemory(memory_id=f"m{i}", memory=f"fact {i}", user_id="u"))

    await asyncio.gather(*(write(i) for i in range(8)))

    memories = await db_call(db.get_user_memories, user_id="u")
    assert sorted(m.memory_id for m in memories) == [f"m{i}" for i in range(8)]
    assert len(db.get_user_memories(user_id="u")) == 8
