"""Guard: async route code must not call a sync database method directly.

A sync driver call inside ``async def`` blocks the event loop and every request
on it. Route code goes through ``db_call`` (or ``run_in_threadpool``), awaits an
async method, or loads components through the ``*_async`` lookups. This scan
flags, inside any ``async def`` under the route layer:

- a call on a db-like object (``db``, ``*_db``, ``x.db``, ``x.*_db``) that is not
  directly awaited, and
- a call to a sync component loader (``get_agent_by_id`` and friends).
"""

import ast
from pathlib import Path
from typing import List, Set, Tuple

import agno.os

OS_DIR = Path(agno.os.__file__).parent
SCANNED = [
    *sorted((OS_DIR / "routers").rglob("*.py")),
    *sorted((OS_DIR / "middleware").rglob("*.py")),
    OS_DIR / "utils.py",
]

SYNC_LOADERS = {
    "allow_draft_preview",
    "get_agent_by_id",
    "get_team_by_id",
    "get_workflow_by_id",
    "get_agents",
    "get_teams",
    "get_workflows",
    # How the *_async lookups import the db loaders.
    "get_agent_by_id_db",
    "get_team_by_id_db",
    "get_workflow_by_id_db",
}

# (path relative to agno/os, called expression): deliberate, non-blocking calls.
ALLOWED = {
    # Reports the vector db's capabilities from its config; no I/O.
    ("routers/knowledge/knowledge.py", "knowledge.vector_db.get_supported_search_types"),
}


def _is_db_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "db" or node.id.endswith("_db")
    if isinstance(node, ast.Attribute):
        return node.attr == "db" or node.attr.endswith("_db")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "cast":
        return len(node.args) == 2 and _is_db_like(node.args[1])
    return False


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.in_async = False
        self.awaited: Set[int] = set()
        self.hits: List[Tuple[int, str]] = []

    def _scope(self, node: ast.AST, in_async: bool) -> None:
        outer, self.in_async = self.in_async, in_async
        self.generic_visit(node)
        self.in_async = outer

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope(node, True)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested sync def (or lambda) is code handed to a thread, not loop code.
        self._scope(node, False)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scope(node, False)

    def visit_Await(self, node: ast.Await) -> None:
        self.awaited.add(id(node.value))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.in_async and id(node) not in self.awaited:
            func = node.func
            if isinstance(func, ast.Attribute) and _is_db_like(func.value):
                self.hits.append((node.lineno, ast.unparse(func)))
            elif isinstance(func, ast.Name) and func.id in SYNC_LOADERS:
                self.hits.append((node.lineno, func.id))
        self.generic_visit(node)


def _scan(source: str) -> List[Tuple[int, str]]:
    scanner = _Scanner()
    scanner.visit(ast.parse(source))
    return scanner.hits


def test_scanner_flags_the_blocking_patterns():
    source = """
async def route(db, os):
    a = db.get_session(session_id="s")
    b = await db.aget_session(session_id="s")
    c = await db_call(db.get_session, session_id="s")
    d = await run_in_threadpool(lambda: os.db.get_sessions())
    e = cast(BaseDb, db).get_learning_by_id("x")
    f = get_agent_by_id("a", db=db)
    g = await get_agent_by_id_async("a", db=db)
    def inner():
        return db.get_session(session_id="s")
"""
    assert _scan(source) == [(3, "db.get_session"), (7, "cast(BaseDb, db).get_learning_by_id"), (8, "get_agent_by_id")]


def test_async_route_code_never_calls_a_sync_db_method_on_the_loop():
    violations = []
    for path in SCANNED:
        rel = path.relative_to(OS_DIR).as_posix()
        for lineno, call in _scan(path.read_text()):
            if (rel, call) not in ALLOWED:
                violations.append(f"agno/os/{rel}:{lineno}: {call}(...)")
    assert not violations, (
        "Sync db calls on the event loop; wrap them in db_call(...) / run_in_threadpool(...) "
        "or use the *_async loaders:\n" + "\n".join(violations)
    )
