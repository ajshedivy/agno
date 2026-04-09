"""Trace commands — list, get, stats, search."""

import json
from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_traces(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Filter by session ID"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Filter by user ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Filter by agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Filter by team ID"),
    workflow_id: Optional[str] = typer.Option(None, "--workflow-id", help="Filter by workflow ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    start_time: Optional[str] = typer.Option(None, "--start-time", help="Start time (ISO format)"),
    end_time: Optional[str] = typer.Option(None, "--end-time", help="End time (ISO format)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
) -> None:
    """List execution traces."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {
        "run_id": run_id,
        "session_id": session_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "team_id": team_id,
        "workflow_id": workflow_id,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "page": page,
        "limit": limit,
        "db_id": get_db_id(),
    }

    try:
        data = client.get("/traces", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["TRACE_ID", "RUN_ID", "STATUS", "DURATION", "CREATED_AT"],
        keys=["trace_id", "run_id", "status", "duration", "created_at"],
        pagination=pagination,
    )


@app.command()
def get(
    trace_id: str = typer.Argument(help="Trace ID"),
    span_id: Optional[str] = typer.Option(None, "--span-id", help="Specific span ID"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run ID"),
) -> None:
    """Get trace details."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {"span_id": span_id, "run_id": run_id, "db_id": get_db_id()}

    try:
        data = client.get(f"/traces/{trace_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(data=data)


@app.command()
def stats(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Filter by user ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Filter by agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Filter by team ID"),
    workflow_id: Optional[str] = typer.Option(None, "--workflow-id", help="Filter by workflow ID"),
    start_time: Optional[str] = typer.Option(None, "--start-time", help="Start time (ISO format)"),
    end_time: Optional[str] = typer.Option(None, "--end-time", help="End time (ISO format)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
) -> None:
    """Get trace session stats."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {
        "user_id": user_id,
        "agent_id": agent_id,
        "team_id": team_id,
        "workflow_id": workflow_id,
        "start_time": start_time,
        "end_time": end_time,
        "page": page,
        "limit": limit,
        "db_id": get_db_id(),
    }

    try:
        data = client.get("/trace_session_stats", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["SESSION_ID", "RUN_COUNT", "TOTAL_TOKENS", "AVG_DURATION"],
        keys=["session_id", "run_count", "total_tokens", "avg_duration"],
        pagination=pagination,
    )


@app.command()
def search(
    filter_expr: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter expression as JSON"),
    group_by: Optional[str] = typer.Option(None, "--group-by", help="Group by: run, session"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
) -> None:
    """Search traces with filter expressions."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    body = {"page": page, "limit": limit}
    if filter_expr:
        body["filter_expr"] = json.loads(filter_expr)
    if group_by:
        body["group_by"] = group_by

    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        data = client.post("/traces/search", data=body, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
        return

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["TRACE_ID", "RUN_ID", "STATUS", "DURATION"],
        keys=["trace_id", "run_id", "status", "duration"],
        pagination=pagination,
    )
