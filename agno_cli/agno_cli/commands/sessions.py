"""Session commands — list, get, create, update, delete, runs."""

import json
from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_sessions(
    session_type: Optional[str] = typer.Option(None, "--type", "-t", help="Session type: agent, team, workflow"),
    component_id: Optional[str] = typer.Option(None, "--component-id", help="Filter by agent/team/workflow ID"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Filter by user ID"),
    session_name: Optional[str] = typer.Option(None, "--name", help="Filter by session name"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    sort_by: Optional[str] = typer.Option(None, "--sort-by", help="Sort field"),
    sort_order: Optional[str] = typer.Option(None, "--sort-order", help="Sort order: asc, desc"),
) -> None:
    """List sessions."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = {
        "session_type": session_type,
        "component_id": component_id,
        "user_id": user_id,
        "session_name": session_name,
        "limit": limit,
        "page": page,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "db_id": resolve_db_id(),
    }

    try:
        data = client.get("/sessions", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["SESSION_ID", "NAME", "AGENT_ID", "TEAM_ID", "USER_ID", "UPDATED_AT"],
        keys=["session_id", "session_name", "agent_id", "team_id", "user_id", "updated_at"],
        pagination=pagination,
    )


@app.command()
def get(
    session_id: str = typer.Argument(help="Session ID"),
    session_type: Optional[str] = typer.Option(None, "--type", "-t", help="Session type: agent, team, workflow"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Get session details."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = {"session_type": session_type, "user_id": user_id, "db_id": resolve_db_id()}

    try:
        data = client.get(f"/sessions/{session_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(data=data)


@app.command()
def create(
    session_type: str = typer.Option("agent", "--type", "-t", help="Session type: agent, team, workflow"),
    session_name: Optional[str] = typer.Option(None, "--name", help="Session name"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Custom session ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Team ID"),
    workflow_id: Optional[str] = typer.Option(None, "--workflow-id", help="Workflow ID"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    state: Optional[str] = typer.Option(None, "--state", help="Initial session state as JSON string"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="Metadata as JSON string"),
) -> None:
    """Create a new session."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    body = {"session_type": session_type}
    if session_id:
        body["session_id"] = session_id
    if session_name:
        body["session_name"] = session_name
    if agent_id:
        body["agent_id"] = agent_id
    if team_id:
        body["team_id"] = team_id
    if workflow_id:
        body["workflow_id"] = workflow_id
    if user_id:
        body["user_id"] = user_id
    if state:
        body["session_state"] = json.loads(state)
    if metadata:
        body["metadata"] = json.loads(metadata)

    db_id = resolve_db_id()
    params = {"db_id": db_id} if db_id else None

    try:
        data = client.post("/sessions", data=body, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        sid = data.get("session_id", "") if data else ""
        print_success(f"Created session: {sid}")


@app.command()
def update(
    session_id: str = typer.Argument(help="Session ID"),
    session_type: Optional[str] = typer.Option(None, "--type", "-t", help="Session type"),
    name: Optional[str] = typer.Option(None, "--name", help="New session name"),
    state: Optional[str] = typer.Option(None, "--state", help="Session state as JSON string"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="Metadata as JSON string"),
    summary: Optional[str] = typer.Option(None, "--summary", help="Session summary as JSON string"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Update a session."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    body = {}
    if name:
        body["session_name"] = name
    if state:
        body["session_state"] = json.loads(state)
    if metadata:
        body["metadata"] = json.loads(metadata)
    if summary:
        body["summary"] = json.loads(summary)

    params = {"session_type": session_type, "user_id": user_id, "db_id": resolve_db_id()}

    try:
        data = client.patch(f"/sessions/{session_id}", data={"body": body, "params": params})
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        print_success(f"Updated session {session_id}")


@app.command()
def delete(
    session_id: str = typer.Argument(help="Session ID"),
    session_type: Optional[str] = typer.Option(None, "--type", "-t", help="Session type"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Delete a session."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = {"session_type": session_type, "user_id": user_id, "db_id": resolve_db_id()}

    try:
        client.delete(f"/sessions/{session_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Deleted session {session_id}")


@app.command("delete-all")
def delete_all(
    session_type: Optional[str] = typer.Option(None, "--type", "-t", help="Session type"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    component_ids: Optional[str] = typer.Option(None, "--component-ids", help="Comma-separated component IDs"),
) -> None:
    """Delete multiple sessions."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = {
        "session_type": session_type,
        "user_id": user_id,
        "db_id": resolve_db_id(),
    }
    body = {}
    if component_ids:
        body["component_ids"] = component_ids.split(",")

    try:
        client.delete("/sessions", data=body if body else None, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success("Deleted sessions.")


@app.command()
def runs(
    session_id: str = typer.Argument(help="Session ID"),
    session_type: Optional[str] = typer.Option(None, "--type", "-t", help="Session type"),
    run_type: Optional[str] = typer.Option(None, "--run-type", help="Filter by run type"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    sort_by: Optional[str] = typer.Option(None, "--sort-by", help="Sort field"),
    sort_order: Optional[str] = typer.Option(None, "--sort-order", help="Sort order: asc, desc"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """List runs for a session."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = {
        "session_type": session_type,
        "run_type": run_type,
        "limit": limit,
        "page": page,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "user_id": user_id,
        "db_id": resolve_db_id(),
    }

    try:
        data = client.get(f"/sessions/{session_id}/runs", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["RUN_ID", "STATUS", "CREATED_AT"],
        keys=["run_id", "status", "created_at"],
        pagination=pagination,
    )
