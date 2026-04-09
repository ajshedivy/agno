"""Eval commands — list, get, delete, update."""

from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_evals(
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Filter by agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Filter by team ID"),
    workflow_id: Optional[str] = typer.Option(None, "--workflow-id", help="Filter by workflow ID"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Filter by model ID"),
    filter_type: Optional[str] = typer.Option(None, "--filter-type", help="Filter type"),
    eval_types: Optional[str] = typer.Option(None, "--eval-types", help="Comma-separated eval types"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    sort_by: Optional[str] = typer.Option(None, "--sort-by", help="Sort field"),
    sort_order: Optional[str] = typer.Option(None, "--sort-order", help="Sort order: asc, desc"),
) -> None:
    """List evaluation runs."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {
        "agent_id": agent_id,
        "team_id": team_id,
        "workflow_id": workflow_id,
        "model_id": model_id,
        "filter_type": filter_type,
        "limit": limit,
        "page": page,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "db_id": get_db_id(),
    }
    if eval_types:
        params["eval_types"] = eval_types

    try:
        data = client.get("/eval-runs", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["ID", "NAME", "STATUS", "AGENT_ID", "CREATED_AT"],
        keys=["eval_run_id", "name", "status", "agent_id", "created_at"],
        pagination=pagination,
    )


@app.command()
def get(eval_run_id: str = typer.Argument(help="Eval run ID")) -> None:
    """Get eval run details."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        data = client.get(f"/eval-runs/{eval_run_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(data=data)


@app.command()
def delete(
    ids: str = typer.Option(..., "--ids", help="Comma-separated eval run IDs to delete"),
) -> None:
    """Delete evaluation runs."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    body = {"eval_run_ids": ids.split(",")}
    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        client.delete("/eval-runs", data=body, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Deleted eval runs: {ids}")


@app.command()
def update(
    eval_run_id: str = typer.Argument(help="Eval run ID"),
    name: str = typer.Option(..., "--name", help="New name for the eval run"),
) -> None:
    """Update an eval run."""
    from agno_cli.main import require_client

    client = require_client()
    body = {"name": name}

    try:
        data = client.patch(f"/eval-runs/{eval_run_id}", data=body)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        print_success(f"Updated eval run {eval_run_id}")
