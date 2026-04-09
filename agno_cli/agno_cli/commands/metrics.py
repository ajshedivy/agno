"""Metrics commands — get, refresh."""

from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


@app.command()
def get(
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date (YYYY-MM-DD)"),
) -> None:
    """Get aggregated metrics."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {
        "starting_date": start_date,
        "ending_date": end_date,
        "db_id": get_db_id(),
    }

    try:
        data = client.get("/metrics", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        from agno_cli.console import output_detail

        output_detail(data=data)


@app.command()
def refresh() -> None:
    """Refresh metrics (recompute aggregations)."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        data = client.post("/metrics/refresh", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        if data and isinstance(data, list):
            output_list(
                data=data,
                columns=["DATE", "RUNS", "TOKENS", "SESSIONS"],
                keys=["date", "total_runs", "total_tokens", "total_sessions"],
            )
        else:
            print_success("Metrics refreshed.")
