"""Models and database commands."""

from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import output_list, print_error, print_success

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_models() -> None:
    """List all models used by agents and teams."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        data = client.get("/models")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_list(
        data=data,
        columns=["ID", "PROVIDER"],
        keys=["id", "provider"],
    )


@app.command()
def migrate(
    db_id: str = typer.Argument(help="Database ID to migrate"),
    target_version: Optional[str] = typer.Option(None, "--target-version", help="Target migration version"),
) -> None:
    """Migrate a database to a target version."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        client.post(f"/databases/{db_id}/migrate", data={"target_version": target_version})
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Database '{db_id}' migration initiated.")
