"""Team commands — list, get, run, cancel."""

import sys
from typing import Optional

import typer

from agno_cli.client import AgnoClientError, parse_sse_events
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_teams() -> None:
    """List all teams."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        data = client.get("/teams")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_list(
        data=data,
        columns=["ID", "NAME", "MODE", "DESCRIPTION"],
        keys=["id", "name", "mode", "description"],
    )


@app.command()
def get(team_id: str = typer.Argument(help="Team ID")) -> None:
    """Get detailed info for a team."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        data = client.get(f"/teams/{team_id}")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(
        data=data,
        fields=[
            ("ID", "id"),
            ("Name", "name"),
            ("Description", "description"),
            ("Mode", "mode"),
            ("Model", "model"),
        ],
    )


@app.command()
def run(
    team_id: str = typer.Argument(help="Team ID"),
    message: str = typer.Argument(help="Message to send to the team"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID for context"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Run a team with a message."""
    from agno_cli.main import require_client

    client = require_client()
    endpoint = f"/teams/{team_id}/runs"

    form_data = {"message": message, "stream": str(stream).lower()}
    if session_id:
        form_data["session_id"] = session_id
    if user_id:
        form_data["user_id"] = user_id

    try:
        if stream:
            lines = client.stream_post(endpoint, data=form_data, as_form=True)
            output_json = get_output_format() == "json"
            events = []
            for event in parse_sse_events(lines):
                if output_json:
                    events.append(event)
                else:
                    event_type = event.get("event", "")
                    if event_type in ("RunContent", ""):
                        content = event.get("content")
                        if content:
                            sys.stdout.write(content)
                            sys.stdout.flush()
            if output_json:
                print_json(events)
            else:
                print()
        else:
            data = client.post(endpoint, data=form_data, as_form=True)
            if get_output_format() == "json":
                print_json(data)
            else:
                content = data.get("content") if data else None
                if content:
                    print(content)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)


@app.command()
def cancel(
    team_id: str = typer.Argument(help="Team ID"),
    run_id: str = typer.Argument(help="Run ID to cancel"),
) -> None:
    """Cancel a running team run."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        client.post(f"/teams/{team_id}/runs/{run_id}/cancel")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Cancelled run {run_id} for team {team_id}")
