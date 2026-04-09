"""Agent commands — list, get, run, continue, cancel."""

import sys
from typing import Any, Optional

import typer

from agno_cli.client import AgnoClientError, parse_sse_events
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_agents() -> None:
    """List all agents."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        data = client.get("/agents")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_list(
        data=data,
        columns=["ID", "NAME", "DESCRIPTION"],
        keys=["id", "name", "description"],
    )


@app.command()
def get(agent_id: str = typer.Argument(help="Agent ID")) -> None:
    """Get detailed info for an agent."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        data = client.get(f"/agents/{agent_id}")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(
        data=data,
        fields=[
            ("ID", "id"),
            ("Name", "name"),
            ("Description", "description"),
            ("Model", "model"),
        ],
    )


@app.command()
def run(
    agent_id: str = typer.Argument(help="Agent ID"),
    message: str = typer.Argument(help="Message to send to the agent"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID for context"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Run an agent with a message."""
    from agno_cli.main import require_client

    client = require_client()
    endpoint = f"/agents/{agent_id}/runs"

    form_data = {"message": message, "stream": str(stream).lower()}
    if session_id:
        form_data["session_id"] = session_id
    if user_id:
        form_data["user_id"] = user_id

    try:
        if stream:
            _stream_run(client, endpoint, form_data)
        else:
            data = client.post(endpoint, data=form_data, as_form=True)
            if get_output_format() == "json":
                print_json(data)
            else:
                _print_run_output(data)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)


@app.command("continue")
def continue_run(
    agent_id: str = typer.Argument(help="Agent ID"),
    run_id: str = typer.Argument(help="Run ID to continue"),
    message: str = typer.Argument(help="Message to continue with"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Continue an existing agent run."""
    from agno_cli.main import require_client

    client = require_client()
    endpoint = f"/agents/{agent_id}/runs/{run_id}/continue"

    form_data = {"message": message, "stream": str(stream).lower()}
    if session_id:
        form_data["session_id"] = session_id
    if user_id:
        form_data["user_id"] = user_id

    try:
        if stream:
            _stream_run(client, endpoint, form_data)
        else:
            data = client.post(endpoint, data=form_data, as_form=True)
            if get_output_format() == "json":
                print_json(data)
            else:
                _print_run_output(data)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)


@app.command()
def cancel(
    agent_id: str = typer.Argument(help="Agent ID"),
    run_id: str = typer.Argument(help="Run ID to cancel"),
) -> None:
    """Cancel a running agent run."""
    from agno_cli.main import require_client

    client = require_client()
    try:
        client.post(f"/agents/{agent_id}/runs/{run_id}/cancel")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Cancelled run {run_id} for agent {agent_id}")


def _stream_run(client: Any, endpoint: str, form_data: dict) -> None:
    """Stream a run response, printing content tokens to stdout."""
    lines = client.stream_post(endpoint, data=form_data, as_form=True)
    output_json = get_output_format() == "json"
    events = []

    for event in parse_sse_events(lines):
        if output_json:
            events.append(event)
        else:
            # Only print content from RunContent events, skip RunCompleted which duplicates full output
            event_type = event.get("event", "")
            if event_type in ("RunContent", ""):
                content = event.get("content")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()

    if output_json:
        print_json(events)
    else:
        # Ensure newline after streamed content
        print()


def _print_run_output(data: dict) -> None:
    """Print non-streaming run output."""
    content = data.get("content")
    if content:
        print(content)

    # Print metrics summary to stderr
    metrics = data.get("metrics", {})
    if metrics:
        parts = []
        tokens = metrics.get("total_tokens")
        duration = metrics.get("duration")
        if tokens:
            parts.append(f"tokens: {tokens}")
        if duration:
            parts.append(f"time: {duration:.2f}s")
        if parts:
            print(f"\n[{', '.join(parts)}]", file=sys.stderr)
