"""Status command — show AgentOS instance info."""

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, print_error, print_json, print_table

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def status(ctx: typer.Context) -> None:
    """Show AgentOS status and configuration."""
    if ctx.invoked_subcommand is not None:
        return

    from agno_cli.main import require_client

    client = require_client()
    try:
        data = client.get("/config")
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
        return

    # Summary info
    agents = data.get("agents", [])
    teams = data.get("teams", [])
    workflows = data.get("workflows", [])

    from agno_cli.console import print_detail

    print_detail(
        [
            ("OS ID", data.get("os_id", "")),
            ("Name", data.get("name", "") or ""),
            ("Description", data.get("description", "") or ""),
            ("Databases", ", ".join(data.get("databases", []))),
            ("Agents", str(len(agents))),
            ("Teams", str(len(teams))),
            ("Workflows", str(len(workflows))),
        ]
    )

    if agents:
        print("\nAgents:")
        print_table(
            ["ID", "NAME", "DESCRIPTION"],
            [[a.get("id", ""), a.get("name", ""), a.get("description", "") or ""] for a in agents],
        )

    if teams:
        print("\nTeams:")
        print_table(
            ["ID", "NAME", "MODE", "DESCRIPTION"],
            [
                [t.get("id", ""), t.get("name", ""), t.get("mode", "") or "", t.get("description", "") or ""]
                for t in teams
            ],
        )

    if workflows:
        print("\nWorkflows:")
        print_table(
            ["ID", "NAME", "DESCRIPTION"],
            [[w.get("id", ""), w.get("name", ""), w.get("description", "") or ""] for w in workflows],
        )
