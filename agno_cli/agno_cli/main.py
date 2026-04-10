"""Main entrypoint for agno-os CLI."""

from typing import Optional

import typer

from agno_cli import __version__
from agno_cli.client import AgnoClient
from agno_cli.config import ContextConfig, resolve_context
from agno_cli.console import print_error, set_output_format

app = typer.Typer(
    name="agno-os",
    help="CLI for interacting with AgentOS instances.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

# Shared state for resolved context and client
_state: dict = {}


def get_client() -> AgnoClient:
    """Get or create the AgnoClient from resolved context."""
    if "client" not in _state:
        ctx: ContextConfig = _state["context"]
        _state["client"] = AgnoClient(
            base_url=ctx.base_url,
            timeout=ctx.timeout,
            security_key=ctx.security_key,
        )
    return _state["client"]


def get_db_id() -> Optional[str]:
    """Get the db_id from global state."""
    return _state.get("db_id")


def resolve_db_id() -> Optional[str]:
    """Resolve db_id: explicit flag > single-db auto-detect > multi-db error."""
    explicit = _state.get("db_id")
    if explicit:
        return explicit

    if "resolved_db_id" in _state:
        return _state["resolved_db_id"]

    # Fetch server config to discover available databases
    try:
        client = get_client()
        config = client.get("/config")
        dbs = config.get("databases", [])
    except Exception:
        # Can't reach server — let the actual command fail with its own error
        _state["resolved_db_id"] = None
        return None

    if len(dbs) == 1:
        _state["resolved_db_id"] = dbs[0]
        return dbs[0]
    elif len(dbs) > 1:
        db_list = ", ".join(dbs)
        print_error(
            f"Multiple databases available: {db_list}. "
            "Specify one with --db-id (e.g., agno-os --db-id <name> ...)."
        )
        raise typer.Exit(1)
    else:
        _state["resolved_db_id"] = None
        return None


def version_callback(value: bool) -> None:
    if value:
        print(f"agno-os {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    context: Optional[str] = typer.Option(
        None, "--context", "-c", help="Override active context", envvar="AGNO_CONTEXT"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json", envvar="AGNO_OUTPUT"),
    url: Optional[str] = typer.Option(None, "--url", help="Override base URL", envvar="AGNO_BASE_URL"),
    key: Optional[str] = typer.Option(None, "--key", help="Override security key", envvar="AGNO_SECURITY_KEY"),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Override timeout in seconds", envvar="AGNO_TIMEOUT"
    ),
    db_id: Optional[str] = typer.Option(None, "--db-id", help="Database ID for multi-db endpoints"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    version: bool = typer.Option(
        False, "--version", "-V", callback=version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    """CLI for interacting with AgentOS instances."""
    set_output_format(output)
    _state["db_id"] = db_id
    _state["verbose"] = verbose

    # Resolve context (defer errors for config commands that don't need it)
    try:
        ctx = resolve_context(
            context_name=context,
            url_override=url,
            key_override=key,
            timeout_override=timeout,
        )
        _state["context"] = ctx
    except SystemExit:
        # Allow config commands to work without a valid context
        _state["context"] = None


def require_client() -> AgnoClient:
    """Get client, raising an error if context is not configured."""
    if _state.get("context") is None:
        print_error("No valid context configured. Run 'agno-os config init' to get started.")
        raise typer.Exit(1)
    return get_client()


def _register_commands() -> None:
    """Register all command groups."""
    from agno_cli.commands.agents import app as agents_app
    from agno_cli.commands.config_cmd import app as config_app
    from agno_cli.commands.evals import app as evals_app
    from agno_cli.commands.knowledge import app as knowledge_app
    from agno_cli.commands.memories import app as memories_app
    from agno_cli.commands.metrics import app as metrics_app
    from agno_cli.commands.models import app as models_app
    from agno_cli.commands.sessions import app as sessions_app
    from agno_cli.commands.status import app as status_app
    from agno_cli.commands.teams import app as teams_app
    from agno_cli.commands.traces import app as traces_app
    from agno_cli.commands.workflows import app as workflows_app

    app.add_typer(config_app, name="config", help="Manage endpoint contexts and configuration.")
    app.add_typer(status_app, name="status", help="Show AgentOS status and configuration.")
    app.add_typer(models_app, name="models", help="List models and manage databases.")
    app.add_typer(agents_app, name="agent", help="Manage and run agents.")
    app.add_typer(teams_app, name="team", help="Manage and run teams.")
    app.add_typer(workflows_app, name="workflow", help="Manage and run workflows.")
    app.add_typer(sessions_app, name="session", help="Manage sessions.")
    app.add_typer(memories_app, name="memory", help="Manage user memories.")
    app.add_typer(knowledge_app, name="knowledge", help="Manage knowledge base content.")
    app.add_typer(traces_app, name="trace", help="View execution traces.")
    app.add_typer(evals_app, name="eval", help="Manage evaluation runs.")
    app.add_typer(metrics_app, name="metrics", help="View and refresh metrics.")


_register_commands()


if __name__ == "__main__":
    app()
