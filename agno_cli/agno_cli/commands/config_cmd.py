"""Config commands for managing AgentOS endpoint contexts."""

from typing import Optional

import typer

from agno_cli.config import (
    AGNO_CONFIG_FILE,
    AgnoConfig,
    ContextConfig,
    config_exists,
    load_config,
    save_config,
)
from agno_cli.console import get_output_format, print_detail, print_error, print_json, print_success, print_table

app = typer.Typer(no_args_is_help=True)


@app.command()
def init(
    url: str = typer.Option("http://localhost:7777", "--url", help="Base URL for the default context"),
    key: Optional[str] = typer.Option(None, "--key", help="Security key for the default context"),
    timeout: float = typer.Option(60.0, "--timeout", help="Timeout in seconds"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
) -> None:
    """Initialize configuration with a default context."""
    if config_exists() and not force:
        print_error(f"Config already exists at {AGNO_CONFIG_FILE}. Use --force to overwrite.")
        raise typer.Exit(1)

    config = AgnoConfig(
        current_context="default",
        contexts={"default": ContextConfig(base_url=url, timeout=timeout, security_key=key)},
    )
    save_config(config)
    print_success(f"Config initialized at {AGNO_CONFIG_FILE}")


@app.command("list")
def list_contexts() -> None:
    """List all configured contexts."""
    config = load_config()

    if get_output_format() == "json":
        print_json(
            {
                "current_context": config.current_context,
                "contexts": {k: v.model_dump(mode="json") for k, v in config.contexts.items()},
            }
        )
        return

    columns = ["NAME", "URL", "ACTIVE"]
    rows = []
    for name, ctx in config.contexts.items():
        active = "*" if name == config.current_context else ""
        rows.append([name, ctx.base_url, active])
    print_table(columns, rows)


@app.command()
def use(name: str = typer.Argument(help="Context name to switch to")) -> None:
    """Switch the active context."""
    config = load_config()
    if name not in config.contexts:
        print_error(f"Context '{name}' not found. Available: {', '.join(config.contexts.keys())}")
        raise typer.Exit(1)
    config.current_context = name
    save_config(config)
    print_success(f"Switched to context '{name}' ({config.contexts[name].base_url})")


@app.command()
def add(
    name: str = typer.Argument(help="Name for the new context"),
    url: str = typer.Option(..., "--url", help="Base URL of the AgentOS instance"),
    key: Optional[str] = typer.Option(None, "--key", help="Security key"),
    timeout: float = typer.Option(60.0, "--timeout", help="Timeout in seconds"),
) -> None:
    """Add a new context."""
    config = load_config()
    if name in config.contexts:
        print_error(f"Context '{name}' already exists. Use 'agno-os config set' to update it.")
        raise typer.Exit(1)
    config.contexts[name] = ContextConfig(base_url=url, timeout=timeout, security_key=key)
    save_config(config)
    print_success(f"Added context '{name}' ({url})")


@app.command()
def remove(name: str = typer.Argument(help="Context name to remove")) -> None:
    """Remove a context."""
    config = load_config()
    if name not in config.contexts:
        print_error(f"Context '{name}' not found.")
        raise typer.Exit(1)
    if name == config.current_context:
        print_error(f"Cannot remove active context '{name}'. Switch to another context first.")
        raise typer.Exit(1)
    del config.contexts[name]
    save_config(config)
    print_success(f"Removed context '{name}'")


@app.command()
def show() -> None:
    """Show the current context details."""
    config = load_config()
    name = config.current_context
    ctx = config.contexts.get(name)
    if ctx is None:
        print_error(f"Active context '{name}' not found in config.")
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json({"name": name, **ctx.model_dump(mode="json")})
        return

    print_detail(
        [
            ("Context", name),
            ("URL", ctx.base_url),
            ("Timeout", str(ctx.timeout)),
            ("Security Key", "***" if ctx.security_key else "(none)"),
            ("Config File", str(AGNO_CONFIG_FILE)),
        ]
    )


@app.command("set")
def set_property(
    name: str = typer.Argument(help="Context name"),
    key: str = typer.Argument(help="Property to set: url, timeout, security_key"),
    value: str = typer.Argument(help="New value"),
) -> None:
    """Update a property of a context."""
    config = load_config()
    if name not in config.contexts:
        print_error(f"Context '{name}' not found.")
        raise typer.Exit(1)

    ctx = config.contexts[name]
    if key == "url":
        ctx.base_url = value
    elif key == "timeout":
        ctx.timeout = float(value)
    elif key == "security_key":
        ctx.security_key = value
    else:
        print_error(f"Unknown property '{key}'. Valid: url, timeout, security_key")
        raise typer.Exit(1)

    config.contexts[name] = ctx
    save_config(config)
    print_success(f"Updated {name}.{key} = {value if key != 'security_key' else '***'}")
