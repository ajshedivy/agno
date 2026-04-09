"""Memory commands — list, get, create, update, delete, topics, stats, optimize."""

from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_memories(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Filter by user ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Filter by agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Filter by team ID"),
    topics: Optional[str] = typer.Option(None, "--topics", help="Comma-separated topics to filter"),
    search: Optional[str] = typer.Option(None, "--search", help="Search content"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    sort_by: Optional[str] = typer.Option(None, "--sort-by", help="Sort field"),
    sort_order: Optional[str] = typer.Option(None, "--sort-order", help="Sort order: asc, desc"),
) -> None:
    """List memories."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {
        "user_id": user_id,
        "agent_id": agent_id,
        "team_id": team_id,
        "search_content": search,
        "limit": limit,
        "page": page,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "db_id": get_db_id(),
    }
    if topics:
        params["topics"] = topics

    try:
        data = client.get("/memories", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["ID", "MEMORY", "TOPICS", "USER_ID"],
        keys=["id", "memory", "topics", "user_id"],
        pagination=pagination,
    )


@app.command()
def get(
    memory_id: str = typer.Argument(help="Memory ID"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Get a specific memory."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {"user_id": user_id, "db_id": get_db_id()}

    try:
        data = client.get(f"/memories/{memory_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(data=data)


@app.command()
def create(
    memory: str = typer.Option(..., "--memory", "-m", help="Memory content"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    topics: Optional[str] = typer.Option(None, "--topics", help="Comma-separated topics"),
) -> None:
    """Create a new memory."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    body = {"memory": memory}
    if user_id:
        body["user_id"] = user_id
    if topics:
        body["topics"] = topics.split(",")

    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        data = client.post("/memories", data=body, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        mid = data.get("id", "") if data else ""
        print_success(f"Created memory: {mid}")


@app.command()
def update(
    memory_id: str = typer.Argument(help="Memory ID"),
    memory: Optional[str] = typer.Option(None, "--memory", "-m", help="Updated memory content"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    topics: Optional[str] = typer.Option(None, "--topics", help="Comma-separated topics"),
) -> None:
    """Update a memory."""
    from agno_cli.main import require_client

    client = require_client()
    body = {}
    if memory:
        body["memory"] = memory
    if user_id:
        body["user_id"] = user_id
    if topics:
        body["topics"] = topics.split(",")

    try:
        data = client.patch(f"/memories/{memory_id}", data=body)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        print_success(f"Updated memory {memory_id}")


@app.command()
def delete(
    memory_id: str = typer.Argument(help="Memory ID"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """Delete a memory."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {"user_id": user_id, "db_id": get_db_id()}

    try:
        client.delete(f"/memories/{memory_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Deleted memory {memory_id}")


@app.command("delete-all")
def delete_all(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Team ID"),
) -> None:
    """Delete all memories matching filters."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    body = {}
    if user_id:
        body["user_id"] = user_id
    if agent_id:
        body["agent_id"] = agent_id
    if team_id:
        body["team_id"] = team_id

    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        client.delete("/memories", data=body if body else None, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success("Deleted memories.")


@app.command()
def topics(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
) -> None:
    """List memory topics."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {"user_id": user_id, "db_id": get_db_id()}

    try:
        data = client.get("/memory_topics", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        if data:
            for topic in data:
                print(topic)
        else:
            print("No topics found.")


@app.command()
def stats(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Agent ID"),
    team_id: Optional[str] = typer.Option(None, "--team-id", help="Team ID"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
) -> None:
    """Get user memory stats."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    params = {
        "user_id": user_id,
        "agent_id": agent_id,
        "team_id": team_id,
        "limit": limit,
        "page": page,
        "db_id": get_db_id(),
    }

    try:
        data = client.get("/user_memory_stats", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["USER_ID", "MEMORY_COUNT", "TOPIC_COUNT"],
        keys=["user_id", "memory_count", "topic_count"],
        pagination=pagination,
    )


@app.command()
def optimize(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID"),
    model: Optional[str] = typer.Option(None, "--model", help="Model to use for optimization"),
    apply: bool = typer.Option(False, "--apply", help="Apply optimizations (dry-run by default)"),
) -> None:
    """Optimize memories (merge duplicates, improve quality)."""
    from agno_cli.main import get_db_id, require_client

    client = require_client()
    body = {"apply": apply}
    if user_id:
        body["user_id"] = user_id
    if model:
        body["model"] = model

    params = {"db_id": get_db_id()} if get_db_id() else None

    try:
        data = client.post("/optimize-memories", data=body, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        if data:
            print_success(f"Optimization {'applied' if apply else 'preview (use --apply to execute)'}:")
            print_json(data)
        else:
            print_success("No optimizations found.")
