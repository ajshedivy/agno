"""Knowledge commands — upload, list, get, update, delete, search, status, config."""

import mimetypes
from pathlib import Path
from typing import Optional

import typer

from agno_cli.client import AgnoClientError
from agno_cli.console import get_output_format, output_detail, output_list, print_error, print_json, print_success

app = typer.Typer(no_args_is_help=True)


def _knowledge_params(db_id: Optional[str] = None, knowledge_id: Optional[str] = None) -> dict:
    params = {}
    if db_id:
        params["db_id"] = db_id
    if knowledge_id:
        params["knowledge_id"] = knowledge_id
    return params or None  # type: ignore[return-value]


@app.command("list")
def list_content(
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    sort_by: Optional[str] = typer.Option(None, "--sort-by", help="Sort field"),
    sort_order: Optional[str] = typer.Option(None, "--sort-order", help="Sort order: asc, desc"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """List knowledge content."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = {
        "limit": limit,
        "page": page,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "db_id": resolve_db_id(),
        "knowledge_id": knowledge_id,
    }

    try:
        data = client.get("/knowledge/content", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["ID", "NAME", "STATUS", "CREATED_AT"],
        keys=["id", "name", "status", "created_at"],
        pagination=pagination,
    )


@app.command()
def get(
    content_id: str = typer.Argument(help="Content ID"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Get knowledge content details."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = _knowledge_params(resolve_db_id(), knowledge_id)

    try:
        data = client.get(f"/knowledge/content/{content_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(data=data)


@app.command()
def upload(
    file_path: Path = typer.Argument(help="Path to the file to upload", exists=True, readable=True),
    name: Optional[str] = typer.Option(None, "--name", help="Content name (defaults to filename)"),
    description: Optional[str] = typer.Option(None, "--description", help="Content description"),
    reader_id: Optional[str] = typer.Option(None, "--reader-id", help="Reader ID to use for processing"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Upload a file to the knowledge base."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    content_name = name or file_path.name
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    form_data = {"name": content_name}
    if description:
        form_data["description"] = description
    if reader_id:
        form_data["reader_id"] = reader_id

    db_id = resolve_db_id()
    if db_id:
        form_data["db_id"] = db_id
    if knowledge_id:
        form_data["knowledge_id"] = knowledge_id

    try:
        with open(file_path, "rb") as f:
            data = client.post_multipart(
                "/knowledge/content",
                files={"file": (file_path.name, f, mime_type)},
                data=form_data,
            )
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        cid = data.get("id", "") if data else ""
        print_success(f"Uploaded content: {cid} ({content_name})")


@app.command()
def update(
    content_id: str = typer.Argument(help="Content ID"),
    name: Optional[str] = typer.Option(None, "--name", help="Updated name"),
    description: Optional[str] = typer.Option(None, "--description", help="Updated description"),
    reader_id: Optional[str] = typer.Option(None, "--reader-id", help="Reader ID"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Update knowledge content metadata."""
    from agno_cli.main import require_client, resolve_db_id

    client = require_client()
    body = {}
    if name:
        body["name"] = name
    if description:
        body["description"] = description
    if reader_id:
        body["reader_id"] = reader_id

    try:
        data = client.patch(f"/knowledge/content/{content_id}", data=body, params={"db_id": resolve_db_id()})
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        print_success(f"Updated content {content_id}")


@app.command()
def delete(
    content_id: str = typer.Argument(help="Content ID"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Delete knowledge content."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = _knowledge_params(resolve_db_id(), knowledge_id)

    try:
        client.delete(f"/knowledge/content/{content_id}", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Deleted content {content_id}")


@app.command("delete-all")
def delete_all(
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Delete all knowledge content."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = _knowledge_params(resolve_db_id(), knowledge_id)

    try:
        data = client.delete("/knowledge/content", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    print_success(f"Deleted all content. {data or ''}")


@app.command()
def status(
    content_id: str = typer.Argument(help="Content ID"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Check processing status of knowledge content."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = _knowledge_params(resolve_db_id(), knowledge_id)

    try:
        data = client.get(f"/knowledge/content/{content_id}/status", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    output_detail(data=data)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum results"),
    search_type: Optional[str] = typer.Option(None, "--search-type", help="Search type"),
    limit: int = typer.Option(20, "--limit", "-l", help="Results per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Search knowledge base."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    body = {"query": query, "limit": limit, "page": page}
    if max_results:
        body["max_results"] = max_results
    if search_type:
        body["search_type"] = search_type

    params = _knowledge_params(resolve_db_id(), knowledge_id)

    try:
        data = client.post("/knowledge/search", data=body, params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
        return

    items = data.get("data", []) if isinstance(data, dict) else data
    pagination = data.get("meta") if isinstance(data, dict) else None

    output_list(
        data=items,
        columns=["ID", "CONTENT", "SCORE", "SOURCE"],
        keys=["id", "content", "score", "source"],
        pagination=pagination,
    )


@app.command("config")
def get_config(
    knowledge_id: Optional[str] = typer.Option(None, "--knowledge-id", help="Knowledge base ID"),
) -> None:
    """Get knowledge base configuration."""
    from agno_cli.main import resolve_db_id, require_client

    client = require_client()
    params = _knowledge_params(resolve_db_id(), knowledge_id)

    try:
        data = client.get("/knowledge/config", params=params)
    except AgnoClientError as e:
        print_error(e.message)
        raise typer.Exit(1)

    if get_output_format() == "json":
        print_json(data)
    else:
        output_detail(data=data)
