"""Console output utilities for agno-os CLI.

Optimized for agent/skill consumption: clean, parseable text output.
No colors by default. Errors go to stderr.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Sequence


# Global output format — set by main.py callback
_output_format: str = "table"


def set_output_format(fmt: str) -> None:
    global _output_format
    _output_format = fmt


def get_output_format() -> str:
    return _output_format


def print_table(columns: List[str], rows: List[List[str]], footer: Optional[str] = None) -> None:
    """Print a plain-text aligned table to stdout.

    Example output:
        ID          NAME              DESCRIPTION
        research    Research Agent    Researches topics
        writer      Writing Agent     Writes content
    """
    if not rows:
        print("No results found.")
        return

    # Calculate column widths
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header = "  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(header)

    # Print rows
    for row in rows:
        line = "  ".join(str(cell).ljust(widths[i]) if i < len(widths) else str(cell) for i, cell in enumerate(row))
        print(line)

    if footer:
        print(f"\n{footer}")


def print_json(data: Any) -> None:
    """Print data as JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def print_detail(fields: List[tuple]) -> None:
    """Print key-value detail view.

    Example output:
        ID:           research
        Name:         Research Agent
        Description:  Researches topics
    """
    if not fields:
        return
    max_key_len = max(len(k) for k, _ in fields)
    for key, value in fields:
        print(f"{key + ':':<{max_key_len + 2}} {value}")


def print_error(msg: str) -> None:
    """Print error message to stderr."""
    print(f"Error: {msg}", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print success message to stdout."""
    print(msg)


def print_warning(msg: str) -> None:
    """Print warning message to stderr."""
    print(f"Warning: {msg}", file=sys.stderr)


def output_list(
    data: Sequence[Dict[str, Any]],
    columns: List[str],
    keys: List[str],
    pagination: Optional[Dict[str, Any]] = None,
) -> None:
    """Output a list of items in the current output format.

    Args:
        data: List of dicts to display
        columns: Column headers for table mode
        keys: Dict keys to extract for each column
        pagination: Optional pagination metadata dict
    """
    if get_output_format() == "json":
        result: Dict[str, Any] = {"data": data}
        if pagination:
            result["pagination"] = pagination
        print_json(result)
        return

    rows = []
    for item in data:
        row = [str(item.get(k, "") or "") for k in keys]
        rows.append(row)

    footer = None
    if pagination:
        page = pagination.get("page", 1)
        total_pages = pagination.get("total_pages", 1)
        total_count = pagination.get("total_count", len(data))
        footer = f"Page {page}/{total_pages} (total: {total_count})"

    print_table(columns, rows, footer=footer)


def output_detail(data: Dict[str, Any], fields: Optional[List[tuple]] = None) -> None:
    """Output a single item detail in the current output format.

    Args:
        data: The full dict to display
        fields: Optional list of (label, key) tuples for table mode.
                If not provided, outputs all keys.
    """
    if get_output_format() == "json":
        print_json(data)
        return

    if fields:
        display_fields = [(label, str(data.get(key, "") or "")) for label, key in fields]
    else:
        display_fields = [(k, str(v) if v is not None else "") for k, v in data.items()]

    print_detail(display_fields)
