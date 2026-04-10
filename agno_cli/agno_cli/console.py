"""Console output utilities for agno-os CLI.

Optimized for agent/skill consumption: clean, parseable text output.
No colors by default. Errors go to stderr.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


# Global output format — set by main.py callback
_output_format: str = "table"


def set_output_format(fmt: str) -> None:
    global _output_format
    _output_format = fmt


def get_output_format() -> str:
    return _output_format


def _get_terminal_width() -> int:
    """Get terminal width, falling back to 120 if not a TTY."""
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 120


def print_table(columns: List[str], rows: List[List[str]], footer: Optional[str] = None) -> None:
    """Print a plain-text aligned table to stdout.

    Compacts output to fit terminal width by:
    1. Dropping columns that are empty across all rows
    2. Truncating wide columns with ellipsis when needed

    Example output:
        ID          NAME              DESCRIPTION
        research    Research Agent    Researches topics
        writer      Writing Agent     Writes content
    """
    if not rows:
        print("No results found.")
        return

    gap = "  "
    term_width = _get_terminal_width()

    keep = []
    for i in range(len(columns)):
        if any(i < len(row) and str(row[i]).strip() for row in rows):
            keep.append(i)
    if not keep:
        keep = list(range(len(columns)))

    columns = [columns[i] for i in keep]
    rows = [[str(row[i]) if i < len(row) else "" for i in keep] for row in rows]

    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Shrink widest columns to fit terminal, distributing reduction evenly
    if len(widths) > 0:
        available = term_width - len(gap) * (len(widths) - 1)
        minimums = [min(len(c), 6) for c in columns]
        while sum(widths) > available:
            max_w = max(widths)
            max_i = widths.index(max_w)
            if max_w <= minimums[max_i]:
                break
            # Shrink to second-widest or minimum, whichever is larger
            sorted_w = sorted(set(widths), reverse=True)
            target = sorted_w[1] if len(sorted_w) > 1 else minimums[max_i]
            target = max(target, minimums[max_i])
            widths[max_i] = max(target, max_w - (sum(widths) - available))

    def _trunc(text: str, width: int) -> str:
        if len(text) <= width:
            return text.ljust(width)
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    def _format_row(cells: List[str]) -> str:
        parts = [_trunc(cells[i], widths[i]) for i in range(min(len(cells), len(widths)))]
        return gap.join(parts).rstrip()

    print(_format_row(columns))
    for row in rows:
        print(_format_row(row))

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
