"""Configuration management for agno-os CLI.

Manages ~/.agno/config.yaml with named contexts for switching between AgentOS endpoints.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

AGNO_CONFIG_DIR: Path = Path.home() / ".agno"
AGNO_CONFIG_FILE: Path = AGNO_CONFIG_DIR / "config.yaml"


class ContextConfig(BaseModel):
    """Configuration for a single AgentOS endpoint context."""

    base_url: str = Field(default="http://localhost:7777")
    timeout: float = Field(default=60.0)
    security_key: Optional[str] = Field(default=None)


class AgnoConfig(BaseModel):
    """Top-level CLI configuration with named contexts."""

    current_context: str = Field(default="default")
    contexts: Dict[str, ContextConfig] = Field(default_factory=lambda: {"default": ContextConfig()})


def load_config() -> AgnoConfig:
    """Load config from ~/.agno/config.yaml. Returns default config if file doesn't exist."""
    if not AGNO_CONFIG_FILE.exists():
        return AgnoConfig()
    try:
        with open(AGNO_CONFIG_FILE) as f:
            data = yaml.safe_load(f)
        if not data:
            return AgnoConfig()
        return AgnoConfig.model_validate(data)
    except Exception:
        return AgnoConfig()


def save_config(config: AgnoConfig) -> None:
    """Save config to ~/.agno/config.yaml."""
    AGNO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    with open(AGNO_CONFIG_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def resolve_context(
    context_name: Optional[str] = None,
    url_override: Optional[str] = None,
    key_override: Optional[str] = None,
    timeout_override: Optional[float] = None,
) -> ContextConfig:
    """Resolve the active context, applying overrides from env vars and CLI flags.

    Priority (highest to lowest):
    1. CLI flags (--url, --key, --timeout)
    2. Environment variables (AGNO_BASE_URL, AGNO_SECURITY_KEY, AGNO_TIMEOUT)
    3. Named context from config file
    4. Default context
    """
    config = load_config()

    # Determine which context to use
    name = context_name or os.environ.get("AGNO_CONTEXT") or config.current_context
    ctx = config.contexts.get(name)
    if ctx is None:
        print(
            f"Error: context '{name}' not found. Run 'agno-os config list' to see available contexts.", file=sys.stderr
        )
        raise SystemExit(1)

    # Apply environment variable overrides
    base_url = os.environ.get("AGNO_BASE_URL", ctx.base_url)
    security_key = os.environ.get("AGNO_SECURITY_KEY", ctx.security_key)
    timeout_str = os.environ.get("AGNO_TIMEOUT")
    timeout = float(timeout_str) if timeout_str else ctx.timeout

    # Apply CLI flag overrides (highest priority)
    if url_override:
        base_url = url_override
    if key_override:
        security_key = key_override
    if timeout_override is not None:
        timeout = timeout_override

    return ContextConfig(base_url=base_url, timeout=timeout, security_key=security_key)


def get_config_or_exit() -> AgnoConfig:
    """Load config, printing an error if it doesn't exist."""
    if not AGNO_CONFIG_FILE.exists():
        print("No configuration found. Run 'agno-os config init' to get started.", file=sys.stderr)
        raise SystemExit(1)
    return load_config()


def config_exists() -> bool:
    """Check if the config file exists."""
    return AGNO_CONFIG_FILE.exists()


def format_config_as_dict(config: AgnoConfig) -> Dict[str, Any]:
    """Convert config to a plain dict for display."""
    return config.model_dump(mode="json")
