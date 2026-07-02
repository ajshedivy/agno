# agno-os CLI

Command-line interface for interacting with [AgentOS](https://docs.agno.com) instances. Manage agents, teams, workflows, sessions, memories, knowledge, traces, evals, and metrics from the terminal.

Designed for both human operators and AI agent skills.

## Install

```bash
pip install -e agno_cli/
```

Or run without installing via `uvx`:

```bash
uvx --from ./agno_cli agno-os --help
```

## Quick Start

```bash
# Initialize config with your AgentOS URL
agno-os config init --url http://localhost:7777

# Check server status
agno-os status

# List available agents
agno-os agent list

# Run an agent
agno-os agent run my-agent "What is the capital of France?"

# Stream a response
agno-os agent run my-agent "Write a haiku about code" --stream

# Get JSON output (for scripting / agent skills)
agno-os -o json agent list
```

## Configuration

The CLI stores configuration at `~/.agno/config.yaml`. Named contexts let you switch between AgentOS endpoints.

### Managing Contexts

```bash
# Initialize with defaults
agno-os config init --url http://localhost:7777

# Add more contexts
agno-os config add staging --url https://staging.example.com --key sk-staging-xxx
agno-os config add production --url https://prod.example.com --key sk-prod-xxx

# Switch between contexts
agno-os config use staging

# View current context
agno-os config show

# List all contexts
agno-os config list

# Update a context property
agno-os config set staging url https://new-staging.example.com
agno-os config set staging security_key sk-new-key

# Remove a context
agno-os config remove staging
```

### Config File Format

```yaml
current_context: dev
contexts:
  dev:
    base_url: "http://localhost:7777"
    timeout: 60.0
    security_key: null
  production:
    base_url: "https://os-api.example.com"
    timeout: 60.0
    security_key: "sk-prod-xxx"
```

### Environment Variable Overrides

Environment variables override config file values (highest priority):

| Variable | Description |
|----------|-------------|
| `AGNO_CONTEXT` | Override active context name |
| `AGNO_BASE_URL` | Override base URL |
| `AGNO_SECURITY_KEY` | Override security key |
| `AGNO_TIMEOUT` | Override timeout (seconds) |
| `AGNO_OUTPUT` | Default output format (`table` or `json`) |

### CLI Flag Overrides

CLI flags override both config and environment variables:

```bash
agno-os --url http://other-server:7777 agent list
agno-os --key sk-override --url http://localhost:7777 status
```

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--context` | `-c` | Override active context |
| `--output` | `-o` | Output format: `table` (default), `json` |
| `--url` | | Override base URL |
| `--key` | | Override security key |
| `--timeout` | | Override timeout (seconds) |
| `--db-id` | | Database ID for multi-db endpoints |
| `--verbose` | `-v` | Enable verbose output |
| `--version` | `-V` | Show CLI version |

Global options must come **before** the subcommand:

```bash
agno-os -o json agent list       # correct
agno-os agent list -o json       # won't work (-o is a global option)
```

## Commands

### Status

```bash
agno-os status                   # Show OS info, agents, teams, workflows
agno-os -o json status           # JSON output
```

### Agents

```bash
agno-os agent list               # List all agents
agno-os agent get <agent_id>     # Get agent details

# Run an agent (non-streaming)
agno-os agent run <agent_id> "your message here"

# Run with streaming
agno-os agent run <agent_id> "your message" --stream

# Run with session context
agno-os agent run <agent_id> "follow-up question" --session-id <session_id>

# Continue a run (e.g., after tool approval)
agno-os agent continue <agent_id> <run_id> "continue message"

# Cancel a running agent
agno-os agent cancel <agent_id> <run_id>
```

### Teams

```bash
agno-os team list
agno-os team get <team_id>
agno-os team run <team_id> "your message" [--stream] [--session-id ID]
agno-os team cancel <team_id> <run_id>
```

### Workflows

```bash
agno-os workflow list
agno-os workflow get <workflow_id>
agno-os workflow run <workflow_id> "your message" [--stream] [--session-id ID]
agno-os workflow cancel <workflow_id> <run_id>
```

### Sessions

```bash
agno-os session list [--type agent] [--user-id USER]
agno-os session get <session_id> [--type agent]
agno-os session create --type agent --agent-id my-agent [--name "My Session"]
agno-os session update <session_id> --name "New Name"
agno-os session delete <session_id>
agno-os session delete-all [--type agent] [--user-id USER]
agno-os session runs <session_id> [--type agent]
```

### Memories

```bash
agno-os memory list [--user-id USER] [--topics "topic1,topic2"]
agno-os memory get <memory_id>
agno-os memory create --memory "User prefers dark mode" --user-id user1
agno-os memory update <memory_id> --memory "Updated content"
agno-os memory delete <memory_id>
agno-os memory delete-all [--user-id USER]
agno-os memory topics [--user-id USER]
agno-os memory stats [--user-id USER]
agno-os memory optimize [--user-id USER] [--apply]
```

### Knowledge

```bash
agno-os knowledge list
agno-os knowledge get <content_id>
agno-os knowledge upload ./document.pdf [--name "My Doc"] [--reader-id pdf]
agno-os knowledge update <content_id> --name "Renamed"
agno-os knowledge delete <content_id>
agno-os knowledge delete-all
agno-os knowledge status <content_id>
agno-os knowledge search "your query" [--max-results 10]
agno-os knowledge config
```

### Traces

```bash
agno-os trace list [--agent-id ID] [--status completed]
agno-os trace get <trace_id>
agno-os trace stats [--agent-id ID]
agno-os trace search --filter '{"op": "EQ", "field": "status", "value": "completed"}'
```

### Evals

```bash
agno-os eval list [--agent-id ID]
agno-os eval get <eval_run_id>
agno-os eval update <eval_run_id> --name "Renamed"
agno-os eval delete --ids "id1,id2"
```

### Metrics

```bash
agno-os metrics get [--start-date 2024-01-01] [--end-date 2024-12-31]
agno-os metrics refresh
```

### Models & Database

```bash
agno-os models list              # List all models used by agents
agno-os models migrate <db_id>   # Run database migrations
```

## Output Formats

### Table (default)

```
$ agno-os agent list
ID        NAME            DESCRIPTION
research  Research Agent  A simple research agent for testing
writer    Writer Agent    A writing assistant for testing
```

### JSON

```
$ agno-os -o json agent list
{
  "data": [
    {"id": "research", "name": "Research Agent", ...},
    {"id": "writer", "name": "Writer Agent", ...}
  ]
}
```

## Using with Agent Skills

The CLI is designed for programmatic use by AI agents and automation tools:

```bash
# Machine-readable output
export AGNO_OUTPUT=json

# Run agent and parse response
RESULT=$(agno-os agent run my-agent "Summarize this document")
echo "$RESULT" | jq '.content'

# List sessions for a user
agno-os session list --user-id user123 --limit 5

# Check agent availability
agno-os status | head -1
```

## Architecture

The CLI is a standalone Python package with no imports from the `agno` library. It uses:

- **httpx** for HTTP requests (sync, with SSE streaming support)
- **typer** for CLI framework
- **pydantic** for config models
- **pyyaml** for config file management

All API endpoint mappings are in individual command files under `agno_cli/commands/`, making maintenance straightforward when the AgentOS API changes.

## Development

```bash
# Install in editable mode
pip install -e "agno_cli/[dev]"

# Format
ruff format agno_cli/

# Lint
ruff check agno_cli/
```
