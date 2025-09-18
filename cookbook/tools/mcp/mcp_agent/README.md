# MCPAgent Examples

These examples show how to use the [`MCPAgent`](../../../libs/agno/agno/agent/mcp_agent.py) class to manage Model Context Protocol (MCP) tool connections automatically. `MCPAgent` extends the base `Agent` with lifecycle helpers that connect to MCP servers just-in-time and optionally clean up the connection when the run finishes.

## Why MCPAgent?

The base `Agent` requires you to manually call `await toolkit.connect()` / `await toolkit.close()` in the right places. `MCPAgent` streamlines that workflow:

- Queues uninitialized `MCPTools` / `MultiMCPTools` during tool discovery.
- Connects toolkits before the agent or workflow actually calls the model.
- Rebuilds the tool list once the MCP server reports its available functions.
- Can auto-cleanup after every run (`auto_cleanup_mcp=True`, default) or keep the connection alive for multi-step workflows (`auto_cleanup_mcp=False` + `await agent.aclose()` when done).

## Examples

### 1. `simple_agent.py`

A minimal async script that creates an `MCPAgent`, asks “What is Agno?”, and prints the streaming response. It disables auto-cleanup so you can inspect connection state before explicitly closing it.

Run it with:

```bash
python cookbook/tools/mcp/mcp_agent/examples/simple_agent.py
```

### 2. `workflow_agent.py`

Demonstrates `MCPAgent` inside a `Workflow`. The research step uses `MCPAgent` to pull documentation, and a regular `Agent` formats the final answer. Auto-cleanup is disabled so the connection persists across the workflow steps; the script closes the agent at the end.

Run it with:

```bash
python cookbook/tools/mcp/mcp_agent/examples/workflow_agent.py
```

## Tips

- Use `auto_cleanup_mcp=False` when you need the same MCP connection across multiple runs or workflow steps, and call `await agent.aclose()` manually.
- Leave `auto_cleanup_mcp` at the default (`True`) for simple one-off runs where you prefer automatic teardown.
- For synchronous usage (e.g., `agent.run()`), MCPAgent handles the blocking connection by temporarily creating an event loop.

## Requirements

These examples rely on:

- The `mcp` Python package (`pip install mcp`).
- Access to an MCP server. The demos point to the Agno docs MCP endpoint.
- Valid OpenAI credentials for the chat model (`OPENAI_API_KEY`).
