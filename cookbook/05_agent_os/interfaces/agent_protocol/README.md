# Agent Protocol Interface

This cookbook demonstrates how to expose Agno agents via the [Agent Protocol](https://github.com/langchain-ai/agent-protocol) -- a framework-agnostic standard for serving LLM agents. Once exposed, any Agent Protocol client (including LangChain's `deepagents` package) can interact with your Agno agents.

## Examples

| File | Description |
|------|-------------|
| `basic.py` | AgentOS server with Agent Protocol interface |
| `async_subagent_example.py` | End-to-end: deepagents AsyncSubAgent connecting to AgentOS |

## Prerequisites

1. **AgentOS server** (basic.py):
   ```bash
   pip install agno
   export OPENAI_API_KEY=your_key_here
   python basic.py
   ```

2. **deepagents client** (async_subagent_example.py):
   ```bash
   pip install deepagents
   export ANTHROPIC_API_KEY=your_key_here
   python async_subagent_example.py
   ```

## How It Works

1. The `AgentProtocol` interface exposes Agno agents via standard endpoints:
   - `POST /ap/threads` - Create conversation threads
   - `POST /ap/threads/{id}/runs` - Execute agent runs
   - `GET /ap/threads/{id}/runs/{run_id}` - Check run status
   - `POST /ap/agents/search` - List available agents
   - `POST /ap/runs/wait` - Stateless run (wait for result)
   - `POST /ap/runs/stream` - Stateless run (streaming)
   - And more (store, thread CRUD, etc.)

2. The `deepagents` package uses the LangGraph SDK to communicate with these endpoints, enabling async subagent delegation.

## Architecture

```
AgentOS (basic.py)                     deepagents (async_subagent_example.py)
+---------------------------+          +----------------------------------+
| Agent Protocol Interface  |  HTTP    | create_deep_agent()              |
| /ap/threads              <----------+   AsyncSubAgent(                 |
| /ap/threads/{id}/runs    |          |     url="http://localhost:7778/ap"|
| /ap/agents/search        |          |     graph_id="research_agent"    |
| /ap/runs/wait            |          |   )                              |
+---------------------------+          +----------------------------------+
| Agno Agent               |
| (research_agent)         |
+---------------------------+
```
