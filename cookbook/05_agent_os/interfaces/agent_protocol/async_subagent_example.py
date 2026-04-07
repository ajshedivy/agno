"""
Async SubAgent Example
======================

End-to-end example: deepagents AsyncSubAgent connecting to an AgentOS
server running the Agent Protocol interface.

Prerequisites:
1. Start the Agent Protocol server:
   python cookbook/05_agent_os/interfaces/agent_protocol/basic.py

   The server will run on http://localhost:7778

2. Install deepagents:
   pip install deepagents

3. Set your ANTHROPIC_API_KEY environment variable:
   export ANTHROPIC_API_KEY=your_key_here
"""

import asyncio

from deepagents import AsyncSubAgent, create_deep_agent

# ---------------------------------------------------------------------------
# Create a Deep Agent with AsyncSubAgents
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[
        AsyncSubAgent(
            name="researcher",
            description="Performs deep research on a topic. Use this for in-depth analysis and investigation.",
            url="http://localhost:7778/ap",
            graph_id="research_agent",
        ),
        AsyncSubAgent(
            name="assistant",
            description="A helpful assistant for quick questions and concise answers.",
            url="http://localhost:7778/ap",
            graph_id="assistant_agent",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Run Example
# ---------------------------------------------------------------------------


async def main():
    """Demonstrate async subagent delegation."""
    print("=" * 60)
    print("Deep Agent with Async SubAgents")
    print("=" * 60)
    print()
    print("This agent delegates tasks to remote Agno agents exposed")
    print("via the Agent Protocol interface.")
    print()

    # The deep agent will decide which subagent to delegate to
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Research the latest breakthroughs in quantum computing in 2025."}]}
    )

    print("Response:")
    messages = response.get("messages", [])
    for msg in messages:
        if hasattr(msg, "content"):
            print(msg.content)
        elif isinstance(msg, dict):
            print(msg.get("content", ""))


if __name__ == "__main__":
    asyncio.run(main())
