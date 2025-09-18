"""Minimal example showing how to use MCPAgent for a single query."""

import asyncio

from agno.agent import MCPAgent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

tools = MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")

agent = MCPAgent(
    name="DocsAgent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[tools],
    markdown=True,
    auto_cleanup_mcp=True,
)


async def main() -> None:
    await agent.aprint_response(
        input="What is Agno?",
        stream=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
