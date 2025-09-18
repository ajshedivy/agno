import asyncio

from agno.agent import Agent, MCPAgent
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools

tools = MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")


async def run_agent(message: str) -> None:
    agent = MCPAgent(
        tools=[tools],
        markdown=True,
        debug_mode=True
    )
    await agent.acli_app(input=message, stream=True)


if __name__ == "__main__":
    asyncio.run(run_agent("What is Agno?"))
