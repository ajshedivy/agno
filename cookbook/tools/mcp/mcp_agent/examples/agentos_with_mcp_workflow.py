"""Workflow example showing MCPAgent alongside a standard Agent."""

import asyncio
from typing import AsyncIterator

from agno.agent import Agent, MCPAgent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.os.app import AgentOS
from agno.tools.mcp import MCPTools
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow


tools = MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")
research_agent = MCPAgent(
    name="ResearchAgent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[tools],
    markdown=True,
    auto_cleanup_mcp=False,
)

writer_agent = Agent(
    name="WriterAgent",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Write a concise summary based on the research results.",
)


async def prepare_research(step_input: StepInput) -> AsyncIterator[StepOutput]:
    topic = step_input.input
    yield StepOutput(content=f"Please research the topic: {topic}")


async def format_for_writer(step_input: StepInput) -> AsyncIterator[StepOutput]:
    research_text = step_input.previous_step_content
    yield StepOutput(content=f"Use this research to answer the user:\n{research_text}")

workflow = Workflow(
    name="MCP Research Workflow",
    description="Research using MCPAgent, then summarize with a standard Agent.",
    db=SqliteDb(session_table="mcp_workflow_sessions", db_file="tmp/mcp_workflow.db"),
    steps=[
        prepare_research,
        research_agent,
        format_for_writer,
        writer_agent,
    ],
)

agent_os = AgentOS(
    name="AgentOS",
    workflows=[workflow],
    agents=[research_agent, writer_agent]
)
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(app="agentos_with_mcp_workflow:app", reload=True)
