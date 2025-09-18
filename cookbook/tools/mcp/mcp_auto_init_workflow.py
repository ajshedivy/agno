import asyncio
from textwrap import dedent
from typing import AsyncIterator

from agno.agent import Agent, MCPAgent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.os.app import AgentOS
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.mcp import MCPTools
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

tools = MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")

# Define agents
docs_agent = MCPAgent(tools=[tools], markdown=True, debug_mode=True, auto_cleanup_mcp=False)


writer_agent = Agent(
    name="Writer Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Write a blog post on the topic",
)


async def prepare_input_for_web_search(
    step_input: StepInput,
) -> AsyncIterator[StepOutput]:
    """Generator function that yields StepOutput"""
    topic = step_input.input

    # Create proper StepOutput content
    content = dedent(
        f"""\
        I'm writing a blog post on the topic
        <topic>
        {topic}
        </topic>

        Search the docs for relevant information.
        """
    )

    # Yield a StepOutput as the final result
    yield StepOutput(content=content)


async def prepare_input_for_writer(step_input: StepInput) -> AsyncIterator[StepOutput]:
    """Generator function that yields StepOutput"""
    topic = step_input.input
    research_team_output = step_input.previous_step_content

    # Create proper StepOutput content
    content = dedent(
        f"""\
        I'm writing a blog post on the topic:
        <topic>
        {topic}
        </topic>

        Here is information from the web:
        <research_results>
        {research_team_output}
        </research_results>\
        """
    )

    # Yield a StepOutput as the final result
    yield StepOutput(content=content)


# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[docs_agent],
    instructions="Research tech topics from Hackernews and the web",
)

content_creation_workflow = Workflow(
    name="Blog Post Workflow",
    description="Automated blog post creation from Hackernews and the web",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="tmp/workflow.db",
    ),
    steps=[
        prepare_input_for_web_search,
        docs_agent,
        prepare_input_for_writer,
        writer_agent,
    ],
)

agent_os = AgentOS(
    name="AgentOS",
    workflows=[content_creation_workflow],
    agents=[docs_agent, writer_agent]
)
app = agent_os.get_app()


# Create and use workflow
def main() -> None:
    content_creation_workflow = Workflow(
        name="Blog Post Workflow",
        description="Automated blog post creation from Hackernews and the web",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        steps=[
            prepare_input_for_web_search,
            docs_agent,
            prepare_input_for_writer,
            writer_agent,
        ],
    )
    
    agent_os = AgentOS(
        name="AgentOS",
        workflows=[content_creation_workflow],
        agents=[docs_agent, writer_agent]
    )
    app = agent_os.get_app()
    agent_os.serve(app="mcp_auto_init_workflow:app")




if __name__ == "__main__":
    agent_os.serve(app="mcp_auto_init_workflow:app")
