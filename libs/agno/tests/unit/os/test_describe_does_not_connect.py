"""Describing an agent or team opens no tool connection.

The AgentOS list and detail routes describe a component's tools with
``AgentResponse.from_agent`` / ``TeamResponse.from_team``. Those built the
run-time tool list, which connects MCP and connectable tools; only a run
closes them, so every listing opened connections that nothing closed (and
sent the tools' credentials to their servers). Describing now passes
``connect_tools=False``; a run still connects as before.
"""

import pytest

from agno.agent import Agent
from agno.os.routers.agents.schema import AgentResponse
from agno.os.routers.teams.schema import TeamResponse
from agno.run.agent import RunOutput
from agno.run.base import RunContext
from agno.session import AgentSession
from agno.team import Team
from agno.tools import Toolkit


class MCPTools(Toolkit):
    """Stands in for agno's MCPTools: the tool assembly detects it by class name."""

    def __init__(self):
        super().__init__(name="fake_mcp")
        self.initialized = False
        self.refresh_connection = True
        self.connects = 0

    async def connect(self, force: bool = False):
        self.connects += 1
        self.initialized = True

    async def is_alive(self):
        return self.initialized

    async def build_tools(self):
        pass

    async def close(self):
        self.initialized = False


class Connectable(Toolkit):
    @property
    def requires_connect(self) -> bool:
        return True

    def __init__(self):
        super().__init__(name="connectable")
        self.connects = 0

    def connect(self):
        self.connects += 1

    def close(self):
        pass


@pytest.mark.asyncio
async def test_describing_an_agent_connects_nothing():
    mcp, conn = MCPTools(), Connectable()
    agent = Agent(id="a", name="A", tools=[mcp, conn])

    await AgentResponse.from_agent(agent)
    await AgentResponse.from_agent(agent)

    assert mcp.connects == 0
    assert conn.connects == 0
    assert agent._mcp_tools_initialized_on_run == []
    assert agent._connectable_tools_initialized_on_run == []


@pytest.mark.asyncio
async def test_describing_a_team_connects_nothing():
    member_mcp, team_conn = MCPTools(), Connectable()
    member = Agent(id="m", name="M", tools=[member_mcp])
    team = Team(id="t", name="T", members=[member], tools=[team_conn])

    await TeamResponse.from_team(team)

    assert member_mcp.connects == 0
    assert team_conn.connects == 0


@pytest.mark.asyncio
async def test_a_run_still_connects_its_tools():
    mcp, conn = MCPTools(), Connectable()
    agent = Agent(id="a", name="A", tools=[mcp, conn])
    run_context = RunContext(run_id="r", session_id="s", session_state={})

    await agent.aget_tools(
        run_response=RunOutput(run_id="r"),
        run_context=run_context,
        session=AgentSession(session_id="s"),
    )

    assert mcp.connects == 1
    assert conn.connects == 1
