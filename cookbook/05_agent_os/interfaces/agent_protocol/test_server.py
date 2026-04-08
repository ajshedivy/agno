"""Test server with a simple agent for Agent Protocol API testing.
No external API keys required -- uses a basic agent with no model that
returns canned responses via mocking.

Usage:
    python test_server.py

Then run the curl tests:
    bash test_curl.sh
"""

from typing import Any, AsyncIterator, Iterator, List

from agno.agent.agent import Agent
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse, ModelResponseEvent
from agno.os import AgentOS


class EchoModel(Model):
    """A model that echoes back the user's message. No API keys needed."""

    name: str = "EchoModel"

    def __init__(self):
        super().__init__(id="echo-model")

    def invoke(self, messages: List[Message], **kwargs) -> ModelResponse:
        last_msg = ""
        for m in reversed(messages):
            if m.role == "user" and m.content:
                last_msg = m.content if isinstance(m.content, str) else str(m.content)
                break
        return ModelResponse(
            content=f"Echo: {last_msg}",
            response_usage=None,
        )

    async def ainvoke(self, messages: List[Message], **kwargs) -> ModelResponse:
        return self.invoke(messages, **kwargs)

    def invoke_stream(
        self, messages: List[Message], **kwargs
    ) -> Iterator[ModelResponseEvent]:
        resp = self.invoke(messages, **kwargs)
        yield ModelResponseEvent(content=resp.content)

    async def ainvoke_stream(
        self, messages: List[Message], **kwargs
    ) -> AsyncIterator[ModelResponseEvent]:
        resp = self.invoke(messages, **kwargs)
        yield ModelResponseEvent(content=resp.content)

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return (
            response
            if isinstance(response, ModelResponse)
            else ModelResponse(content=str(response))
        )

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return (
            response
            if isinstance(response, ModelResponse)
            else ModelResponse(content=str(response))
        )


# ---------------------------------------------------------------------------
# Create Agents
# ---------------------------------------------------------------------------

echo_agent = Agent(
    name="echo-agent",
    model=EchoModel(),
    id="echo_agent",
    description="An echo agent that repeats your message. For testing purposes.",
    instructions="You are an echo agent. Repeat whatever the user says.",
)

research_agent = Agent(
    name="research-agent",
    model=EchoModel(),
    id="research_agent",
    description="A research agent for testing.",
    instructions="You are a research agent.",
)

# ---------------------------------------------------------------------------
# Setup AgentOS
# ---------------------------------------------------------------------------

agent_os = AgentOS(
    agents=[echo_agent, research_agent],
    agent_protocol_interface=True,
)
app = agent_os.get_app()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Agent Protocol test server on http://localhost:7778")
    print("Agents: echo_agent, research_agent")
    print("Interface: /ap/*")
    print()
    print("Test with:")
    print("  curl -s http://localhost:7778/ap/agents/search -X POST -d '{}'")
    print("  bash test_curl.sh")
    agent_os.serve(app="test_server:app", reload=False, port=7778)
