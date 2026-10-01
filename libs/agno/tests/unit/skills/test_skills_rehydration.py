"""Skills round-trip through a stored agent or team config.

to_dict stores named Skills as a ``{"name": ...}`` reference and from_dict
resolves it from the Registry, like learning and knowledge. A stored component
must come back with the same skills in its system message and tools; a missing
registry entry refuses under strict and degrades lenient; a config stored
before skills were saved loads with no skills.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agno.agent import Agent
from agno.agent._tools import get_tools
from agno.exceptions import ComponentRehydrationError
from agno.models.base import Function
from agno.os.utils import collect_components_from_agent, collect_components_from_team
from agno.registry import Registry
from agno.run.agent import RunOutput
from agno.run.base import RunContext
from agno.session import AgentSession, TeamSession
from agno.skills import LocalSkills, Skills
from agno.team import Team
from agno.team._messages import get_system_message as get_team_system_message

SKILL_TOOL_NAMES = {"get_skill_instructions", "get_skill_reference", "get_skill_script"}


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.get_instructions_for_model = MagicMock(return_value=None)
    model.get_system_message_for_model = MagicMock(return_value=None)
    return model


def _agent_system_message(agent: Agent) -> str:
    agent.model = _mock_model()
    message = agent.get_system_message(session=AgentSession(session_id="s"))
    assert message is not None
    return str(message.content)


def _agent_skill_tool_names(agent: Agent) -> set:
    tools = get_tools(
        agent,
        RunOutput(run_id="r", session_id="s", agent_id=agent.id),
        RunContext(run_id="r", session_id="s"),
        AgentSession(session_id="s"),
    )
    return {t.name for t in tools if isinstance(t, Function) and t.name in SKILL_TOOL_NAMES}


@pytest.fixture
def skills(temp_skills_dir: Path) -> Skills:
    return Skills(loaders=[LocalSkills(str(temp_skills_dir))], name="review-skills")


def test_agent_round_trip_restores_the_registered_skills(skills: Skills) -> None:
    original = Agent(id="reviewer", name="Reviewer", skills=skills)
    config = original.to_dict()
    assert config["skills"] == {"name": "review-skills"}

    restored = Agent.from_dict(config, registry=Registry(skills=[skills]), strict=True)

    assert restored.skills is skills
    original_message = _agent_system_message(original)
    assert "<skills_system>" in original_message
    assert _agent_system_message(restored) == original_message
    assert _agent_skill_tool_names(restored) == _agent_skill_tool_names(original) == SKILL_TOOL_NAMES


def test_agent_missing_registry_entry_refuses_strict_and_degrades_lenient(skills: Skills) -> None:
    config = Agent(id="reviewer", skills=skills).to_dict()

    for registry in (None, Registry()):
        with pytest.raises(ComponentRehydrationError, match="review-skills"):
            Agent.from_dict(config, registry=registry, strict=True)
        assert Agent.from_dict(config, registry=registry, strict=False).skills is None


def test_agent_ambiguous_name_refuses_strict_and_binds_first_lenient(skills: Skills, temp_skills_dir: Path) -> None:
    other = Skills(loaders=[LocalSkills(str(temp_skills_dir))], name="review-skills")
    registry = Registry(skills=[skills, other])
    config = Agent(id="reviewer", skills=skills).to_dict()

    with pytest.raises(ComponentRehydrationError, match="two distinct Skills"):
        Agent.from_dict(config, registry=registry, strict=True)
    assert Agent.from_dict(config, registry=registry, strict=False).skills is skills


def test_agent_config_stored_without_skills_loads_with_none(skills: Skills) -> None:
    config = Agent(id="reviewer", name="Reviewer").to_dict()
    assert "skills" not in config

    restored = Agent.from_dict(config, registry=Registry(skills=[skills]), strict=True)

    assert restored.skills is None


def test_unnamed_skills_are_not_saved(temp_skills_dir: Path) -> None:
    unnamed = Skills(loaders=[LocalSkills(str(temp_skills_dir))])

    assert "skills" not in Agent(id="reviewer", skills=unnamed).to_dict()
    assert "skills" not in Team(id="review-team", members=[], skills=unnamed).to_dict()


def test_team_round_trip_restores_the_registered_skills(skills: Skills) -> None:
    original = Team(id="review-team", name="Review Team", mode="coordinate", members=[], skills=skills)
    config = original.to_dict()
    assert config["skills"] == {"name": "review-skills"}

    restored = Team.from_dict(config, registry=Registry(skills=[skills]), strict=True)

    assert restored.skills is skills
    original.model = _mock_model()
    restored.model = _mock_model()
    original_message = get_team_system_message(original, TeamSession(session_id="s"))
    restored_message = get_team_system_message(restored, TeamSession(session_id="s"))
    assert original_message is not None and restored_message is not None
    assert "<skills_system>" in str(original_message.content)
    assert restored_message.content == original_message.content


def test_team_missing_registry_entry_refuses_strict_and_degrades_lenient(skills: Skills) -> None:
    config = Team(id="review-team", members=[], skills=skills).to_dict()

    with pytest.raises(ComponentRehydrationError, match="review-skills"):
        Team.from_dict(config, registry=Registry(), strict=True)
    assert Team.from_dict(config, registry=Registry(), strict=False).skills is None


def test_component_collection_registers_named_skills(skills: Skills, temp_skills_dir: Path) -> None:
    unnamed = Skills(loaders=[LocalSkills(str(temp_skills_dir))])
    registry = Registry()

    collect_components_from_agent(Agent(id="reviewer", skills=skills), registry, set())
    collect_components_from_team(Team(id="review-team", members=[], skills=unnamed), registry, set())

    assert registry.skills == [skills]
    assert registry.get_skills("review-skills") is skills
