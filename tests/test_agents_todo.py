"""Legacy skeleton tests — updated to reflect implemented agents."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_without_error() -> None:
    """SupervisorAgent is now implemented — should not raise StudentTodoError."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    # Should route to researcher (first step in pipeline)
    assert "researcher" in result.route_history
