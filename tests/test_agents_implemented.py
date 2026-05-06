"""Integration tests for agent implementations (no API key required)."""

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(query: str = "Explain multi-agent systems and GraphRAG") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


# ---------------------------------------------------------------------------
# SupervisorAgent tests
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    def test_routes_to_researcher_when_empty(self) -> None:
        state = _make_state()
        result = SupervisorAgent().run(state)
        assert "researcher" in result.route_history

    def test_routes_to_analyst_after_research(self) -> None:
        state = _make_state()
        state.research_notes = "Some research notes"
        result = SupervisorAgent().run(state)
        assert "analyst" in result.route_history

    def test_routes_to_writer_after_analysis(self) -> None:
        state = _make_state()
        state.research_notes = "Research notes"
        state.analysis_notes = "Analysis notes"
        result = SupervisorAgent().run(state)
        assert "writer" in result.route_history

    def test_routes_done_when_complete(self) -> None:
        state = _make_state()
        state.research_notes = "Notes"
        state.analysis_notes = "Analysis"
        state.final_answer = "Final answer"
        result = SupervisorAgent().run(state)
        assert "done" in result.route_history

    def test_enforces_max_iterations(self) -> None:
        state = _make_state()
        state.iteration = 100  # Simulate max exceeded
        result = SupervisorAgent().run(state)
        assert "done" in result.route_history

    def test_records_trace_event(self) -> None:
        state = _make_state()
        result = SupervisorAgent().run(state)
        event_names = [e["name"] for e in result.trace]
        assert "supervisor.decision" in event_names

    def test_increments_iteration(self) -> None:
        state = _make_state()
        initial_iter = state.iteration
        result = SupervisorAgent().run(state)
        assert result.iteration == initial_iter + 1


# ---------------------------------------------------------------------------
# ResearcherAgent tests
# ---------------------------------------------------------------------------

class TestResearcherAgent:
    def test_populates_sources(self) -> None:
        state = _make_state("GraphRAG and multi-hop retrieval")
        result = ResearcherAgent().run(state)
        assert len(result.sources) > 0

    def test_populates_research_notes(self) -> None:
        state = _make_state()
        result = ResearcherAgent().run(state)
        assert result.research_notes is not None
        assert len(result.research_notes) > 50

    def test_records_agent_result(self) -> None:
        state = _make_state()
        result = ResearcherAgent().run(state)
        agent_names = [r.agent for r in result.agent_results]
        assert "researcher" in agent_names

    def test_sources_have_required_fields(self) -> None:
        state = _make_state("LangGraph workflow")
        result = ResearcherAgent().run(state)
        for src in result.sources:
            assert src.title
            assert src.snippet


# ---------------------------------------------------------------------------
# AnalystAgent tests
# ---------------------------------------------------------------------------

class TestAnalystAgent:
    def test_populates_analysis_notes(self) -> None:
        state = _make_state()
        state.research_notes = (
            "GraphRAG improves multi-hop QA by 15-25% over flat RAG. "
            "Multi-agent systems add 3x cost but improve quality by 30%."
        )
        result = AnalystAgent().run(state)
        assert result.analysis_notes is not None
        assert len(result.analysis_notes) > 50

    def test_handles_missing_research_notes(self) -> None:
        state = _make_state()
        # No research_notes set
        result = AnalystAgent().run(state)
        # Should not crash — should set analysis_notes to a fallback
        assert result.analysis_notes is not None

    def test_records_agent_result(self) -> None:
        state = _make_state()
        state.research_notes = "Some research notes about RAG"
        result = AnalystAgent().run(state)
        agent_names = [r.agent for r in result.agent_results]
        assert "analyst" in agent_names


# ---------------------------------------------------------------------------
# WriterAgent tests
# ---------------------------------------------------------------------------

class TestWriterAgent:
    def test_populates_final_answer(self) -> None:
        state = _make_state()
        state.research_notes = "Research on GraphRAG shows 15-25% improvement in multi-hop QA."
        state.analysis_notes = "Evidence is strong; recommend GraphRAG for knowledge-intensive tasks."
        result = WriterAgent().run(state)
        assert result.final_answer is not None
        assert len(result.final_answer) > 100

    def test_handles_no_notes(self) -> None:
        state = _make_state()
        result = WriterAgent().run(state)
        assert result.final_answer is not None  # Fallback message

    def test_records_agent_result(self) -> None:
        state = _make_state()
        state.research_notes = "Notes"
        state.analysis_notes = "Analysis"
        result = WriterAgent().run(state)
        agent_names = [r.agent for r in result.agent_results]
        assert "writer" in agent_names


# ---------------------------------------------------------------------------
# CriticAgent tests
# ---------------------------------------------------------------------------

class TestCriticAgent:
    def test_runs_review_on_final_answer(self) -> None:
        state = _make_state()
        state.final_answer = "GraphRAG improves multi-hop QA by 15-25%. See [1] Microsoft GraphRAG."
        state.research_notes = "Source: Microsoft GraphRAG paper"
        result = CriticAgent().run(state)
        assert any(r.agent == "critic" for r in result.agent_results)

    def test_handles_missing_final_answer(self) -> None:
        state = _make_state()
        result = CriticAgent().run(state)
        # Should not crash, should add to errors
        assert len(result.errors) > 0

    def test_quality_score_in_metadata(self) -> None:
        state = _make_state()
        state.final_answer = "Detailed answer with [1] citation. https://example.com reference."
        result = CriticAgent().run(state)
        critic_results = [r for r in result.agent_results if r.agent == "critic"]
        if critic_results:
            assert "quality_score" in critic_results[0].metadata
            score = critic_results[0].metadata["quality_score"]
            assert 0 <= score <= 10
