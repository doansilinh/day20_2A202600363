"""End-to-end workflow and benchmark tests."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import _score_quality, run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def _make_state(query: str) -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


# ---------------------------------------------------------------------------
# MultiAgentWorkflow tests
# ---------------------------------------------------------------------------

class TestMultiAgentWorkflow:
    def test_workflow_completes(self) -> None:
        state = _make_state("Explain GraphRAG and multi-agent systems")
        workflow = MultiAgentWorkflow()
        result = workflow.run(state)
        assert result.final_answer is not None

    def test_workflow_produces_research_notes(self) -> None:
        state = _make_state("Research LangGraph orchestration patterns")
        result = MultiAgentWorkflow().run(state)
        assert result.research_notes is not None

    def test_workflow_produces_analysis_notes(self) -> None:
        state = _make_state("Analyze RAG vs GraphRAG trade-offs")
        result = MultiAgentWorkflow().run(state)
        assert result.analysis_notes is not None

    def test_workflow_route_history_ordered(self) -> None:
        state = _make_state("Multi-agent research on vector databases")
        result = MultiAgentWorkflow().run(state)
        # Route history must contain researcher before analyst before writer
        history = result.route_history
        assert "researcher" in history
        assert "analyst" in history
        assert "writer" in history
        ri = history.index("researcher")
        ai = history.index("analyst")
        wi = history.index("writer")
        assert ri < ai < wi

    def test_workflow_ends_with_done(self) -> None:
        state = _make_state("What is LangGraph?")
        result = MultiAgentWorkflow().run(state)
        assert result.route_history[-1] == "done"

    def test_workflow_records_trace_events(self) -> None:
        state = _make_state("GraphRAG benchmark results")
        result = MultiAgentWorkflow().run(state)
        assert len(result.trace) > 0

    def test_workflow_has_multiple_agent_results(self) -> None:
        state = _make_state("Compare multi-agent vs single-agent systems")
        result = MultiAgentWorkflow().run(state)
        agents = {r.agent for r in result.agent_results}
        assert "researcher" in agents
        assert "analyst" in agents
        assert "writer" in agents


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmark:
    def _simple_runner(self, query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.final_answer = (
            f"Answer to: {query}. "
            "GraphRAG improves retrieval by 20%. "
            "Multi-agent systems improve quality by 30%. "
            "Reference: [1] https://arxiv.org/abs/2404.16130"
        )
        return state

    def test_run_benchmark_returns_metrics(self) -> None:
        state, metrics = run_benchmark("test", "GraphRAG", self._simple_runner)
        assert metrics.run_name == "test"
        assert metrics.latency_seconds >= 0
        assert metrics.quality_score is not None

    def test_quality_score_range(self) -> None:
        state = ResearchState(request=ResearchQuery(query="GraphRAG"))
        state.final_answer = "A good answer with research and analysis findings."
        score = _score_quality(state)
        assert 0 <= score <= 10

    def test_quality_score_zero_for_empty_answer(self) -> None:
        state = ResearchState(request=ResearchQuery(query="GraphRAG"))
        assert _score_quality(state) == 0.0

    def test_quality_score_higher_for_rich_answer(self) -> None:
        state_simple = ResearchState(request=ResearchQuery(query="Query"))
        state_simple.final_answer = "Short answer"

        state_rich = ResearchState(request=ResearchQuery(query="Query"))
        state_rich.final_answer = (
            "# Research Summary\n\n"
            "## Findings\nGraphRAG improves multi-hop retrieval by 15-25%. "
            "Multi-agent systems provide better analysis and synthesis. "
            "The benchmark results show significant quality improvements. "
            "Recommendations: use multi-agent for complex research tasks. "
            "Limitations: higher cost and latency overhead.\n\n"
            "## References\n[1] https://arxiv.org/abs/2404.16130\n"
            "[2] https://langchain-ai.github.io/langgraph/\n"
            "[3] https://www.anthropic.com/engineering/building-effective-agents\n"
            " " * 500  # Pad to ensure word count threshold met
        )
        assert _score_quality(state_rich) > _score_quality(state_simple)


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_render_with_comparison(self) -> None:
        metrics = [
            BenchmarkMetrics(
                run_name="single_agent_baseline",
                latency_seconds=1.5,
                estimated_cost_usd=0.001,
                quality_score=6.0,
                notes="Simple baseline",
            ),
            BenchmarkMetrics(
                run_name="multi_agent",
                latency_seconds=5.2,
                estimated_cost_usd=0.003,
                quality_score=8.5,
                notes="Full pipeline",
            ),
        ]
        report = render_markdown_report(metrics)
        assert "Benchmark Report" in report
        assert "single_agent_baseline" in report
        assert "multi_agent" in report
        assert "Comparative Analysis" in report
        assert "Failure Mode Analysis" in report

    def test_quality_delta_calculation(self) -> None:
        metrics = [
            BenchmarkMetrics(run_name="single_agent_baseline", latency_seconds=1.0, quality_score=6.0),
            BenchmarkMetrics(run_name="multi_agent", latency_seconds=4.0, quality_score=8.0),
        ]
        report = render_markdown_report(metrics)
        # Delta should be +2.0
        assert "+2.0" in report
