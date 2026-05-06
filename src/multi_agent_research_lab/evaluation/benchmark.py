"""Benchmark for single-agent vs multi-agent comparison."""

import re
import time
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

# Keywords expected in good research outputs
_QUALITY_KEYWORDS = [
    "research", "analysis", "summary", "findings", "approach", "recommendation",
    "limitation", "benchmark", "performance", "evidence", "conclusion",
]

# Citation/reference markers
_CITATION_PATTERNS = [r"\[\d+\]", r"https?://", r"\bReference", r"\bSource"]


def _score_quality(state: ResearchState) -> float:
    """Heuristic quality scoring (0-10) based on output richness.

    Criteria:
    - Answer presence: 0 or 4 pts
    - Word count (≥300 = full marks): up to 2 pts
    - Keyword coverage (12 keywords): up to 2 pts
    - Citation count: up to 2 pts
    """
    if not state.final_answer:
        return 0.0

    answer = state.final_answer
    score = 4.0  # Base: answer exists

    # Word count score (max 2)
    word_count = len(answer.split())
    word_score = min(2.0, word_count / 300 * 2)
    score += word_score

    # Keyword coverage (max 2)
    answer_lower = answer.lower()
    hits = sum(1 for kw in _QUALITY_KEYWORDS if kw in answer_lower)
    keyword_score = min(2.0, hits / len(_QUALITY_KEYWORDS) * 2)
    score += keyword_score

    # Citation count (max 2)
    citation_count = sum(
        len(re.findall(p, answer)) for p in _CITATION_PATTERNS
    )
    citation_score = min(2.0, citation_count / 3 * 2)
    score += citation_score

    return round(min(10.0, score), 2)


def _estimate_total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd across all agent results."""
    costs = [
        r.metadata.get("cost_usd", 0.0)
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    return round(sum(costs), 6) if costs else None


def _count_citations(state: ResearchState) -> int:
    """Count citation markers in final answer."""
    if not state.final_answer:
        return 0
    return sum(len(re.findall(p, state.final_answer)) for p in _CITATION_PATTERNS)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, quality, and cost for a single run.

    Returns (final_state, BenchmarkMetrics).
    """
    started = time.perf_counter()
    state = runner(query)
    latency = time.perf_counter() - started

    quality = _score_quality(state)
    cost = _estimate_total_cost(state)
    citations = _count_citations(state)
    agents_used = sorted({r.agent.value for r in state.agent_results})
    error_note = f" | errors={len(state.errors)}" if state.errors else ""
    notes = f"agents={agents_used} citations={citations}{error_note}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=notes,
    )
    return state, metrics


def compare_runs(
    query: str,
    baseline_runner: Runner,
    multi_agent_runner: Runner,
) -> tuple[BenchmarkMetrics, BenchmarkMetrics]:
    """Run both baselines and return their metrics for comparison."""
    _, baseline_metrics = run_benchmark("single_agent_baseline", query, baseline_runner)
    _, multi_metrics = run_benchmark("multi_agent", query, multi_agent_runner)
    return baseline_metrics, multi_metrics
