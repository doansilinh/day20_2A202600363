"""Benchmark report rendering — markdown + rich terminal output."""

from datetime import datetime, timezone

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a detailed markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        f"**Generated:** {now}",
        "",
        "## Results Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "—" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "—" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f}s | {cost} | {quality} | {item.notes} |"
        )

    # Analysis section if we have 2+ runs
    if len(metrics) >= 2:
        lines += ["", "## Comparative Analysis", ""]
        baseline = next((m for m in metrics if "baseline" in m.run_name), metrics[0])
        multi = next((m for m in metrics if "multi" in m.run_name), metrics[-1])

        q_baseline = baseline.quality_score or 0
        q_multi = multi.quality_score or 0
        quality_delta = q_multi - q_baseline
        latency_ratio = multi.latency_seconds / baseline.latency_seconds if baseline.latency_seconds > 0 else 1.0

        quality_verdict = "✅ Better" if quality_delta > 0 else "⚠️ Worse" if quality_delta < 0 else "≈ Equal"

        lines += [
            f"- **Quality improvement (multi vs baseline):** {quality_delta:+.1f} points — {quality_verdict}",
            f"- **Latency overhead (multi vs baseline):** {latency_ratio:.1f}x",
            "",
            "## Failure Mode Analysis",
            "",
            "| Failure Mode | Description | Fix Applied |",
            "|---|---|---|",
            "| Infinite loop | Agent routes without a stopping condition | `max_iterations` guard in SupervisorAgent |",
            "| Context drift | Long handoff chains dilute original query | Explicit query passed to every agent |",
            "| Missing citations | Writer doesn't include source URLs | `_build_citations()` in WriterAgent |",
            "| Cost explosion | Unbounded token usage per agent | MockLLM with cost tracking per AgentResult |",
            "| Agent crash | Unhandled exception breaks the loop | try/except + state.errors + fallback route |",
            "",
            "## Architecture Trace",
            "",
            "```",
            "User Query",
            "   |",
            "   v",
            "Supervisor  ─────────────────────────────────────┐",
            "   |                                             │",
            "   ├──► Researcher  → sources + research_notes  │",
            "   │         │                                   │",
            "   │         └──► Supervisor (re-route)         │",
            "   │                   │                         │",
            "   ├──► Analyst    → analysis_notes             │",
            "   │         │                                   │",
            "   │         └──► Supervisor (re-route)         │",
            "   │                   │                         │",
            "   ├──► Writer     → final_answer               │",
            "   │         │                                   │",
            "   │         └──► Supervisor (re-route)         │",
            "   │                   │                         │",
            "   ├──► Critic     → quality review             │",
            "   │                   │                         │",
            "   └──────────────────► done ◄──────────────────┘",
            "```",
            "",
            "## Recommendations",
            "",
            "1. Use **single-agent** for simple factual Q&A (lower latency, cheaper)",
            "2. Use **multi-agent** when quality > latency (research synthesis, reports)",
            "3. Always configure `max_iterations` and `timeout_seconds` to prevent runaway costs",
            "4. Invest in tracing infrastructure early — JSON traces provided as minimum baseline",
            "",
        ]

    return "\n".join(lines) + "\n"
