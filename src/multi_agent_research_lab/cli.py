"""Command-line entrypoint for the lab starter."""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

# Force UTF-8 output on Windows to handle unicode characters in LLM responses
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing, trace_span
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()
logger = logging.getLogger(__name__)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ---------------------------------------------------------------------------
# baseline command
# ---------------------------------------------------------------------------

@app.command("baseline")
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: bool = typer.Option(
        True, "--save-trace", help="Save execution trace to JSON"
    ),
) -> None:
    """Run a minimal single-agent baseline."""
    _init()
    console.print(Panel.fit(f"Query: [bold]{query}[/bold]", title="Single-Agent Baseline"))

    tracer = setup_tracing(run_name="baseline") if save_trace else None
    
    from multi_agent_research_lab.services.llm_client import get_llm_client
    with trace_span("baseline.complete") as span:
        llm = get_llm_client()
        system_prompt = (
            "You are a helpful research assistant. "
            "Provide a comprehensive, structured answer to the user's query."
        )
        response = llm.complete(system_prompt, query)
        span["attributes"]["input_tokens"] = response.input_tokens
        span["attributes"]["output_tokens"] = response.output_tokens

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    state.final_answer = response.content

    console.print(Panel(state.final_answer or "", title="Answer"))
    console.print(f"[dim]Tokens: in={response.input_tokens} out={response.output_tokens} "
                  f"cost=${response.cost_usd or 0:.4f}[/dim]")

    if tracer and save_trace:
        from multi_agent_research_lab.evaluation.benchmark import _score_quality
        quality = _score_quality(state)
        
        trace_path = tracer.flush(
            state_summary={
                "query": query,
                "mode": "baseline",
                "quality_score": quality,
                "tokens": {"in": response.input_tokens, "out": response.output_tokens},
                "final_answer_length": len(state.final_answer or ""),
            }
        )
        console.print(f"\n[dim]Trace saved to: {trace_path}[/dim]")


# ---------------------------------------------------------------------------
# multi-agent command
# ---------------------------------------------------------------------------

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: bool = typer.Option(
        True, "--save-trace", help="Save full execution trace to JSON"
    ),
) -> None:
    """Run the full multi-agent workflow (Supervisor→Researcher→Analyst→Writer→Critic)."""
    _init()
    console.print(Panel.fit(f"Query: [bold]{query}[/bold]", title="Multi-Agent Workflow"))

    tracer = setup_tracing(run_name="multi_agent") if save_trace else None
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()

    with console.status("[bold green]Running multi-agent pipeline...[/bold green]"):
        result = workflow.run(state)

    # Display results
    _print_route_history(result)

    if result.final_answer:
        console.print(Panel(result.final_answer, title="[green]Final Answer[/green]"))
    else:
        console.print("[yellow]No final answer generated[/yellow]")

    if result.errors:
        console.print(f"[red]Errors: {result.errors}[/red]")

    # Save trace — flush state.trace events + agent_results into JsonFileTracer
    if tracer and save_trace:
        # Record all state trace events as spans
        tracer.record_trace_events(result.trace)
        # Record per-agent metadata as structured spans
        for ar in result.agent_results:
            tracer.record_span({
                "type": "agent_result",
                "agent": ar.agent,
                "output_length": len(ar.content),
                "metadata": ar.metadata,
            })

        from multi_agent_research_lab.evaluation.benchmark import _score_quality
        quality = _score_quality(result)

        trace_path = tracer.flush(
            state_summary={
                "query": query,
                "mode": "multi_agent",
                "quality_score": quality,
                "route_history": result.route_history,
                "iterations": result.iteration,
                "sources_count": len(result.sources),
                "sources": [{"title": s.title, "url": s.url} for s in result.sources],
                "research_notes_length": len(result.research_notes or ""),
                "analysis_notes_length": len(result.analysis_notes or ""),
                "final_answer_length": len(result.final_answer or ""),
                "has_answer": result.final_answer is not None,
                "errors": result.errors,
                "agent_results_count": len(result.agent_results),
            }
        )
        console.print(f"\n[dim]Trace saved to: {trace_path}[/dim]")

    # Print trace events summary
    if result.trace:
        console.print(f"\n[dim]Trace events: {len(result.trace)} | Agent results: {len(result.agent_results)}[/dim]")


def _print_route_history(result: ResearchState) -> None:
    table = Table(title="Agent Execution Trace", show_header=True, header_style="bold blue")
    table.add_column("Step", style="dim", width=6)
    table.add_column("Route", width=16)
    table.add_column("Status", width=10)

    for i, route in enumerate(result.route_history, 1):
        status = "[green]OK[/green]" if route != "done" else "[bold]DONE[/bold]"
        table.add_row(str(i), route, status)

    console.print(table)


if __name__ == "__main__":
    app()
