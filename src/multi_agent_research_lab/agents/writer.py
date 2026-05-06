"""Writer agent — synthesizes final answer from research and analysis notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert technical writer producing a comprehensive research summary.
Your output should be:
1. Well-structured with clear sections (Executive Summary, Core Findings, Benchmarks, Limitations, Recommendations, References)
2. Written for the specified target audience
3. ~500 words minimum, substantive and actionable
4. Include a References section citing the sources provided
5. Use markdown formatting with headers, bold text, and tables where appropriate
6. Avoid unsupported claims — ground every assertion in the research and analysis provided

Always end with a numbered References section."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.final_answer by synthesizing research + analysis."""
        if not state.research_notes and not state.analysis_notes:
            msg = "WriterAgent: no notes available to write from"
            logger.warning(msg)
            state.errors.append(msg)
            state.final_answer = f"Unable to generate answer for: {state.request.query}"
            return state

        with trace_span("writer.run") as span:
            try:
                # Build source citations
                citations = self._build_citations(state)
                context_parts = []

                if state.research_notes:
                    context_parts.append(f"## Research Notes\n{state.research_notes}")
                if state.analysis_notes:
                    context_parts.append(f"## Analytical Assessment\n{state.analysis_notes}")

                context = "\n\n".join(context_parts)

                user_prompt = (
                    f"Query: {state.request.query}\n"
                    f"Target audience: {state.request.audience}\n\n"
                    f"{context}\n\n"
                    f"Available sources for citation:\n{citations}\n\n"
                    "Write a comprehensive, well-structured research summary (~500+ words)."
                )

                logger.info(f"[Writer] Synthesizing final answer from {len(context)} chars of context")
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                state.final_answer = response.content

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=response.content,
                        metadata={
                            "context_length": len(context),
                            "answer_length": len(response.content),
                            "sources_cited": len(state.sources),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "writer.complete",
                    {
                        "answer_length": len(state.final_answer),
                        "cost_usd": response.cost_usd,
                    },
                )
                span["attributes"]["answer_length"] = len(state.final_answer)
                logger.info(f"[Writer] Generated final answer ({len(state.final_answer)} chars)")

            except Exception as e:
                msg = f"WriterAgent error: {e}"
                logger.error(msg, exc_info=True)
                state.errors.append(msg)
                raise AgentExecutionError(msg) from e

        return state

    @staticmethod
    def _build_citations(state: ResearchState) -> str:
        """Format available sources as a numbered citation list."""
        if not state.sources:
            return "No sources available."
        lines = []
        for i, src in enumerate(state.sources, 1):
            url_part = f" — {src.url}" if src.url else ""
            lines.append(f"[{i}] {src.title}{url_part}")
        return "\n".join(lines)
