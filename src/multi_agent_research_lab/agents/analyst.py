"""Analyst agent — turns research notes into structured analytical insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior research analyst. Your role is to:
1. Extract and enumerate key claims from research notes
2. Assess the strength of evidence for each claim (Strong / Moderate / Weak)
3. Compare different viewpoints and highlight contradictions
4. Flag weak evidence, unsupported assertions, and potential hallucinations
5. Create a structured comparison table where applicable
6. Provide a final recommendation based on the evidence

Structure your output with clear sections:
- Key Claims Identified
- Comparative Analysis (table if multiple options)
- Weak Evidence / Red Flags
- Recommended Approach"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.analysis_notes from research_notes."""
        if not state.research_notes:
            msg = "AnalystAgent: no research_notes to analyze"
            logger.warning(msg)
            state.errors.append(msg)
            state.analysis_notes = "No research notes available for analysis."
            return state

        with trace_span("analyst.run", {"notes_length": len(state.research_notes)}) as span:
            try:
                logger.info(f"[Analyst] Analyzing {len(state.research_notes)} chars of research notes")

                user_prompt = (
                    f"Original query: {state.request.query}\n\n"
                    f"Target audience: {state.request.audience}\n\n"
                    f"Research Notes to Analyze:\n{state.research_notes}\n\n"
                    "Please provide a thorough analytical assessment."
                )

                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                state.analysis_notes = response.content

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.ANALYST,
                        content=response.content,
                        metadata={
                            "input_length": len(state.research_notes),
                            "output_length": len(response.content),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "analyst.complete",
                    {
                        "analysis_length": len(state.analysis_notes),
                        "cost_usd": response.cost_usd,
                    },
                )
                span["attributes"]["analysis_length"] = len(state.analysis_notes)
                logger.info(f"[Analyst] Generated {len(state.analysis_notes)} chars of analysis")

            except Exception as e:
                msg = f"AnalystAgent error: {e}"
                logger.error(msg, exc_info=True)
                state.errors.append(msg)
                raise AgentExecutionError(msg) from e

        return state
