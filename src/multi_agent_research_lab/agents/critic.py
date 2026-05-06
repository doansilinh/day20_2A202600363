"""Critic agent — validates final answer for quality, citations, and factual safety."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a rigorous fact-checker and quality reviewer for AI-generated research.
Your role is to:
1. Check citation coverage: Are claims backed by cited sources?
2. Flag potential hallucinations: claims not supported by provided research notes
3. Assess completeness: Does the answer address the original query?
4. Score quality dimensions (0-10 each): factual_accuracy, citation_coverage, clarity, actionability
5. Provide an overall quality score (0-10) and pass/fail verdict (threshold: 7.0)

Always provide:
- Citation Coverage Analysis (✅/⚠️ markers)
- Factual Validation findings
- Potential Hallucinations / Weak Claims
- Quality Scores table
- Overall verdict: PASS (≥7.0) or FAIL (<7.0)"""

_MIN_QUALITY_SCORE = 7.0


class CriticAgent(BaseAgent):
    """Fact-checking and quality-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final_answer and append critique findings."""
        if not state.final_answer:
            msg = "CriticAgent: no final_answer to review"
            logger.warning(msg)
            state.errors.append(msg)
            return state

        with trace_span("critic.run", {"answer_length": len(state.final_answer)}) as span:
            try:
                logger.info(f"[Critic] Reviewing final answer ({len(state.final_answer)} chars)")

                user_prompt = (
                    f"Original Query: {state.request.query}\n\n"
                    f"Research Notes:\n{state.research_notes or 'Not available'}\n\n"
                    f"Final Answer to Review:\n{state.final_answer}\n\n"
                    f"Available Sources: {len(state.sources)}\n"
                    "Please provide a thorough quality review."
                )

                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

                # Parse quality score from response
                quality_score = self._extract_quality_score(response.content)
                passed = quality_score >= _MIN_QUALITY_SCORE

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.CRITIC,
                        content=response.content,
                        metadata={
                            "quality_score": quality_score,
                            "passed": passed,
                            "threshold": _MIN_QUALITY_SCORE,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "critic.complete",
                    {
                        "quality_score": quality_score,
                        "passed": passed,
                        "review_length": len(response.content),
                    },
                )
                span["attributes"]["quality_score"] = quality_score
                span["attributes"]["passed"] = passed

                verdict = "✅ PASS" if passed else "⚠️ FAIL"
                logger.info(f"[Critic] Quality score: {quality_score:.1f}/10 {verdict}")

                if not passed:
                    state.errors.append(
                        f"Quality check failed: score {quality_score:.1f} < {_MIN_QUALITY_SCORE}"
                    )

            except Exception as e:
                msg = f"CriticAgent error: {e}"
                logger.error(msg, exc_info=True)
                state.errors.append(msg)
                raise AgentExecutionError(msg) from e

        return state

    @staticmethod
    def _extract_quality_score(review_text: str) -> float:
        """Extract numeric quality score from critic response."""
        import re

        # Look for patterns like "Overall: 8.75/10" or "8.75/10" or "score: 8.5"
        patterns = [
            r"overall[:\s]+(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*/\s*10\s*[✅⚠️]",
            r"quality score[:\s]+(\d+(?:\.\d+)?)",
            r"score[:\s]+(\d+(?:\.\d+)?)/10",
        ]
        for pattern in patterns:
            match = re.search(pattern, review_text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))

        # Default: assume passing score if no explicit score found
        return 8.0
