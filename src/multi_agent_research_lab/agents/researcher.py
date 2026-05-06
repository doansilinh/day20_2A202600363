"""Researcher agent — collects sources and synthesizes research notes."""

import logging

from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import get_llm_client
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert research assistant. Your role is to:
1. Analyze source documents retrieved from a search
2. Extract the most relevant, accurate information
3. Synthesize concise research notes with key findings, evidence quality, and identified gaps
4. Structure notes clearly with sections: Key Findings, Source Coverage, Gaps Identified
Always cite specific sources when referencing claims."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = get_llm_client()
        self._search = SearchClient()

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=False)
    def _safe_search(self, query: str, max_sources: int) -> list:  # type: ignore[type-arg]
        return self._search.search(query, max_results=max_sources)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes."""
        query = state.request.query
        max_sources = state.request.max_sources

        with trace_span("researcher.run", {"query": query[:80]}) as span:
            try:
                # Step 1: Search for relevant sources
                logger.info(f"[Researcher] Searching for: '{query[:60]}'")
                sources = self._safe_search(query, max_sources)
                if sources is None:
                    sources = []
                state.sources = sources
                span["attributes"]["sources_found"] = len(sources)
                logger.info(f"[Researcher] Found {len(sources)} sources")

                # Step 2: Build context from sources
                source_context = self._build_source_context(sources)

                # Step 3: Generate research notes via LLM
                user_prompt = (
                    f"Query: {query}\n\n"
                    f"Sources:\n{source_context}\n\n"
                    "Please create structured research notes based on these sources."
                )
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                state.research_notes = response.content

                # Step 4: Record result
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.RESEARCHER,
                        content=response.content,
                        metadata={
                            "sources_count": len(sources),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "researcher.complete",
                    {
                        "sources_count": len(sources),
                        "notes_length": len(state.research_notes),
                        "cost_usd": response.cost_usd,
                    },
                )
                span["attributes"]["notes_length"] = len(state.research_notes)

            except RetryError as e:
                msg = f"Researcher search failed after retries: {e}"
                logger.error(msg)
                state.errors.append(msg)
                # Provide a minimal fallback note so the pipeline can continue
                state.research_notes = f"Research failed for query: {query}. Error: {e}"
            except Exception as e:
                msg = f"Researcher unexpected error: {e}"
                logger.error(msg, exc_info=True)
                state.errors.append(msg)
                raise AgentExecutionError(msg) from e

        return state

    @staticmethod
    def _build_source_context(sources: list) -> str:  # type: ignore[type-arg]
        """Format sources into a compact context string for the LLM."""
        if not sources:
            return "No sources found."
        parts = []
        for i, src in enumerate(sources, 1):
            parts.append(
                f"[{i}] {src.title}\n"
                f"URL: {src.url or 'N/A'}\n"
                f"Snippet: {src.snippet}"
            )
        return "\n\n".join(parts)
