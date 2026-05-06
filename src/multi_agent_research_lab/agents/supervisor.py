"""Supervisor / router — decides which worker runs next and when to stop.

Routing policy:
  start  →  researcher   (always gather sources first)
  researcher  →  analyst  (after research_notes are populated)
  analyst  →  writer      (after analysis_notes are populated)
  writer  →  critic       (quality gate — optional)
  critic  →  done         (after final_answer passes review)
  any     →  done         (on max_iterations or unrecoverable error)
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Routing constants
_ROUTE_RESEARCHER = AgentName.RESEARCHER
_ROUTE_ANALYST = AgentName.ANALYST
_ROUTE_WRITER = AgentName.WRITER
_ROUTE_CRITIC = AgentName.CRITIC
_ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update state.route_history with the next route.

        Routing logic:
        1. Enforce max_iterations guard (hard stop).
        2. If any unrecoverable errors → done (fail-safe).
        3. Rule-based sequential routing based on what's missing.
        4. All work complete → done.
        """
        settings = get_settings()

        with trace_span("supervisor.route", {"iteration": state.iteration}) as span:
            next_route = self._decide_route(state, settings.max_iterations)
            span["attributes"]["next_route"] = next_route

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.decision",
            {
                "iteration": state.iteration,
                "next_route": next_route,
                "has_sources": len(state.sources) > 0,
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
                "error_count": len(state.errors),
            },
        )
        logger.info(
            f"[Supervisor] iter={state.iteration} route={next_route} "
            f"sources={len(state.sources)} errors={len(state.errors)}"
        )
        return state

    def _decide_route(self, state: ResearchState, max_iterations: int) -> str:
        """Pure routing decision — no side effects."""

        # Guard 1: Max iterations hard stop
        if state.iteration >= max_iterations:
            logger.warning(
                f"[Supervisor] Max iterations ({max_iterations}) reached — forcing done"
            )
            if state.final_answer is None:
                state.errors.append(f"Stopped after {max_iterations} iterations without completing")
            return _ROUTE_DONE

        # Guard 2: Unrecoverable error accumulation (>3 errors)
        if len(state.errors) > 3:
            logger.error(f"[Supervisor] Too many errors ({len(state.errors)}) — aborting")
            return _ROUTE_DONE

        # Sequential routing based on state completeness
        if not state.sources and state.research_notes is None:
            # Phase 1: Need research
            return _ROUTE_RESEARCHER

        if state.research_notes is not None and state.analysis_notes is None:
            # Phase 2: Research done, need analysis
            return _ROUTE_ANALYST

        if state.analysis_notes is not None and state.final_answer is None:
            # Phase 3: Analysis done, need writing
            return _ROUTE_WRITER

        if state.final_answer is not None:
            # All phases complete
            return _ROUTE_DONE

        # Fallback: restart from research (shouldn't reach here normally)
        logger.warning("[Supervisor] Unexpected state — routing back to researcher")
        return _ROUTE_RESEARCHER
