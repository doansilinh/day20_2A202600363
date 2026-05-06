"""LangGraph workflow — builds and runs the multi-agent graph.

Graph topology:
  [START] → supervisor → {researcher|analyst|writer|critic|END}
  Each worker node feeds back to supervisor for re-routing.
"""

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Try to import LangGraph; fall back to a pure Python loop if not installed
try:
    from langgraph.graph import END, START, StateGraph  # type: ignore[import]

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed — using built-in Python loop fallback")


# ---------------------------------------------------------------------------
# Node wrappers (convert ResearchState ↔ LangGraph dict state)
# ---------------------------------------------------------------------------

def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    return state.model_dump()


def _dict_to_state(d: dict[str, Any]) -> ResearchState:
    return ResearchState.model_validate(d)


def _make_supervisor_node(supervisor: SupervisorAgent):  # type: ignore[no-untyped-def]
    def _node(state_dict: dict[str, Any]) -> dict[str, Any]:
        state = _dict_to_state(state_dict)
        updated = supervisor.run(state)
        return _state_to_dict(updated)
    return _node


def _make_worker_node(agent: Any):  # type: ignore[no-untyped-def]
    def _node(state_dict: dict[str, Any]) -> dict[str, Any]:
        state = _dict_to_state(state_dict)
        updated = agent.run(state)
        return _state_to_dict(updated)
    return _node


def _route_after_supervisor(state_dict: dict[str, Any]) -> str:
    """Conditional edge: read last route decision from state."""
    route_history = state_dict.get("route_history", [])
    if not route_history:
        return "researcher"
    last_route = route_history[-1]
    if last_route == "done":
        return END
    return last_route


# ---------------------------------------------------------------------------
# Main workflow class
# ---------------------------------------------------------------------------

class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()
        self._settings = get_settings()

    def build(self) -> Any:
        """Create and compile a LangGraph StateGraph (or None if not available)."""
        if not _LANGGRAPH_AVAILABLE:
            logger.info("LangGraph not available; build() returns None (will use Python loop)")
            return None

        logger.info("Building LangGraph StateGraph")

        # Use dict-based state (compatible with all LangGraph versions)
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("supervisor", _make_supervisor_node(self._supervisor))
        graph.add_node("researcher", _make_worker_node(self._researcher))
        graph.add_node("analyst", _make_worker_node(self._analyst))
        graph.add_node("writer", _make_worker_node(self._writer))
        graph.add_node("critic", _make_worker_node(self._critic))

        # Entry point
        graph.add_edge(START, "supervisor")

        # Conditional routing from supervisor
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # Each worker loops back to supervisor
        for worker in ["researcher", "analyst", "writer", "critic"]:
            graph.add_edge(worker, "supervisor")

        compiled = graph.compile()
        logger.info("LangGraph compiled successfully")
        return compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return final state."""
        settings = self._settings

        with trace_span("workflow.run", {"query": state.request.query[:80]}) as span:
            if _LANGGRAPH_AVAILABLE:
                result_state = self._run_langgraph(state)
            else:
                result_state = self._run_python_loop(state, settings.max_iterations)

            span["attributes"]["final_route_count"] = len(result_state.route_history)
            span["attributes"]["error_count"] = len(result_state.errors)
            span["attributes"]["has_answer"] = result_state.final_answer is not None

        return result_state

    def _run_langgraph(self, state: ResearchState) -> ResearchState:
        """Run via LangGraph StateGraph."""
        compiled = self.build()
        if compiled is None:
            return self._run_python_loop(state, self._settings.max_iterations)

        config = {"recursion_limit": self._settings.max_iterations * 3}
        state_dict = _state_to_dict(state)

        logger.info("[Workflow] Starting LangGraph execution")
        try:
            for step in compiled.stream(state_dict, config=config):
                node_name = list(step.keys())[0] if step else "?"
                logger.debug(f"[Workflow] Step: {node_name}")
            # Get final state
            final_dict = compiled.invoke(state_dict, config=config)
            return _dict_to_state(final_dict)
        except Exception as e:
            logger.error(f"LangGraph execution error: {e}; falling back to Python loop")
            return self._run_python_loop(state, self._settings.max_iterations)

    def _run_python_loop(self, state: ResearchState, max_iterations: int) -> ResearchState:
        """Pure Python agent loop — fallback when LangGraph is not available."""
        logger.info("[Workflow] Running Python agent loop (LangGraph fallback)")

        _WORKER_MAP = {
            "researcher": self._researcher,
            "analyst": self._analyst,
            "writer": self._writer,
            "critic": self._critic,
        }

        for _ in range(max_iterations * 2):  # Safety bound
            # Run supervisor to get next route
            state = self._supervisor.run(state)
            last_route = state.route_history[-1] if state.route_history else "done"

            if last_route == "done":
                logger.info(f"[Workflow] Completed after {state.iteration} iterations")
                break

            worker = _WORKER_MAP.get(last_route)
            if worker is None:
                logger.error(f"[Workflow] Unknown route: {last_route}")
                state.errors.append(f"Unknown route: {last_route}")
                break

            logger.info(f"[Workflow] → Running {last_route}")
            state = worker.run(state)

        return state
