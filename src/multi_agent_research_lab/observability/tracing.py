"""Tracing hooks.

Provides:
1. trace_span — context manager for timing and attribute collection
2. JsonFileTracer — writes traces to JSON files for offline analysis
3. setup_tracing — initialize tracing provider based on config
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Timing + attribute context manager.

    Records duration_seconds and emits a DEBUG log on exit.
    The caller may mutate the returned span dict inside the with-block to add
    attributes that are only known after the work completes.
    """
    started = perf_counter()
    now_utc = datetime.now(timezone.utc).isoformat()
    span: dict[str, Any] = {
        "name": name,
        "start_time": now_utc,
        "attributes": dict(attributes or {}),
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug(
            f"[trace] {name} | {span['duration_seconds']:.3f}s | "
            f"status={span['status']} | attrs={span['attributes']}"
        )


class JsonFileTracer:
    """Writes spans and trace events to a local JSON file."""

    def __init__(self, output_path: Path | None = None, run_name: str = "trace") -> None:
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            output_path = Path("reports") / "traces" / f"{ts}_{run_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self._spans: list[dict[str, Any]] = []
        logger.info(f"JsonFileTracer initialized — output: {output_path}")

    def record_span(self, span: dict[str, Any]) -> None:
        self._spans.append(span)

    def record_trace_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self._spans.append({"type": "trace_event", **event})

    def flush(self, state_summary: dict[str, Any] | None = None) -> Path:
        """Write all spans to JSON file and return path."""
        output = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "span_count": len(self._spans),
            },
            "state_summary": state_summary or {},
            "spans": self._spans,
        }
        self.output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Trace written to {self.output_path} ({len(self._spans)} spans)")
        return self.output_path


def setup_tracing(run_name: str = "run") -> JsonFileTracer:
    """Initialize tracing. Returns a JsonFileTracer always.

    TODO: Replace/augment with LangSmith or Langfuse provider if keys are available.
    """
    settings = get_settings()

    # Attempt LangSmith setup if key available
    if settings.langsmith_api_key:
        try:
            import os

            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            logger.info(f"LangSmith tracing enabled for project: {settings.langsmith_project}")
        except Exception as e:
            logger.warning(f"LangSmith setup failed: {e}")

    # Always create a local JSON tracer as fallback
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # trim to milliseconds
    trace_path = Path("reports") / "traces" / f"{ts}_{run_name}.json"
    return JsonFileTracer(output_path=trace_path)
