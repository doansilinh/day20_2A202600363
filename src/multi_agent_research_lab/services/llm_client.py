"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
Supports both real OpenAI calls and an offline MockLLMClient for no-key environments.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# ---------------------------------------------------------------------------
# Knowledge base for the mock client
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, str] = {
    "graphrag": (
        "GraphRAG (Graph Retrieval-Augmented Generation) is an advanced RAG paradigm that "
        "augments language models with structured knowledge graphs. Unlike flat RAG that retrieves "
        "text chunks, GraphRAG builds a knowledge graph of entities and relationships, enabling "
        "multi-hop reasoning. Microsoft Research released a notable open-source GraphRAG framework "
        "in 2024. Key advantages: better global query answering, structured entity linking, "
        "and reduced hallucination for factual queries. Limitations: higher indexing cost, "
        "requires entity extraction pipeline (NER + relation extraction), harder to update "
        "incrementally. State-of-the-art implementations include Microsoft GraphRAG, LightRAG, "
        "and HippoRAG. Benchmarks show GraphRAG outperforms flat RAG on multi-hop QA datasets "
        "like MuSiQue and HotpotQA by 15-25% in F1 score, but at 3-5x higher latency."
    ),
    "multi_agent": (
        "Multi-agent systems (MAS) coordinate multiple specialized AI agents to solve complex tasks. "
        "Key patterns: Supervisor/Worker (one orchestrator routes tasks to specialists), "
        "Peer-to-peer (agents communicate directly), and Hierarchical (nested supervisors). "
        "Frameworks: LangGraph (graph-based stateful agents), CrewAI (role-based), "
        "AutoGen (conversational multi-agent). Core design principles: (1) Clear role boundaries, "
        "(2) Shared state for handoffs, (3) Guardrails (max_iterations, timeouts), "
        "(4) Observability via tracing. Benchmark studies show multi-agent systems outperform "
        "single agents on complex tasks by 20-40% in quality, but incur 2-4x latency overhead. "
        "Failure modes: infinite loops without proper termination conditions, "
        "context drift when handoff state is too large, and coordination overhead."
    ),
    "langgraph": (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "Built on top of LangChain, it models workflows as directed graphs (StateGraph). "
        "Key concepts: Nodes (functions/agents), Edges (transitions), Conditional edges (routing), "
        "State (shared dict passed between nodes), and Checkpointing (persistence). "
        "LangGraph supports cycles (unlike DAG-only frameworks), making it ideal for agentic "
        "loops with retry/correction steps. Version 0.2+ introduced the MessageGraph and "
        "cleaner state management with TypedDict annotations. Best practice: keep node functions "
        "pure (read state → compute → return state update), use conditional_edges for routing, "
        "and set recursion_limit to prevent infinite loops."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG) grounds LLM responses in external knowledge. "
        "Pipeline: Query → Retrieval → Context assembly → Generation. "
        "Retrieval methods: dense (vector search via FAISS, Chroma, Pinecone), "
        "sparse (BM25, TF-IDF), and hybrid (combine both with RRF). "
        "Chunking strategies: fixed-size, recursive, semantic, and document-aware. "
        "Advanced: reranking (cross-encoders), HyDE (hypothetical document embeddings), "
        "RAG-Fusion (multi-query generation). Evaluation: RAGAS (faithfulness, answer relevance, "
        "context precision/recall), TruLens, and DeepEval frameworks."
    ),
    "agent_patterns": (
        "Production agent design patterns: (1) Tool use — agents call external APIs/functions, "
        "(2) ReAct — interleave reasoning and acting, (3) Plan-and-Execute — upfront planning then execution, "
        "(4) Reflexion — self-critique and correction loops, (5) Chain-of-Thought — structured reasoning traces. "
        "Guardrails: always set max_iterations (6-10 typical), timeout_seconds (30-120), "
        "input/output validation with Pydantic, and fallback handlers. "
        "Observability: LangSmith for LangChain apps, Langfuse for provider-agnostic tracing, "
        "OpenTelemetry for enterprise integration. Cost control: use cheaper models for routing "
        "and reserve powerful models for generation tasks."
    ),
    "single_vs_multi": (
        "Single-agent vs multi-agent trade-offs: Single agents are simpler, lower latency (1 LLM call), "
        "cheaper, and easier to debug. Multi-agent systems excel at: task decomposition for complex queries, "
        "specialization (each agent optimized for its role), parallel execution (where possible), "
        "and quality improvement through critique loops. Empirical results from AgentBench and GAIA: "
        "multi-agent systems score 30% higher on complex reasoning tasks but use 3x more tokens. "
        "Recommendation: use single agents for straightforward Q&A; switch to multi-agent for "
        "research synthesis, report generation, and tasks requiring multiple perspectives."
    ),
}


def _mock_generate(system_prompt: str, user_prompt: str) -> str:
    """Generate a mock response based on system prompt role and user content."""
    sys_lower = system_prompt.lower()
    user_lower = user_prompt.lower()
    combined = sys_lower + " " + user_lower

    # Detect role from system prompt
    if "researcher" in sys_lower or "research notes" in sys_lower or "search" in sys_lower:
        return _mock_researcher_response(user_prompt, combined)
    elif "analyst" in sys_lower or "analysis" in sys_lower or "key claims" in sys_lower:
        return _mock_analyst_response(user_prompt, combined)
    elif "writer" in sys_lower or "final answer" in sys_lower or "synthesize" in sys_lower:
        return _mock_writer_response(user_prompt, combined)
    elif "critic" in sys_lower or "fact-check" in sys_lower or "validate" in sys_lower:
        return _mock_critic_response(user_prompt, combined)
    elif "supervisor" in sys_lower or "route" in sys_lower or "next agent" in sys_lower:
        return _mock_supervisor_response(user_prompt, combined)
    else:
        return _mock_general_response(user_prompt, combined)


def _find_relevant_knowledge(combined: str) -> list[str]:
    """Find relevant knowledge entries based on keyword matching."""
    relevant = []
    keywords_map: dict[str, list[str]] = {
        "graphrag": ["graphrag", "graph rag", "graph retrieval", "knowledge graph"],
        "multi_agent": ["multi-agent", "multi agent", "supervisor", "worker agent", "mas"],
        "langgraph": ["langgraph", "lang graph", "stategraph", "stateful agent"],
        "rag": ["rag", "retrieval", "vector search", "chunking", "embedding"],
        "agent_patterns": ["agent pattern", "react", "plan-and-execute", "tool use", "guardrail"],
        "single_vs_multi": ["single agent", "single-agent", "benchmark", "comparison", "trade-off"],
    }
    for key, kws in keywords_map.items():
        if any(kw in combined for kw in kws):
            relevant.append(_KNOWLEDGE_BASE[key])
    if not relevant:
        # Default: return all
        relevant = list(_KNOWLEDGE_BASE.values())[:2]
    return relevant


def _mock_researcher_response(query: str, combined: str) -> str:
    knowledge = _find_relevant_knowledge(combined)
    topics = re.findall(r"(?:about|on|for|research)\s+([A-Za-z\s\-]+?)(?:\s+and|\s+in|\.|,|$)", query, re.I)
    topic_str = topics[0].strip() if topics else query[:60]
    
    notes = f"""## Research Notes: {topic_str}

### Key Findings

"""
    for i, k in enumerate(knowledge[:3], 1):
        sentences = k.split(". ")[:4]
        notes += f"**Finding {i}:** {'. '.join(sentences)}.\n\n"

    notes += """### Source Coverage
- Academic papers and technical reports analyzed
- Industry blogs and framework documentation reviewed
- Benchmark datasets and evaluation results examined

### Gaps Identified
- Limited longitudinal studies on production deployments
- Few ablation studies comparing individual components
- Need for more diverse domain benchmarks
"""
    return notes


def _mock_analyst_response(query: str, combined: str) -> str:
    knowledge = _find_relevant_knowledge(combined)
    
    return f"""## Analytical Assessment

### Key Claims Identified

1. **Primary Claim**: Multi-agent systems provide measurable quality improvements (20-40%) over single agents on complex research tasks.
   - Evidence strength: **Strong** — supported by multiple benchmark studies (AgentBench, GAIA)
   - Caveat: Gains vary significantly by task complexity

2. **Secondary Claim**: GraphRAG outperforms flat RAG on multi-hop queries.
   - Evidence strength: **Moderate** — demonstrated on specific datasets (MuSiQue, HotpotQA)
   - Caveat: Higher indexing cost (3-5x) must be factored into deployment decisions

3. **Supporting Claim**: LangGraph's cycle support enables reliable agent loop patterns.
   - Evidence strength: **Strong** — verified by production deployments
   - Caveat: Requires careful recursion_limit configuration

### Comparative Analysis

| Dimension | Single-Agent | Multi-Agent |
|-----------|-------------|-------------|
| Latency | Low (1 LLM call) | High (3-5 calls) |
| Quality | Moderate | High |
| Cost | Low | 3-5x higher |
| Debuggability | Easy | Requires tracing |
| Scalability | Limited | High |

### Weak Evidence / Red Flags
- Claims about 100% hallucination elimination are unsupported
- Benchmark results may not transfer to domain-specific corpora
- Production latency overhead often underestimated in academic papers

### Recommended Approach
Use multi-agent architecture when query complexity justifies cost overhead. 
Implement circuit breakers and fallback to single-agent on timeout/error.
"""


def _mock_writer_response(query: str, combined: str) -> str:
    knowledge = _find_relevant_knowledge(combined)
    topics = re.findall(r"(?:about|on|for|research|write|summary)\s+([A-Za-z\s\-]+?)(?:\s+and|\s+in|\.|,|$)", query, re.I)
    topic_str = topics[0].strip() if topics else "the requested topic"
    
    return f"""# Research Summary: {topic_str.title()}

## Executive Summary

This report synthesizes current state-of-the-art research on {topic_str}, drawing from academic literature, 
technical documentation, and empirical benchmarks. The analysis reveals both significant advances 
and important limitations that practitioners should consider.

## Core Findings

### 1. Architecture & Design Principles

Modern AI research systems increasingly adopt **multi-agent architectures** that decompose complex 
tasks into specialized roles. The Supervisor-Researcher-Analyst-Writer pattern has emerged as a 
production-proven framework, with clear separation of concerns enabling:
- Independent optimization of each agent's prompts and models
- Better traceability and debugging through explicit handoffs
- Modular replacement of individual components

### 2. Retrieval-Augmented Generation

**GraphRAG** represents a significant evolution over flat RAG, particularly for queries requiring 
multi-hop reasoning across entity relationships. Key metrics from benchmark evaluations:
- 15-25% improvement in F1 score on multi-hop QA (MuSiQue, HotpotQA)
- 3-5x higher indexing cost compared to flat RAG
- Recommended for knowledge-intensive, structured domains

### 3. Orchestration with LangGraph

LangGraph's stateful graph model enables robust agentic workflows with:
- Cycle support for iterative refinement loops
- Built-in checkpointing for fault tolerance
- Conditional routing for dynamic task allocation

### 4. Performance Benchmarks

| System | Quality Score | Latency | Token Cost |
|--------|--------------|---------|------------|
| Single-Agent | 6.2/10 | 2.1s | 1x |
| Multi-Agent | 8.7/10 | 8.4s | 3.8x |
| GraphRAG + Multi-Agent | 9.1/10 | 12.3s | 5.2x |

## Limitations & Failure Modes

1. **Context drift**: Long handoff chains can dilute the original query intent
2. **Coordination overhead**: Each agent invocation adds latency (typically 1-3s per call)
3. **Cost explosion**: Poorly bounded iterations can multiply token costs 10x+
4. **Debugging complexity**: Multi-agent traces require dedicated observability tooling

## Recommendations

1. Start with single-agent baseline; upgrade to multi-agent only when quality targets aren't met
2. Always set `max_iterations` (6-10) and `timeout_seconds` (60-120)
3. Invest in tracing infrastructure early (LangSmith, Langfuse, or custom JSON traces)
4. Use cheaper models (GPT-4o-mini, Gemini Flash) for routing and summarization tasks

## References

1. Microsoft GraphRAG (2024) — https://github.com/microsoft/graphrag
2. LangGraph Documentation — https://langchain-ai.github.io/langgraph/
3. AgentBench Benchmark (2023) — https://arxiv.org/abs/2308.03688
4. Anthropic: Building Effective Agents — https://www.anthropic.com/engineering/building-effective-agents
5. RAGAS Evaluation Framework — https://docs.ragas.io/
"""


def _mock_critic_response(query: str, combined: str) -> str:
    return """## Critic Review

### Citation Coverage Analysis
- Citations present: ✅ 5 references provided
- Citation quality: ✅ Mix of academic papers and technical documentation
- Broken links: ⚠️ URLs not verified (offline mode)

### Factual Validation
- Benchmark numbers (15-25% F1 improvement for GraphRAG): ✅ Consistent with published research
- Latency estimates (1-3s per agent call): ✅ Plausible for GPT-4o-mini API
- Cost multiplier (3-5x for multi-agent): ✅ Conservative, realistic estimate

### Potential Hallucinations / Weak Claims
- "100% reliability" — NOT claimed in output ✅
- Specific model benchmark scores are approximate ⚠️ (within reasonable range)

### Safety & Quality Score
- Factual accuracy: 8.5/10
- Citation coverage: 9.0/10
- Clarity: 9.0/10
- Actionability: 8.5/10
- **Overall: 8.75/10** ✅ Passes quality threshold
"""


def _mock_supervisor_response(query: str, combined: str) -> str:
    return "researcher"


def _mock_general_response(query: str, combined: str) -> str:
    knowledge = _find_relevant_knowledge(combined)
    response = f"Based on available knowledge about your query: '{query[:100]}'\n\n"
    for k in knowledge[:2]:
        response += k[:300] + "...\n\n"
    return response


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Offline LLM client using template-based responses. No API key required."""

    def __init__(self) -> None:
        self.model = "mock-llm-v1"
        logger.info("Using MockLLMClient (offline mode — no API key required)")

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        logger.debug(f"MockLLMClient.complete called | system=[{system_prompt[:60]}...]")
        content = _mock_generate(system_prompt, user_prompt)
        # Estimate token counts
        input_tokens = (len(system_prompt) + len(user_prompt)) // 4
        output_tokens = len(content) // 4
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # Free (mock)
        )


class LLMClient:
    """Real OpenAI LLM client with retry logic."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            raise AgentExecutionError("openai package not installed. Run: pip install openai") from e

        settings = get_settings()
        if not settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY not set. Use MockLLMClient or set the env var."
            )
        self.client = OpenAI(api_key=settings.openai_api_key)  # type: ignore[no-untyped-call]
        self.model = settings.openai_model
        logger.info(f"Using real OpenAI LLMClient with model={self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry logic."""
        logger.debug(f"Calling OpenAI LLM with model {self.model}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
        cost_usd = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


def get_llm_client() -> Any:
    """Factory: returns real LLMClient if API key available, else MockLLMClient."""
    settings = get_settings()
    if settings.openai_api_key:
        try:
            return LLMClient()
        except Exception:
            logger.warning("Failed to initialize real LLMClient; falling back to MockLLMClient")
            return MockLLMClient()
    return MockLLMClient()
