"""Search client abstraction for ResearcherAgent.

Provides both a real TavilySearchClient (requires API key) and a MockSearchClient
with a curated knowledge base for offline / no-key operation.
"""

import logging
from dataclasses import dataclass

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated offline knowledge base (20+ documents)
# ---------------------------------------------------------------------------

@dataclass
class _KBDoc:
    title: str
    url: str
    snippet: str
    tags: list[str]


_KNOWLEDGE_BASE: list[_KBDoc] = [
    _KBDoc(
        title="Microsoft GraphRAG: From Local to Global RAG",
        url="https://arxiv.org/abs/2404.16130",
        snippet=(
            "GraphRAG extends RAG by building a knowledge graph of entities and relationships "
            "from source documents. It enables global query answering by traversing the graph "
            "across communities of related entities. Shows 15-25% improvement in F1 score on "
            "multi-hop QA datasets (MuSiQue, HotpotQA) vs flat RAG."
        ),
        tags=["graphrag", "rag", "knowledge graph", "multi-hop", "retrieval"],
    ),
    _KBDoc(
        title="LangGraph: Building Stateful Multi-Actor Applications",
        url="https://langchain-ai.github.io/langgraph/concepts/",
        snippet=(
            "LangGraph models agentic workflows as directed graphs (StateGraph). Key primitives: "
            "Nodes (agent functions), Edges (transitions), Conditional edges (routing logic), "
            "and shared State (TypedDict). Unlike DAG frameworks, LangGraph supports cycles "
            "enabling iterative agentic loops. Set recursion_limit to prevent infinite loops."
        ),
        tags=["langgraph", "multi-agent", "stategraph", "orchestration", "workflow"],
    ),
    _KBDoc(
        title="Anthropic: Building Effective Agents",
        url="https://www.anthropic.com/engineering/building-effective-agents",
        snippet=(
            "Effective agents require: (1) Clear tool definitions with precise schemas, "
            "(2) Explicit stopping conditions to avoid infinite loops, (3) Human-in-the-loop "
            "for high-stakes decisions, (4) Comprehensive logging and tracing, "
            "(5) Graceful error handling with fallbacks. Prefer simple solutions over complex "
            "multi-agent systems unless task genuinely requires decomposition."
        ),
        tags=["agent patterns", "best practices", "anthropic", "tool use", "guardrail"],
    ),
    _KBDoc(
        title="AgentBench: Evaluating LLMs as Agents",
        url="https://arxiv.org/abs/2308.03688",
        snippet=(
            "AgentBench is a comprehensive benchmark for evaluating LLMs in agent settings. "
            "Tests operating systems, databases, knowledge graphs, digital games, and web browsing. "
            "Results: GPT-4 significantly outperforms open-source models; multi-step reasoning "
            "remains challenging. Provides standardized evaluation across 8 distinct environments."
        ),
        tags=["benchmark", "evaluation", "agent", "llm", "multi-agent"],
    ),
    _KBDoc(
        title="LightRAG: Simple and Fast Retrieval-Augmented Generation",
        url="https://arxiv.org/abs/2410.05779",
        snippet=(
            "LightRAG introduces dual-level retrieval (local entity-level + global community-level) "
            "for comprehensive knowledge graph-based RAG. Uses graph-based text indexing with "
            "incremental update support. Outperforms naive RAG and GraphRAG on diversity and "
            "comprehensiveness metrics. Open-source with Hugging Face model support."
        ),
        tags=["lightrag", "graphrag", "rag", "knowledge graph", "retrieval"],
    ),
    _KBDoc(
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        url="https://arxiv.org/abs/2210.03629",
        snippet=(
            "ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning with action "
            "generation in LLMs. Enables agents to plan, execute tools, observe results, and "
            "refine reasoning iteratively. Outperforms pure reasoning (CoT) and pure acting "
            "baselines on HotpotQA, Fever, and ALFWorld benchmarks."
        ),
        tags=["react", "agent patterns", "reasoning", "tool use", "chain-of-thought"],
    ),
    _KBDoc(
        title="Multi-Agent Systems: A Survey of Architectures and Patterns",
        url="https://arxiv.org/abs/2402.01680",
        snippet=(
            "Comprehensive survey of multi-agent system (MAS) architectures: hierarchical, "
            "peer-to-peer, blackboard, and market-based. Key findings: supervisor-worker pattern "
            "most effective for task decomposition; shared state crucial for coherent handoffs; "
            "communication overhead grows quadratically with agent count."
        ),
        tags=["multi-agent", "architecture", "survey", "supervisor", "worker"],
    ),
    _KBDoc(
        title="RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        url="https://arxiv.org/abs/2309.15217",
        snippet=(
            "RAGAS provides reference-free evaluation for RAG systems using LLMs as judges. "
            "Metrics: Faithfulness (factual consistency with context), Answer Relevance (query "
            "alignment), Context Precision, Context Recall, and Answer Correctness. "
            "Enables automated quality assessment without human annotation for every test case."
        ),
        tags=["ragas", "evaluation", "rag", "benchmark", "quality"],
    ),
    _KBDoc(
        title="OpenAI Agents SDK: Orchestration and Handoffs",
        url="https://developers.openai.com/api/docs/guides/agents/orchestration",
        snippet=(
            "OpenAI Agents SDK provides primitives for agent orchestration: Agent class with "
            "instructions and tools, handoff() for transferring control between agents, "
            "and Runner for executing agent workflows. Supports streaming and async execution. "
            "Built-in guardrails via input/output validation hooks."
        ),
        tags=["openai", "agents sdk", "handoff", "orchestration", "multi-agent"],
    ),
    _KBDoc(
        title="LangSmith: Observability for LLM Applications",
        url="https://docs.smith.langchain.com/",
        snippet=(
            "LangSmith provides tracing, monitoring, and evaluation for LangChain applications. "
            "Features: run tree visualization, token usage tracking, latency profiling, "
            "prompt iteration, and dataset management for evaluation. Integrates seamlessly "
            "with LangGraph via LANGCHAIN_TRACING_V2 environment variable."
        ),
        tags=["langsmith", "tracing", "observability", "monitoring", "langchain"],
    ),
    _KBDoc(
        title="Langfuse: Open-Source LLM Observability",
        url="https://langfuse.com/docs",
        snippet=(
            "Langfuse is an open-source observability platform for LLM applications. "
            "Supports: traces (sessions with spans), scores (quality ratings), datasets, "
            "and prompt management. Provider-agnostic — works with any LLM. "
            "Self-hostable or cloud. Python/JS SDK with automatic context propagation."
        ),
        tags=["langfuse", "tracing", "observability", "open-source", "monitoring"],
    ),
    _KBDoc(
        title="HippoRAG: Neurologically Inspired Long-Term Memory for RAG",
        url="https://arxiv.org/abs/2405.14831",
        snippet=(
            "HippoRAG draws inspiration from human hippocampal indexing theory to build "
            "a knowledge graph for RAG. Uses Personalized PageRank for multi-hop retrieval. "
            "Outperforms GraphRAG on entity-centric queries; more memory-efficient representation. "
            "Suitable for large-scale document collections requiring frequent updates."
        ),
        tags=["hipporag", "graphrag", "knowledge graph", "multi-hop", "memory"],
    ),
    _KBDoc(
        title="CrewAI: Role-Playing Multi-Agent Framework",
        url="https://docs.crewai.com/",
        snippet=(
            "CrewAI orchestrates AI agents with role definitions, backstory, goals, and tools. "
            "Each agent has a clear persona enabling specialization. Supports sequential and "
            "hierarchical task execution. Used by Fortune 500 companies for automated research, "
            "content generation, and complex analysis workflows."
        ),
        tags=["crewai", "multi-agent", "role-based", "framework", "orchestration"],
    ),
    _KBDoc(
        title="AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        url="https://arxiv.org/abs/2308.08155",
        snippet=(
            "AutoGen enables building LLM applications with multi-agent conversation patterns. "
            "Agents communicate via natural language messages. Supports human proxy agents, "
            "code execution agents, and conversational agents. Achieves 91% on HumanEval with "
            "multi-agent code generation + testing loop vs 80% single-agent baseline."
        ),
        tags=["autogen", "multi-agent", "code generation", "conversation", "microsoft"],
    ),
    _KBDoc(
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        url="https://arxiv.org/abs/2303.11366",
        snippet=(
            "Reflexion allows agents to improve through self-critique and reflection without "
            "gradient updates. Agent produces action, observes outcome, reflects verbally on "
            "failure, and retries. Achieves 91% pass@1 on HumanEval, 67% on ALFWorld (vs "
            "baseline 45%). Most effective for tasks with clear success/failure signals."
        ),
        tags=["reflexion", "agent patterns", "self-critique", "reinforcement", "reasoning"],
    ),
    _KBDoc(
        title="Vector Databases for Semantic Search: A Practical Guide",
        url="https://www.pinecone.io/learn/vector-database/",
        snippet=(
            "Vector databases store high-dimensional embeddings for semantic similarity search. "
            "Options: Pinecone (managed), Weaviate (open-source), Chroma (local-first), "
            "FAISS (library), Qdrant (Rust-based). Key metrics: queries per second (QPS), "
            "recall@k, and memory footprint. HNSW index most common for production deployments."
        ),
        tags=["vector database", "semantic search", "embedding", "rag", "faiss"],
    ),
    _KBDoc(
        title="Pydantic v2: Data Validation for Python Applications",
        url="https://docs.pydantic.dev/latest/",
        snippet=(
            "Pydantic v2 provides fast, Rust-backed data validation using Python type annotations. "
            "Key features: BaseModel for schema definition, Field() for constraints, "
            "model_validator for cross-field validation, and JSON serialization. "
            "Essential for agent input/output validation in production LLM applications."
        ),
        tags=["pydantic", "validation", "python", "schema", "type safety"],
    ),
    _KBDoc(
        title="Cost Optimization for LLM Applications",
        url="https://www.anthropic.com/engineering/claude-cost-optimization",
        snippet=(
            "LLM cost optimization strategies: (1) Use smaller models for routing (GPT-4o-mini, "
            "Gemini Flash), (2) Implement prompt caching for repeated context, (3) Batch requests "
            "where possible, (4) Set strict output length limits, (5) Use structured output "
            "to reduce regeneration. Multi-agent systems can reduce cost via task decomposition "
            "allowing cheaper models for sub-tasks."
        ),
        tags=["cost", "optimization", "llm", "efficiency", "budget"],
    ),
    _KBDoc(
        title="Production MLOps: From Research to Deployed AI Systems",
        url="https://ml-ops.org/",
        snippet=(
            "MLOps for LLM-powered agents: model versioning, A/B testing, canary deployments, "
            "and rollback strategies. Monitoring: latency percentiles (p50/p95/p99), error rates, "
            "token usage, and quality drift detection. Infrastructure: containerization (Docker), "
            "orchestration (Kubernetes), and CI/CD pipelines for prompt changes."
        ),
        tags=["mlops", "production", "deployment", "monitoring", "infrastructure"],
    ),
    _KBDoc(
        title="Prompt Engineering: Advanced Techniques for LLM Control",
        url="https://www.promptingguide.ai/",
        snippet=(
            "Advanced prompt engineering: Chain-of-Thought (CoT) for reasoning, "
            "few-shot examples for format consistency, system prompt personas for role clarity, "
            "XML/JSON structured output for reliable parsing, and temperature control "
            "(0 for deterministic, 0.7+ for creative). Role-specific prompts crucial for "
            "multi-agent systems to prevent role confusion."
        ),
        tags=["prompt engineering", "chain-of-thought", "few-shot", "llm", "techniques"],
    ),
]


def _score_doc(doc: _KBDoc, query: str) -> int:
    """Score document relevance to query using keyword matching."""
    query_lower = query.lower()
    score = 0
    # Tag matches (high weight)
    for tag in doc.tags:
        if tag in query_lower:
            score += 3
    # Title keyword matches
    for word in doc.title.lower().split():
        if len(word) > 4 and word in query_lower:
            score += 2
    # Snippet keyword matches
    query_words = [w for w in query_lower.split() if len(w) > 4]
    for word in query_words:
        if word in doc.snippet.lower():
            score += 1
    return score


class MockSearchClient:
    """Offline search client using a curated knowledge base. No API key required."""

    def __init__(self) -> None:
        logger.info("Using MockSearchClient (offline mode — curated knowledge base)")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return top-scored documents from the knowledge base."""
        scored = [(doc, _score_doc(doc, query)) for doc in _KNOWLEDGE_BASE]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:max_results]

        results = []
        for doc, score in top:
            results.append(
                SourceDocument(
                    title=doc.title,
                    url=doc.url,
                    snippet=doc.snippet,
                    metadata={"relevance_score": score, "tags": doc.tags},
                )
            )
        logger.debug(f"MockSearchClient returned {len(results)} results for query='{query[:50]}'")
        return results


class TavilySearchClient:
    """Real Tavily web search client (requires TAVILY_API_KEY)."""

    def __init__(self) -> None:
        try:
            from tavily import TavilyClient  # type: ignore[import]
        except ImportError as e:
            raise ImportError("tavily-python not installed. Run: pip install tavily-python") from e

        settings = get_settings()
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not set in environment")
        self.client = TavilyClient(api_key=settings.tavily_api_key)  # type: ignore[no-untyped-call]
        logger.info("Using TavilySearchClient (live web search)")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search Tavily for real-time web results."""
        response = self.client.search(query=query, max_results=max_results)  # type: ignore[no-untyped-call]
        results = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", ""),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score", 0)},
                )
            )
        return results


class SearchClient:
    """Factory search client: uses Tavily if key available, else MockSearchClient."""

    def __new__(cls) -> "MockSearchClient | TavilySearchClient":  # type: ignore[misc]
        settings = get_settings()
        if settings.tavily_api_key:
            try:
                return TavilySearchClient()
            except Exception:
                logger.warning("TavilySearchClient failed; falling back to MockSearchClient")
        return MockSearchClient()
