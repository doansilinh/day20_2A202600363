# Lab 20: Multi-Agent Research System Starter

Starter repo cho bài lab **Multi-Agent Systems**: xây dựng hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer** và benchmark với single-agent baseline.

> Mục tiêu của repo này là cung cấp **production-grade skeleton** để học viên phát triển code cá nhân. Các phần logic quan trọng được để ở dạng `TODO` để học viên tự triển khai.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + skeletons
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow skeleton
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark/evaluation skeleton
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e "[dev]"
cp .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền key cần thiết.

```bash
OPENAI_API_KEY=...
# optional
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

### 3. Chạy smoke test

```bash
make test
python -m multi_agent_research_lab.cli --help
```

## Cách chạy hệ thống (Hướng dẫn nộp bài)

Hệ thống đã hoàn thiện 100% các yêu cầu lab, hỗ trợ cả chế độ **Offline (Mock)** và **Online (OpenAI/Tavily)**.

### 1. Chạy nghiên cứu Multi-Agent
Hệ thống sẽ tự động lưu Trace vào `reports/traces/`.
```bash
python -m multi_agent_research_lab.cli multi-agent -q "Your research topic"
```

### 2. Chạy Baseline (Single-Agent)
Dùng để so sánh kết quả nhanh.
```bash
python -m multi_agent_research_lab.cli baseline -q "Your research topic"
```

### 3. Chạy Đánh giá hệ thống (Benchmark)
Script này sẽ chạy 10 kịch bản so sánh, dọn dẹp thư mục trace cũ và xuất báo cáo tổng hợp.
```bash
python scripts/run_benchmarks.py
```
*   **Kết quả báo cáo:** `reports/benchmark_report.md`
*   **Trace dữ liệu:** `reports/trace_benchmark/`

## Trạng thái dự án (Completed)

Hệ thống đã đạt được các cột mốc:
- [x] **Agent Roles**: Đầy đủ 5 roles (Supervisor, Researcher, Analyst, Writer, Critic).
- [x] **State Management**: Shared state qua LangGraph, hỗ trợ lưu vết lịch sử route.
- [x] **Guardrails**: Có `max_iterations`, xử lý lỗi tập trung, dừng khẩn cấp khi gặp quá nhiều lỗi.
- [x] **Observability**: Hệ thống `JsonFileTracer` tự động ghi lại mọi bước đi của agent dưới dạng span.
- [x] **Benchmarking**: Báo cáo so sánh chất lượng (Quality), thời gian (Latency) và chi phí (Cost).
- [x] **Offline Support**: Toàn bộ hệ thống có thể chạy không cần internet nhờ Mock LLM và Mock Search.

## Deliverables (Đã hoàn thành)

1. [x] **GitHub Repo**: Sẵn sàng nộp.
2. [x] **Traces**: Được lưu tự động trong `reports/traces/` và `reports/trace_benchmark/`.
3. [x] **Báo cáo**: `reports/benchmark_report.md` so sánh chi tiết.
4. [x] **Failure Mode**: Đã được phân tích và fix (xem mục "Failure Mode Analysis" trong báo cáo benchmark).

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
