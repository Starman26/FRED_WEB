# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SENTINEL is a multi-agent orchestration system for laboratory automation, technical documentation research, and robot control. Built with LangGraph, it coordinates specialized worker agents to handle complex industrial automation tasks.

## Development Commands

### Running the Application

**Streamlit UI (Development)**
```bash
streamlit run app/app_streamlit.py
```

**FastAPI Server (Production)**
```bash
# Local development
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# Production (Cloud Run)
docker build -t fredie-agent .
docker run -p 8080:8080 --env-file .env fredie-agent
```

### Testing
```bash
# Run all tests
pytest

# Run with async support
pytest-asyncio
```

### Code Quality
```bash
# Format code
black src/ app/

# Type checking (if configured)
# mypy src/
```

### Environment Setup

**Required environment variables** (see `env.yaml`):
- `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `TAVILY_API_KEY` (for web search)
- `DEFAULT_MODEL` (e.g., `gpt-4o-mini`)

## Architecture

### Multi-Agent Orchestration Flow

The system uses a **ReAct planner + adaptive router** pattern with LangGraph:

```
START → bootstrap → planner → [worker₁ → adaptive_router → worker₂ → ...] → synthesize → END
                                                ↓
                                          human_input (HITL)
```

**Key Nodes:**
- `bootstrap`: Initializes state, validates services (Supabase, embeddings)
- `planner`: ReAct planner that fuses intent analysis + plan generation in one node. Fast-path (regex, ~70% of queries, 0 LLM calls) and smart-path (1 LLM call with chain-of-thought reasoning for ambiguous queries)
- `adaptive_router`: Evaluates worker output quality and adapts the plan dynamically (e.g., skips tutor if research found no evidence, adds tutor if troubleshooter recommends it)
- `synthesize`: Combines worker outputs into coherent response. Bypasses LLM for lightweight single-worker outputs
- `human_input`: Handles clarification questions (Human-in-the-Loop)

### Worker Agents

Located in [src/agent/workers/](src/agent/workers/), each worker is a specialized agent:

| Worker | Purpose | Key Tools |
|--------|---------|-----------|
| `chat` | Casual conversation, greetings | None (direct LLM) |
| `research` | RAG search in technical documents | `rag_tools.py` |
| `tutor` | Educational explanations | Synthesis of research results |
| `troubleshooting` | Lab diagnostics, equipment status | `lab_tools/*` |
| `robot_operator` | xArm robot control | `robot_tools/xarm_tools.py` |
| `analysis` | SQL data analysis on Supabase | `analyst_tools.py` |
| `summarizer` | Memory compression | N/A |

**Adding a New Worker:**
1. Create `src/agent/workers/your_worker_node.py` following the pattern in existing workers
2. Add worker name to `VALID_WORKERS` in [src/agent/graph.py](src/agent/graph.py) and [src/agent/orchestrator.py](src/agent/orchestrator.py)
3. Register node in `create_graph()` in [src/agent/graph.py](src/agent/graph.py)
4. Add routing logic in [src/agent/nodes/planner.py](src/agent/nodes/planner.py) (plan mapping table + REACT_PLANNER_PROMPT workers list)
5. Update Streamlit and API server graph construction if they build their own graph

### State Management

State is defined in [src/agent/state.py](src/agent/state.py) using `TypedDict` with custom reducers:

**Critical Fields:**
- `messages`: Conversation history (accumulated with `operator.add`)
- `events`: Debug/UI logs (accumulated)
- `orchestration_plan`: List of workers to execute (e.g., `["research", "tutor"]`)
- `current_step`: Index in plan
- `worker_outputs`: Accumulated results from workers (uses `merge_worker_outputs` reducer)
- `pending_context`: Context passed between workers (uses `merge_dicts` reducer)
- `needs_human_input`: Triggers HITL flow
- `clarification_questions`: Questions for user

**State Lifecycle:**
1. Bootstrap initializes with `STATE_DEFAULTS`
2. Each worker updates state via return dict
3. LangGraph merges updates using annotated reducers
4. State persists via `MemorySaver` checkpointer

### Tools Architecture

Tools are organized by domain in [src/agent/tools/](src/agent/tools/):

**Lab Tools** ([src/agent/tools/lab_tools/](src/agent/tools/lab_tools/)):
- `station_tools.py`: Station status queries
- `equipment_tools.py`: Equipment diagnostics (PLC, cobot, sensors)
- `error_tools.py`: Error log analysis
- `repair_tools.py`: Repair history
- `formatters.py`: Markdown output formatting

**Robot Tools** ([src/agent/tools/robot_tools/](src/agent/tools/robot_tools/)):
- `xarm_tools.py`: xArm control (move, home, gripper, emergency stop)

**RAG Tools** ([src/agent/tools/rag_tools.py](src/agent/tools/rag_tools.py)):
- Semantic search in Supabase vector store
- PDF document ingestion

### Services Layer

**DO NOT** store Supabase/embeddings clients in state (not serializable). Use:
```python
from src.agent.services import get_supabase, get_embeddings
```

Services are initialized once in [src/agent/bootstrap.py](src/agent/bootstrap.py).

## Key Patterns

### Worker Contract

All workers follow `WorkerOutput` contract from [src/agent/contracts/worker_contract.py](src/agent/contracts/worker_contract.py):

```python
{
    "worker": "worker_name",
    "task_id": "unique_id",
    "status": "success" | "needs_context" | "error",
    "content": "markdown_response",
    "evidence": [{"title": "doc", "page": 1, "snippet": "..."}],
    "clarification_questions": [...],  # if needs_context
    "follow_up_suggestions": ["suggestion1", ...],
}
```

### Human-in-the-Loop (HITL)

Workers can request clarification by returning `status: "needs_context"`:
```python
return {
    "worker": "troubleshooting",
    "status": "needs_context",
    "clarification_questions": [
        {
            "id": "q1",
            "question": "Which station has the error?",
            "options": [
                {"label": "Station 1", "value": "1"},
                {"label": "Station 2", "value": "2"}
            ]
        }
    ]
}
```

Orchestrator routes to `human_input` node, which triggers Streamlit wizard UI.

### LLM Model Selection

Use `llm_factory.py` for dynamic model selection:
```python
from src.agent.utils.llm_factory import get_llm

llm = get_llm(state, temperature=0.7, max_tokens=1000)
```

Supports: Claude (Anthropic), GPT (OpenAI), Gemini (Google). Model is specified in `state["llm_model"]`.

### Token Management

User token tracking via [src/agent/utils/token_manager.py](src/agent/utils/token_manager.py):
```python
from src.agent.utils.token_manager import check_balance, deduct_tokens

balance = check_balance(user_id)
if balance["has_credits"]:
    # ... execute task
    deduct_tokens(user_id, tokens_used, description="Research query")
```

### Learning Profiles

User preferences loaded from Supabase via [src/agent/utils/learning_profile.py](src/agent/utils/learning_profile.py):
```python
from src.agent.utils.learning_profile import get_learning_profile

profile = get_learning_profile(user_id, state)  # Cached per session
```

## Important Implementation Notes

### Worker Execution

- Workers are called sequentially based on `orchestration_plan`
- Each worker can add to `pending_context` to pass data forward
- Workers emit events via `event_*` helpers in [src/agent/utils/run_events.py](src/agent/utils/run_events.py)
- Events are displayed in Streamlit as timeline UI

### Synthesize Node

[src/agent/orchestrator.py:335](src/agent/orchestrator.py#L335) combines worker outputs:
- Uses LLM to align response with user's original question
- Supports mode-specific formatting (Agent/Voice/Chat)
- Strips emojis (UI adds them separately)
- Includes sources from evidence

### Planner (ReAct)

[src/agent/nodes/planner.py](src/agent/nodes/planner.py) fuses intent analysis + plan generation:
- **Fast-path** (~70%): regex-based classification, 0 LLM calls, <1ms
- **Smart-path** (~30%): 1 LLM call with chain-of-thought reasoning
- Output: `intent_analysis` (backward-compat), `orchestration_plan`, `plan_reasoning`, `planner_method`
- **Analysis mode**: `interaction_mode == 'analysis'` plans normally (fast/smart path) then appends `analysis` as the final step. This makes analysis a hybrid enrichment worker — it receives previous worker outputs as context and adds SQL queries, charts, and statistics on top
- `intent_analyzer.py` is deprecated; it re-exports from planner.py

### Adaptive Router

[src/agent/orchestrator.py](src/agent/orchestrator.py) `adaptive_router_node` evaluates worker outputs:
- Rule 1: Worker error → stop early, synthesize what's available
- Rule 2: Research found no evidence → remove tutor from remaining plan
- Rule 3: High-confidence single worker → stop early
- Rule 4: Troubleshooter recommends tutor → add to plan dynamically
- Anti-loop: forces `synthesize` after 3+ routing cycles

## File Organization

```
src/agent/
├── graph.py              # LangGraph workflow definition
├── state.py              # Shared state schema + reducers
├── bootstrap.py          # State initialization
├── orchestrator.py       # Plan/route/synthesize nodes
├── services.py           # Supabase, embeddings clients
├── nodes/
│   ├── planner.py            # ReAct planner (intent + plan in one node)
│   ├── intent_analyzer.py    # DEPRECATED - re-exports from planner.py
│   ├── human_input.py
│   └── verify_info.py
├── workers/              # Specialized agents
│   ├── chat_node.py
│   ├── research_node.py
│   ├── tutor_node.py
│   ├── troubleshooter_node.py
│   ├── robot_operator_node.py
│   ├── analysis_node.py
│   └── summarizer_node.py
├── tools/                # Domain-specific tools
│   ├── lab_tools/
│   ├── robot_tools/
│   ├── analyst_tools.py
│   └── rag_tools.py
├── contracts/            # Type definitions
│   └── worker_contract.py
└── utils/                # Helpers
    ├── llm_factory.py
    ├── token_manager.py
    ├── learning_profile.py
    ├── run_events.py
    └── logger.py

app/
└── app_streamlit.py      # Streamlit UI with HITL wizard

api_server.py             # FastAPI SSE endpoint
```

## Common Tasks

### Add a New Lab Equipment Type

1. Add tool functions in `src/agent/tools/lab_tools/equipment_tools.py`
2. Register tools in `src/agent/tools/lab_tools/__init__.py`
3. Update troubleshooter worker to use new tools
4. Add database schema if needed (Supabase migrations)

### Add Support for New Robot

1. Create `src/agent/tools/robot_tools/your_robot_tools.py`
2. Create or modify worker in `src/agent/workers/`
3. Update intent analyzer to recognize robot keywords
4. Add to `VALID_WORKERS` and routing logic

### Modify Response Formatting

Edit `synthesize_node` in [src/agent/orchestrator.py:335](src/agent/orchestrator.py#L335):
- `_synthesize_with_llm`: LLM-based synthesis
- `_condense_for_agent_mode`: Ultra-concise first-person mode
- `_condense_for_voice_mode`: LLM-based voice condensation (1-2 sentences, first person, radio style)
- `_format_as_markdown`: Structured markdown output

### Debug Worker Execution

Enable detailed logging:
```python
from src.agent.utils.logger import logger
logger.set_level(logging.DEBUG)
```

Check events in Streamlit sidebar "System Logs" or via `state["events"]`.

## Database Schema (Supabase)

Key tables:
- `profiles`: User accounts with learning preferences, token balance
- `teams`: Team membership for multi-user labs
- `documents`: RAG document metadata
- `document_chunks`: Vector embeddings for semantic search
- `equipment_status`: Real-time lab equipment state
- `error_logs`: Equipment error history

## Testing Strategy

- Unit tests for individual tools/workers
- Integration tests for full graph execution
- Use `MemorySaver` checkpointer for test isolation
- Mock Supabase client in tests to avoid DB dependencies
