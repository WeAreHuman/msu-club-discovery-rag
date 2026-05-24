# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG (Retrieval-Augmented Generation) chatbot for discovering MSU student clubs. Students have multi-turn conversations; the system retrieves relevant club data from Pinecone and generates answers via Groq/Llama. The architecture is **headless**: a FastAPI backend exposes the RAG engine as a REST API, and Streamlit is a thin frontend client.

## Commands

```bash
# Start the FastAPI backend (required before running the UI)
uvicorn api.main:app --reload --port 8000

# Run the Streamlit UI (in a separate terminal)
streamlit run app.py

# Ingest club data into Pinecone (resumes from checkpoint automatically)
python ingest_data.py

# Ingest a single club by slug (for testing)
python ingest_data.py --club accessibility-club

# Wipe Pinecone namespace + checkpoint and restart ingestion from scratch
python ingest_data.py --clear

# Run system tests
python test_system.py

# Run RAGAS evaluation (requires eval deps; uses GROQ_API_KEY — no OpenAI needed)
pip install -r eval/requirements-eval.txt
python eval/run_eval.py --label baseline

# Compare two eval runs
python eval/compare.py eval/results/<file_a>.json eval/results/<file_b>.json
```

## Environment Setup

Copy `.env.example` to `.env` and fill in:
- `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` — required
- `LLM_PROVIDER=groq` and `GROQ_API_KEY` — recommended (free tier)
- `SCRAPER_ORGS_DIR` — path to the `msu_scraper/data/orgs/` directory
- `API_BASE_URL` — URL of the FastAPI backend (default: `http://localhost:8000`)
- `OPENAI_API_KEY` — NOT required; eval uses Groq + local HuggingFace embeddings (no OpenAI needed)

## Architecture

### Headless Design

The system is split into two independently runnable layers:

```
┌─────────────────────────────────────┐
│   Streamlit UI  (app.py)            │
│   — thin HTTP client                │
│   — no RAG logic, no Pinecone       │
└──────────────┬──────────────────────┘
               │ HTTP POST /chat
               │ HTTP POST /query
               ▼
┌─────────────────────────────────────┐
│   FastAPI backend  (api/main.py)    │
│   POST /chat   — multi-turn RAG     │
│   POST /query  — single-turn RAG    │
│   GET  /health — liveness check     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   RAGEngine  (src/rag_engine.py)    │
│   VectorStore → Pinecone            │
│   LLMClient  → Groq                 │
└─────────────────────────────────────┘
```

Any HTTP client (curl, Postman, mobile app, another UI) can talk directly to the API.

### Two Separate Workflows

**Ingestion (one-time, run locally):**
```
msu_scraper/data/orgs/<slug>/
  profile.json + documents/*.txt
       ↓
ClubDataProcessor  →  typed chunks (profile / event / constitution)
       ↓
VectorStore.upsert_chunks()  →  Pinecone (llama-text-embed-v2, 1024 dims)
```

**Chat (real-time, multi-turn):**
```
Streamlit UI
    ↓ POST /chat  { question, conversation_history, vibe, filters }
FastAPI  →  RAGEngine.chat(question, conversation_history)
  ├── _rewrite_query()                ← LLM rewrites follow-ups into standalone search queries
  │                                      (skipped on first turn; uses last 4 messages only)
  ├── _extract_filters_from_query()   ← keyword-based, runs on rewritten query
  ├── VectorStore.search()            ← embed rewritten query via Pinecone inference API
  ├── _build_context_from_chunks()    ← formats [Source N] strings + citation dicts
  ├── messages = chat_history + current user message (original question) with injected context
  └── LLM.generate_with_history(messages, system_prompt)
    ↓
{ answer, citations, filters_applied, num_chunks }  →  Streamlit chat UI
```

### Key Design Decisions

**Headless API** — `RAGEngine` has no Streamlit dependency. The FastAPI layer adds Pydantic validation and HTTP transport. Any client can consume it. `API_BASE_URL` in `config.py` (default `http://localhost:8000`) tells the Streamlit app where the API lives.

**Chunk types** — Profile, event, and constitution chunks are stored separately with a `chunk_type` metadata field. This allows metadata-filtered retrieval (e.g., only event chunks when the query mentions "meeting" or "schedule"). Never mix chunk types in the same record.

**Pinecone integrated inference** — The app uses `pc.inference.embed()` with `llama-text-embed-v2` for query embedding, matching the model used at index time. Changing embedding models requires re-indexing all chunks from scratch.

**Flat metadata** — Pinecone requires flat records (no nested dicts). All metadata fields are at the top level. Lists of strings are supported (`categories`). Nested objects (like `events_by_month`) are serialized to JSON strings.

**Deterministic chunk IDs** — IDs follow `{slug}_{type}_{index}` patterns so re-ingestion upserts to the same IDs (idempotent). The checkpoint file `.ingest_checkpoint.txt` tracks completed slugs to allow resuming after rate-limit failures.

**Query rewriting** — On follow-up turns, `_rewrite_query()` makes a fast LLM call (temp=0, max 80 tokens) to turn ambiguous references ("tell me more about the first one", "what about their dues?") into a standalone search query before hitting Pinecone. The original question is still passed to the final LLM call so the answer sounds natural. Skipped on the first turn (no history = no ambiguity).

**Multi-turn conversation** — Streamlit `st.session_state` holds two stores: `messages` (full display state per turn) and `chat_history` (plain `{role, content}` pairs for the LLM). Context is injected only into the current user message — prior user messages in history are raw questions only. This keeps history lightweight.

**Vibe selector** — `_build_system_prompt(vibe)` returns one of three system prompts (`"scholar"`, `"buddy"`, `"nofilter"`). Each prompt includes an instruction to end every answer with one empathetic follow-up question. Retrieval is identical across all vibes; only the system prompt string changes.

**LLM abstraction** — `BaseLLMClient` in `src/llm_client.py` with `generate()` (single-turn) and `generate_with_history()` (multi-turn) methods. `GroqClient` implements both. Switch providers by changing `LLM_PROVIDER` in `.env` — no code changes needed.

**Prompt injection hardening** — Four lightweight defenses added 2026-05-24:
1. All system prompts append a confidentiality instruction so the LLM won't quote its own prompt when asked.
2. Retrieved Pinecone chunks are wrapped in `<club_data>` delimiters with an "untrusted data" framing in `_build_user_prompt()` — guards against indirect injection from poisoned club records.
3. User input is whitespace-normalized (`" ".join(prompt.split())`) in `app.py` before any LLM call — closes newline injection that could escape the rewrite prompt.
4. Input is capped at 500 characters in `app.py` and via `Field(max_length=500)` in `api/models.py` — prevents unbounded token spend in the query rewrite step.

**Evaluation is separate from prod** — `eval/` has its own `requirements-eval.txt`. RAGAS + eval deps are never installed on Streamlit Cloud. The evaluator LLM is Groq `llama-3.3-70b-versatile`; embeddings use local HuggingFace `all-MiniLM-L6-v2` (~90 MB, cached after first download). No OpenAI key required.

### File Roles

| File | Role |
|------|------|
| `app.py` | Streamlit chat UI — thin HTTP client, no RAG logic |
| `api/main.py` | FastAPI app — lifespan init, `/chat`, `/query`, `/health` routes |
| `api/models.py` | Pydantic request/response models for the API |
| `config.py` | Single source of truth for all env vars and constants (incl. `API_BASE_URL`) |
| `src/rag_engine.py` | Orchestrates retrieval + generation; owns system prompts and `chat()` method |
| `src/vector_store.py` | Pinecone upsert/search; handles embedding and metadata flattening |
| `src/llm_client.py` | LLM provider abstraction — `generate()` and `generate_with_history()` |
| `src/data_processing.py` | Converts raw scraper JSON into typed, token-bounded chunks |
| `ingest_data.py` | CLI for batch ingestion with checkpoint/resume |
| `eval/run_eval.py` | RAGAS evaluation runner — 6 metrics, saves timestamped JSON results |
| `eval/compare.py` | Diffs two eval result files side by side |
| `eval/dataset.json` | 8 test Q&A pairs with ground truth answers |

## Evaluation

RAGAS measures 6 metrics on each run:

| Metric | What it catches | Needs ground truth |
|--------|----------------|--------------------|
| Faithfulness | Hallucination — answer not grounded in retrieved context | No |
| Answer Relevancy | Off-topic answers | No |
| Context Precision | Irrelevant chunks retrieved | Yes |
| Context Recall | Missing chunks (retrieval gaps) | Yes |
| Answer Correctness | Factual errors vs. reference answer | Yes |
| Answer Similarity | Semantic drift from reference answer | Yes |

Run eval before/after: chunking changes, `top_k` tuning, system prompt edits, embedding model swaps. Results go to `eval/results/` (gitignored). The `eval/dataset.json` ground truths should be kept accurate — inaccurate ground truths corrupt the 4 metrics that depend on them.

## Deployment

**Local development** requires two processes: `uvicorn api.main:app --reload` + `streamlit run app.py`. Set `API_BASE_URL` in `.env` if the API runs on a different host/port.

**Streamlit Cloud** currently deploys only the Streamlit app. For full headless deployment, host the FastAPI backend separately (e.g., Railway, Render, or a VPS) and set `API_BASE_URL` to that URL in Streamlit Cloud secrets.
