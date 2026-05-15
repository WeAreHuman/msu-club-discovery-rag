# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG (Retrieval-Augmented Generation) chatbot for discovering MSU student clubs. Students have multi-turn conversations; the system retrieves relevant club data from Pinecone and generates answers via Groq/Llama. Deployed on Streamlit Cloud.

## Commands

```bash
# Run the app locally
streamlit run app.py

# Ingest club data into Pinecone (resumes from checkpoint automatically)
python ingest_data.py

# Ingest a single club by slug (for testing)
python ingest_data.py --club accessibility-club

# Wipe Pinecone namespace + checkpoint and restart ingestion from scratch
python ingest_data.py --clear

# Run system tests
python test_system.py

# Run RAGAS evaluation (requires eval deps + OPENAI_API_KEY in .env)
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
- `OPENAI_API_KEY` — required only for running RAGAS eval (not used by the app)

On Streamlit Cloud, secrets are set in the dashboard and exposed as env vars — no special handling needed; `os.getenv()` works identically.

## Architecture

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
User message
    ↓
RAGEngine.chat(question, conversation_history)
  ├── _extract_filters_from_query()   ← keyword-based, steers to event chunks
  ├── VectorStore.search()            ← embed query via Pinecone inference API, cosine search
  ├── _build_context_from_chunks()    ← formats [Source N] strings + citation dicts
  ├── messages = chat_history + current user message with injected context
  └── LLM.generate_with_history(messages, system_prompt)
    ↓
Answer + citations + follow-up suggestion → Streamlit chat UI
```

### Key Design Decisions

**Chunk types** — Profile, event, and constitution chunks are stored separately with a `chunk_type` metadata field. This allows metadata-filtered retrieval (e.g., only event chunks when the query mentions "meeting" or "schedule"). Never mix chunk types in the same record.

**Pinecone integrated inference** — The app uses `pc.inference.embed()` with `llama-text-embed-v2` for query embedding, matching the model used at index time. Changing embedding models requires re-indexing all chunks from scratch.

**Flat metadata** — Pinecone requires flat records (no nested dicts). All metadata fields are at the top level. Lists of strings are supported (`categories`). Nested objects (like `events_by_month`) are serialized to JSON strings.

**Deterministic chunk IDs** — IDs follow `{slug}_{type}_{index}` patterns so re-ingestion upserts to the same IDs (idempotent). The checkpoint file `.ingest_checkpoint.txt` tracks completed slugs to allow resuming after rate-limit failures.

**Multi-turn conversation** — `st.session_state` holds two stores: `messages` (full display state per turn) and `chat_history` (plain `{role, content}` pairs for the LLM). Context is injected only into the current user message — prior user messages in history are raw questions only. This keeps history lightweight.

**Vibe selector** — `_build_system_prompt(vibe)` returns one of three system prompts (`"scholar"`, `"buddy"`, `"nofilter"`). Each prompt includes an instruction to end every answer with one empathetic follow-up question. Retrieval is identical across all vibes; only the system prompt string changes.

**LLM abstraction** — `BaseLLMClient` in `src/llm_client.py` with `generate()` (single-turn) and `generate_with_history()` (multi-turn) methods. `GroqClient` implements both. Switch providers by changing `LLM_PROVIDER` in `.env` — no code changes needed.

**Evaluation is separate from prod** — `eval/` has its own `requirements-eval.txt`. RAGAS + langchain-openai are never installed on Streamlit Cloud. The evaluator LLM is OpenAI `gpt-4o-mini`; the app LLM is Groq. These are intentionally different.

### File Roles

| File | Role |
|------|------|
| `app.py` | Streamlit chat UI — sidebar, hero, `st.chat_message` rendering, session state |
| `config.py` | Single source of truth for all env vars and constants |
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

Streamlit Cloud deploys from the `main` branch automatically on push. Changes on feature branches must be merged to `main` to take effect in production.
