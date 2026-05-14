# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG (Retrieval-Augmented Generation) app for discovering MSU student clubs. Students ask natural-language questions; the system retrieves relevant club data from Pinecone and generates answers via Groq/Llama. Deployed on Streamlit Cloud.

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
```

## Environment Setup

Copy `.env.example` to `.env` and fill in:
- `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` — required
- `LLM_PROVIDER=groq` and `GROQ_API_KEY` — recommended (free tier)
- `SCRAPER_ORGS_DIR` — path to the `msu_scraper/data/orgs/` directory

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

**Query (real-time, per request):**
```
User question
    ↓
RAGEngine.query()
  ├── _extract_filters_from_query()   ← keyword-based, steers to event chunks
  ├── VectorStore.search()            ← embed query via Pinecone inference API, cosine search
  ├── _build_context_from_chunks()    ← formats [Source N] strings + citation dicts
  └── LLM.generate(system_prompt, user_prompt)
    ↓
Answer + citations → Streamlit UI
```

### Key Design Decisions

**Chunk types** — Profile, event, and constitution chunks are stored separately with a `chunk_type` metadata field. This allows metadata-filtered retrieval (e.g., only event chunks when the query mentions "meeting" or "schedule"). Never mix chunk types in the same record.

**Pinecone integrated inference** — The app uses `pc.inference.embed()` with `llama-text-embed-v2` for query embedding, matching the model used at index time. Changing embedding models requires re-indexing all chunks from scratch.

**Flat metadata** — Pinecone requires flat records (no nested dicts). All metadata fields are at the top level. Lists of strings are supported (`categories`). Nested objects (like `events_by_month`) are serialized to JSON strings.

**Deterministic chunk IDs** — IDs follow `{slug}_{type}_{index}` patterns so re-ingestion upserts to the same IDs (idempotent). The checkpoint file `.ingest_checkpoint.txt` tracks completed slugs to allow resuming after rate-limit failures.

**Vibe selector** — `_build_system_prompt(vibe)` returns one of three system prompts (`"scholar"`, `"buddy"`, `"nofilter"`). Retrieval is identical across all vibes; only the system prompt string changes.

**LLM abstraction** — `BaseLLMClient` in `src/llm_client.py` with `GroqClient` and `AnthropicClient` implementations. Switch providers by changing `LLM_PROVIDER` in `.env` — no code changes needed.

### File Roles

| File | Role |
|------|------|
| `app.py` | Streamlit UI — sidebar filters, hero, search, results, vibe badge |
| `config.py` | Single source of truth for all env vars and constants |
| `src/rag_engine.py` | Orchestrates retrieval + generation; owns system prompts |
| `src/vector_store.py` | Pinecone upsert/search; handles embedding and metadata flattening |
| `src/llm_client.py` | LLM provider abstraction |
| `src/data_processing.py` | Converts raw scraper JSON into typed, token-bounded chunks |
| `ingest_data.py` | CLI for batch ingestion with checkpoint/resume |

## Deployment

Streamlit Cloud deploys from the `main` branch automatically on push. Changes on feature branches must be merged to `main` to take effect in production.
