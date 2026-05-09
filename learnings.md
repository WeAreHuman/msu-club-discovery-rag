# MSU Club Discovery RAG — Learnings & Architecture Trace

A personal learning document covering every architectural decision, trade-off, and bug
encountered while building this system. Written to be interview-ready.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Source — msu_scraper](#3-data-source--msu_scraper)
4. [Chunking Strategy](#4-chunking-strategy)
5. [Vector Store — Pinecone](#5-vector-store--pinecone)
6. [Retrieval — Query Flow](#6-retrieval--query-flow)
7. [LLM Layer](#7-llm-layer)
8. [Ingestion Pipeline — Operational Design](#8-ingestion-pipeline--operational-design)
9. [Bugs Encountered and How They Were Fixed](#9-bugs-encountered-and-how-they-were-fixed)
10. [Key Trade-offs Worth Knowing](#10-key-trade-offs-worth-knowing)

---

## 1. What This System Does

MSU has 1,400+ student clubs. Students have no good way to discover which clubs match
their interests. This system lets a student ask a natural-language question like
"Are there any clubs focused on accessibility or disability advocacy?" and get an
accurate answer with citations.

This is a classic RAG (Retrieval-Augmented Generation) problem:
- We can't put 1,400 clubs into an LLM's context window every query
- The LLM doesn't know MSU's specific clubs from training data
- We need answers grounded in real data, not hallucinations

RAG solves this: embed club data as vectors, retrieve the relevant chunks at query time,
pass only those chunks to the LLM as context.

---

## 2. High-Level Architecture

```
INGESTION (one-time, resumable)
  msu_scraper/data/orgs/<slug>/
    profile.json          <- club info, events list, officers
    documents/<id>.txt    <- constitution/bylaws plain text
         |
  ClubDataProcessor       <- produces typed chunks (profile / event / constitution)
         |
  VectorStore.upsert_chunks()
         |
  Pinecone index (llama-text-embed-v2, 1024 dims, integrated inference)

QUERY (per request, real-time)
  User question
         |
  RAGEngine.query()
    +-- extract metadata filters from question text
    +-- VectorStore.search()  <- embed query, cosine similarity search
    +-- build context string from top-k chunks
    +-- LLM.generate(context + question)
         |
  Answer + citations -> Streamlit UI
```

Two distinct workflows: ingestion (run once, resumes from checkpoint) and query (real-time).

---

## 3. Data Source — msu_scraper

The scraper writes one directory per club:

```
orgs/
  accessibility-club/
    profile.json          <- everything: name, categories, events[], documents[], officers[]
    documents/
      12345.txt           <- constitution text, pre-extracted from PDF/DOCX
```

Key fields from `profile.json`:
```json
{
  "org_id": 12345,
  "name": "Accessibility Club at MSU",
  "slug": "accessibility-club",
  "categories": ["Academic", "Service"],
  "status": "Active",
  "description": "<p>HTML description...</p>",
  "contact": { "email": "...", "website": "..." },
  "events": [
    { "event_id": 99, "title": "Kickoff", "start_datetime": "2025-09-01T18:00:00", ... }
  ],
  "documents": [{ "doc_id": 12345, "title": "Constitution" }],
  "officers": [{ "role": "President", "name": "Jane Doe" }]
}
```

The scraper is a separate project. This RAG system only consumes its output.

---

## 4. Chunking Strategy

### Why Three Chunk Types?

A club has three fundamentally different kinds of information:
- **profile** — general identity, description, contact info, officers; changes rarely
- **event** — time-bound structured data; changes frequently; one chunk per event
- **constitution** — legal/policy document; article-structured; rarely changes

Mixing these in the same chunk would hurt retrieval precision. If someone asks "when is
the next robotics club meeting?", we want to hit event chunks, not the constitution.
Keeping types separate lets us filter at retrieval time using Pinecone metadata filters.

### Profile Chunks — The Anchor-Body Pattern

**Problem**: A long club description may need to be split into multiple chunks. If chunk 2
is just a paragraph about officer election procedures, it no longer contains the club's
name — the LLM can't identify which club it's about.

**Solution**: Anchor-body pattern. Every profile chunk, regardless of how many pieces the
body is split into, starts with the same anchor:

```
Organization: Accessibility Club at MSU
Categories: Academic, Service
Status: Active

[body content here — description, contact, officers]
```

The anchor is computed first, its token cost subtracted from the budget, and then the body
is split into pieces that fit within `CHUNK_SIZE - anchor_tokens`. Every piece gets the
anchor prepended before storing. This guarantees: no matter which chunk is retrieved, it
always identifies the club.

### Event Chunks

One chunk per event — no splitting needed because events are short structured records.
The chunk text includes: organization name, event title, date range, venue, categories,
and description. Metadata includes aggregate stats: `total_events` (int) and
`events_by_month` (JSON string of `{"YYYY-MM": count}` pairs) so the LLM can answer
"how active is this club overall?"

### Constitution Chunks — Article-Aware Splitting

**Problem**: Constitutions use `Article I:`, `Article II:` structure. Splitting blindly
would cut across article boundaries and mix two articles in one chunk.

**Solution**: Parse article boundaries first with a regex lookahead:
```python
re.compile(r'(?=Article\s*[IVXLCDM]+\s*[:\-])', re.IGNORECASE)
```
This splits at article headings without consuming them. Each article gets a prefix:
```
Organization: Accessibility Club at MSU
Constitution — Article III: Membership
```
If any article body exceeds 350 tokens (`CONSTITUTION_MAX_TOKENS`), it's split further
with overlap. Text before Article I becomes Article "0" (preamble).

### Token Counting

We use `tiktoken` with `cl100k_base` to count tokens accurately. Character-based
splitting would be wrong because 1 token ≠ 1 character. `RecursiveCharacterTextSplitter`
(LangChain) handles the actual splitting but uses our `_count_tokens` as its length
function, so the budget is in tokens not chars.

Splitting hierarchy: paragraph breaks -> line breaks -> sentence ends -> spaces -> chars.

### Chunk IDs — Deterministic for Idempotency

IDs are derived from metadata, not random:
```
profile      -> {slug}_profile_{chunk_index}
event        -> {slug}_event_{event_id}
constitution -> {slug}_constitution_{article_num}_{chunk_index}
```

Re-running ingestion for a club that's already in Pinecone will upsert to the same IDs
and overwrite cleanly — no duplicates. This property is called idempotency.
If IDs were random UUIDs, every re-run would create duplicate records.

---

## 5. Vector Store — Pinecone

### Why Pinecone Integrated Inference?

We use `llama-text-embed-v2` (1024 dims), hosted by Pinecone. The alternative is running
a local embedding model (e.g., `sentence-transformers`). The hosted approach:
- No GPU required locally
- No model download or versioning concerns
- Same model guaranteed at both upsert and query time
- Simpler code: send text, Pinecone embeds and stores it

### Upsert — Flat Record Format

Pinecone SDK v7 with integrated inference uses `index.upsert_records()`. Records must be
flat — no nested "metadata" dict:

```python
{
  "_id": "accessibility-club_profile_0",  # primary key
  "text": "Organization: ...",             # the field Pinecone embeds
  "org_name": "Accessibility Club",       # all metadata at top level
  "chunk_type": "profile",
  "categories": ["Academic", "Service"],  # list[str] -> supports $in filter
}
```

Pinecone only supports scalar metadata (str/int/float/bool) and list[str].
Nested dicts are silently dropped. This is why `events_by_month` is serialized
as a JSON string rather than a nested object.

The API uses keyword-only arguments:
```python
index.upsert_records(records=records, namespace=self.namespace)
# NOT positional: index.upsert_records(records, namespace)  <- TypeError
```

### Query — Two-Step Embedding

For retrieval we embed the query with Pinecone's inference API, then query with the vector:
```python
embeddings = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[query],
    parameters={"input_type": "query", "truncate": "END"},
)
results = index.query(vector=embeddings[0].values, top_k=k, filter=..., include_metadata=True)
```

Why not use `index.search()` with text? Because `index.query()` (which supports metadata
filters) requires a raw vector. So we embed once, then query.

### Metadata Filters

Pinecone supports Mongo-style filter syntax applied server-side:
```python
{"chunk_type": {"$eq": "event"}}
{"categories": {"$in": ["Academic"]}}
```

---

## 6. Retrieval — Query Flow

Inside `RAGEngine.query()`:

1. **Filter extraction** — keyword-match the question for terms like "event", "meeting",
   "constitution", "bylaw" and build a Pinecone filter dict.

2. **Vector search** — `VectorStore.search(query, top_k, filters)` returns
   `[{"id", "score", "text", "metadata"}, ...]` sorted by cosine similarity.

3. **Context building** — format chunks as:
   ```
   [Source 1] Accessibility Club (profile):
   Organization: Accessibility Club...

   [Source 2] Robotics Club (event):
   ...
   ```

4. **LLM call** — system prompt instructs the model to answer only from context and cite
   with `[Source N]` markers.

5. **Response** — `{"answer": str, "citations": list, "retrieved_chunks": list}`.

---

## 7. LLM Layer

### Provider Abstraction

```
BaseLLMClient (ABC)
  .generate(prompt, system_prompt, temperature, max_tokens) -> str
    +-- GroqClient        (llama-3.3-70b-versatile, free tier)
    +-- AnthropicClient   (claude-*, paid)
```

`get_llm_client()` reads `config.LLM_PROVIDER` and returns the right client.
Switching providers requires only changing the env var — no code changes.

### Config vs. Environment

All values flow through `config.py` via `os.getenv()`. Works in three environments:
- **Local**: `.env` file loaded by `python-dotenv`
- **Streamlit Cloud**: Secrets dashboard auto-exposed as env vars
- **Docker**: `-e` flags

One `app.py`, one `config.py`. No dual-file pattern needed because Streamlit Cloud
exposes secrets as standard environment variables — `os.getenv()` works everywhere.

---

## 8. Ingestion Pipeline — Operational Design

### Checkpoint System

Ingesting 1,400 clubs takes time and can be interrupted by API rate limits (429).
`ingest_data.py` processes one club at a time:

```python
for each club_dir:
    chunks = processor.process_club_dir(club_dir)    # ~3-20 chunks
    upserted = vector_store.upsert_chunks(chunks)
    if upserted == len(chunks):
        save_checkpoint(club_dir.name)               # append slug to .ingest_checkpoint.txt
```

Re-run skips any slug already in `.ingest_checkpoint.txt`. Ingestion is resumable
from the exact club where it stopped.

### Partial Upsert Detection

`upsert_chunks()` returns the count of chunks actually upserted. If a batch fails,
the function returns early with the partial count:
```python
if upserted != len(chunks):
    print("[STOP] Partial upsert")
    break
```
The failed club is not checkpointed, so it retries from scratch on the next run
(safe because chunk IDs are deterministic — no duplicates).

### CLI Reference

```bash
python ingest_data.py                          # resume from checkpoint
python ingest_data.py --club asa               # single club by slug (debug)
python ingest_data.py --clear                  # wipe namespace + checkpoint, restart
python ingest_data.py --scraper-orgs-dir PATH  # override scraper dir
```

---

## 9. Bugs Encountered and How They Were Fixed

Real bugs, real fixes — good interview material about debugging production systems.

### Bug 1: ImportError — Wrong Class Name

**Error**: `ImportError: cannot import name 'DocumentProcessor' from 'src.data_processing'`

**Cause**: `test_system.py` still imported `DocumentProcessor` (old name). The class had
been renamed to `ClubDataProcessor` during the chunking rewrite.

**Fix**: Updated all import sites. **Lesson**: When renaming a class, grep for all
import sites before committing.

---

### Bug 2: UnicodeEncodeError on Windows

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character '✓'`

**Cause**: Windows terminal defaults to cp1252. Unicode symbols (checkmarks, arrows)
that are outside cp1252 fail at print time — silently unless you see the terminal.

**Fix**: Added to the top of `config.py` (imported by everything):
```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
```
**Lesson**: `config.py` is the right place — it's always the first import, so the fix
propagates to all modules automatically.

---

### Bug 3: Wrong Pinecone Upsert API

**Error**: `Index.upsert() got an unexpected keyword argument 'documents'`

**Cause**: The original code used `index.upsert(documents=[...])` — an API that doesn't
exist in Pinecone SDK v7. The SDK changed significantly between v2 and v7.

**Fix**: Switched to `index.upsert_records(records=records, namespace=self.namespace)`
with flat records (no nested metadata dict, use `_id` key not `id`).

---

### Bug 4: upsert_records Positional Argument Error

**Error**: `Index.upsert_records() takes 1 positional argument but 3 were given`

**Cause**: `index.upsert_records(records, namespace)` — positional args. The function
signature uses `*` to enforce keyword-only arguments.

**Fix**: `index.upsert_records(records=records, namespace=self.namespace)`

**Lesson**: When a library function fails with unexpected arg count, check if its
signature uses `*` to make all params keyword-only.

---

### Bug 5: Silent Semantic Failure (Dimension Mismatch)

**Symptom**: Search always returned the same wrong results regardless of query.

**Cause**: The original `search()` used local `sentence-transformers` (768 dims) to embed
queries, then queried a Pinecone index built with `llama-text-embed-v2` (1024 dims).
The 768-dim vectors were zero-padded to 1024 — completely destroying semantic meaning.
No exception was thrown.

**Fix**: Use `pc.inference.embed()` with `llama-text-embed-v2` for query embedding —
the exact same model used at index time.

**Lesson**: The model used to embed documents MUST match the model used to embed queries.
A mismatch is a silent bug — queries just return garbage results.

---

### Bug 6: Stale Metadata Field Names

**Symptom**: Citations showed empty club names; filter sidebar had no effect.

**Cause**: Old schema had fields `club_name`, `source_file`, `dues`, `meeting_frequency`.
After the chunking rewrite these became `org_name`, `chunk_type`, and the others were
removed. `rag_engine.py` and `app.py` still referenced the old names.

**Fix**: Grep for all old field names, update everywhere.

Old -> New: `club_name` -> `org_name`, `source_file` -> `chunk_type`, `dues`/`meeting_frequency` -> removed.

**Lesson**: Schema changes have a blast radius — search every file, not just the
module that was rewritten.

---

### Bug 7: Pinecone 429 Rate Limit Mid-Ingestion (No Resume)

**Error**: `429 Too Many Requests` after ~34k of 48k chunks were uploaded.

**Cause**: Original ingest loaded ALL clubs into memory first, then upserted in one giant
loop. One 429 error meant starting over from scratch — 34k chunks of work lost.

**Fix**: Process one club at a time. Write slug to checkpoint file after each successful
upsert. Re-run automatically skips already-done clubs.

**Lesson**: Any long-running batch job talking to external APIs needs checkpoint/resume.
Assume failures will happen.

---

## 10. Key Trade-offs Worth Knowing

### Chunking granularity

Smaller chunks -> more precise retrieval, less context per chunk.
Larger chunks -> more context, more retrieval noise.

300 tokens (profile) and 350 tokens (constitution) were set by the chunking strategy
spec, not tuned by experimentation. In production, you'd measure retrieval accuracy
with different chunk sizes and pick empirically.

### Embedding model lock-in

The index is built with `llama-text-embed-v2`. Switching models requires re-indexing
all 48k+ chunks from scratch. Embedding model choice is a consequential upfront decision
— changing it later is expensive.

### Stateless queries (no conversation memory)

Each query is independent. The LLM sees only the retrieved chunks and the current
question — not previous turns. This keeps things simple and avoids prompt length
issues, but means follow-up questions like "tell me more about the first one" won't work.

### Groq free tier

~14,000 requests/day for the 70B model. Fine for a student project. For real campus
traffic, you'd need a paid API. The provider abstraction makes swapping easy.

### Hosted vs. local embeddings

Pinecone hosted inference: no GPU, no model management, but adds API latency and uses
credits. For this project (1,400 clubs, one-time batch ingestion), hosted is the right
trade-off. For high-throughput ingestion at scale, a local model would be faster.

---

## Enhancement Log

### 2026-05-08 — Vibe Selector (Prompt Personalization)

**What was built**

A mood toggle in the sidebar with 3 options that change how the LLM responds. Same vector retrieval, same citations, different system prompt personality.

The 3 vibes:
- 🎓 Scholar Mode — formal, thorough, always cites (original behavior, unchanged)
- 🤝 Spartan Buddy — chill upperclassman energy, casual, reads between the lines
- 🔥 No Filter Spartan — gen-Z energy, acknowledges the real human motivation behind questions, direct

**How it works (pseudo-code)**

```
sidebar radio → vibe key ("scholar" | "buddy" | "nofilter")
       ↓
render_sidebar() returns vibe in filters dict
       ↓
app.py passes vibe= to rag_engine.query()
       ↓
_build_system_prompt(vibe) returns one of 3 system prompt strings
       ↓
LLM.generate(user_prompt, system_prompt=vibe_prompt)
       ↓
answer displayed + small personality badge shown above answer box
```

**Files changed**
- `src/rag_engine.py`: `_build_system_prompt()` now takes `vibe: str = "scholar"`. Both `query()` and `query_with_metadata_filter()` accept and forward `vibe`.
- `app.py`: `st.radio` in sidebar for vibe selection, `vibe` passed through filters dict to query calls, `VIBE_META` dict drives a colored personality badge rendered above the answer.

**Key insight**

Retrieval is completely vibe-agnostic — the same Pinecone vector search runs regardless of mode. Only the system prompt string changes. This means zero extra latency, zero extra API calls. Personality is just a string you hand to the LLM.

**Trade-off to know**

"No Filter Spartan" has the most personality but is also the hardest prompt to control precisely — edge cases or ambiguous questions may need prompt tuning over time. Scholar Mode is the safest for accuracy-critical use.
