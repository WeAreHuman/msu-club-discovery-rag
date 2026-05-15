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

### Conversation history (multi-turn)

Each query now passes prior turns as lightweight conversation history to the LLM.
The history only contains plain Q&A pairs (raw question text + assistant answer) —
the retrieved context blob is injected only for the current question. This keeps
the history payload small while still giving the LLM enough context for follow-ups
like "tell me more about the first one." See the Enhancement Log for implementation details.

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

---

### 2026-05-14 — Multi-turn Chatbot (Conversation Memory)

**What was built**

Converted the app from one-shot Q&A (text input + Search button → single answer) to a persistent chatbot. The user can ask follow-up questions like "how much are their dues?" or "what about that first club?" and the LLM understands the prior context.

**How it works**

Two session-state stores:
- `messages` — full display state per turn: `{role, content, citations, vibe, num_chunks, filters_applied}`. Used to re-render the full conversation on every Streamlit rerun.
- `chat_history` — lightweight LLM payload: plain `{role, content}` pairs where the user content is just the raw question (no context blob). Grows by 2 entries per turn.

On each new question:
```
User question
    ↓
RAGEngine.chat()
  ├── extract/apply filters (same as before)
  ├── VectorStore.search() for current question   ← fresh retrieval every turn
  ├── build context string from top-k chunks
  ├── messages = chat_history + {"role":"user", "content": context + question}
  └── LLM.generate_with_history(messages, system_prompt)
    ↓
Answer + citations displayed via st.chat_message("assistant")
chat_history updated with plain Q&A (no context blob in user turn)
```

Why inject context only for the current question and not prior turns? Because the LLM's prior assistant answers already referenced the retrieved context. Re-injecting old context blobs would balloon the prompt size quadratically. The plain question is enough for the LLM to understand what was asked.

**Files changed**
- `src/llm_client.py`: Added `generate_with_history(messages, system_prompt, ...)` abstract method + `GroqClient` implementation. Prepends system prompt, then passes the full messages list to the Groq API.
- `src/rag_engine.py`: Added `RAGEngine.chat()` method. Accepts `conversation_history`, `org_name`, `chunk_type` (explicit filter overrides), and all existing params. Builds LLM messages as `history + current_user_msg_with_context`.
- `app.py`: Replaced text-input + button pattern with `st.chat_input` + `st.chat_message`. Example chips use `st.session_state.pending_prompt` + `st.rerun()` to feed into the same processing path. "New Chat" button in sidebar resets both session-state stores. `render_assistant_message()` helper renders vibe badge + answer box + citations consistently for both live and history messages.

**Key design decision — where context lives in the message list**

```
[system prompt]
[user: "What clubs are good for beginners?"]          ← turn 1 (in history, no context)
[assistant: "Here are some options: ..."]             ← turn 1
[user: "Context: [Source 1]...\n\nQuestion: tell me more about the first one"]  ← current turn
```

Context is injected only in the current user message. Prior user messages in history are just the raw question. This is the right trade-off: the LLM can follow the thread, but history doesn't grow with every set of retrieved chunks.

**Trade-off to know**

History grows indefinitely during a session — no trimming. For very long conversations the prompt could exceed the model's context window. For a student project this is fine (sessions are short). In production, you'd trim to the last N turns or summarize old turns.

---

### 2026-05-14 — Empathetic Follow-up Suggestions + Sidebar Cleanup

**What was built**

Two improvements in one pass.

**1. Empathetic follow-up at the end of every response**

The problem: when a user asks something vague like "clubs for beginners", the RAG engine returns whatever chunks match — but it has no way of knowing what the user actually means by beginner-friendly (no dues? no experience required? casual time commitment? a specific domain like coding or art?). Without context, the retrieval stays generic.

The fix: updated all three system prompts in `_build_system_prompt()` to instruct the LLM to end every answer with one natural, empathetic follow-up question. The question is meant to draw out details that would make the next query more precise — things like area of interest, schedule, budget, or what a vague word means to them.

Each vibe gets a follow-up instruction that fits its tone:
- Scholar Mode: formal phrasing, asks for domain/preferences
- Spartan Buddy: casual, "like texting a friend" style
- No Filter: punchy, direct — "what's your major tho?"

Why one question only: multiple questions feels like a form. One question feels like a conversation.

Why in the system prompt and not injected post-hoc in app.py: keeping it in the prompt means the follow-up is aware of what was just said and can ask something genuinely relevant, not a generic template tacked on.

**2. Sidebar cleanup**

- Moved "New Chat" button to the very top of the sidebar and styled it as a primary button so it's always visible and easy to hit
- Removed emojis from the response style radio options (was "🎓 Scholar Mode", "🤝 Spartan Buddy", "🔥 No Filter Spartan") — labels are now plain text: "Scholar Mode", "Spartan Buddy", "No Filter"
- Renamed "Response Vibe" subheader to "Response Style" — slightly more neutral
- Collapsed "Content Type" and "Specific Club" filters under a single "Filters" subheader to reduce visual noise
- Removed the separate "Settings" subheader — the sources slider now stands alone with its label

Vibe badge icons inside the chat (the colored pill shown above each answer) are unchanged — those are visual indicators for the user, not labels, so they're fine.

**Files changed**
- `src/rag_engine.py`: Updated all three system prompt strings in `_build_system_prompt()` to include a follow-up instruction block at the end of each
- `app.py`: Rewrote `render_sidebar()` — New Chat at top, removed emojis from radio keys, consolidated filter subheaders

---

### 2026-05-14 — RAGAS Evaluation Pipeline

**What was built**

A local evaluation harness using [RAGAS](https://docs.ragas.io) to measure retrieval and generation quality across 6 standard RAG metrics. Runs locally on demand — not part of the deployed app.

**The 6 metrics**

| Metric | What it measures | Needs ground truth? |
|---|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? (no hallucination) | No |
| Answer Relevancy | Does the answer actually address the question? | No |
| Context Precision | Are the retrieved chunks relevant to the question? | Yes |
| Context Recall | Do the retrieved chunks cover what the ground truth says? | Yes |
| Answer Correctness | Is the answer factually correct vs. ground truth? | Yes |
| Answer Similarity | Is the answer semantically close to the ground truth? | Yes |

All scores are 0–1. Higher is better. The first two (faithfulness, answer relevancy) are the most important for catching hallucinations and off-topic answers.

**Architecture decision: where eval lives**

Inside the repo under `eval/` but with its own `eval/requirements-eval.txt`. It is NOT in the main `requirements.txt` — Streamlit Cloud must not install ragas, langchain-openai, datasets etc. The prod app stays lean; eval is a developer tool.

```
eval/
  requirements-eval.txt   ← ragas, langchain-openai, datasets, pandas
  dataset.json            ← 8 test Q&A pairs (question + ground_truth)
  run_eval.py             ← runs the full pipeline, saves to results/
  compare.py              ← diff two result JSON files side by side
  results/                ← timestamped JSON outputs (gitignored)
    .gitkeep
```

**Two-LLM setup**

The app generates answers using Groq (free tier). RAGAS needs its own LLM to score faithfulness, relevancy, and correctness — it uses OpenAI `gpt-4o-mini` + `text-embedding-3-small`. These are the evaluator models, completely separate from the generation model. This is standard practice: use a capable, well-calibrated model as a judge, regardless of what model the app uses.

Requires `OPENAI_API_KEY` in `.env` to run eval. Normal app keys (Pinecone, Groq) are also needed since it runs real queries.

**How the pipeline works**

```
dataset.json (8 questions + ground truths)
         ↓
For each question:
  RAGEngine.query()  →  answer + retrieved_chunks
                ↓
datasets.Dataset: {question, answer, contexts, ground_truth}
         ↓
ragas.evaluate(dataset, metrics=[...], llm=gpt-4o-mini)
         ↓
eval/results/<timestamp>[_label].json
  { summary: {metric: score}, per_question: [...] }
```

**How to run**

```bash
# First time setup
pip install -r eval/requirements-eval.txt

# Run baseline
python eval/run_eval.py --label baseline

# Run after a change (e.g. tweaking top_k)
python eval/run_eval.py --label top_k_8 --top-k 8

# Compare the two
python eval/compare.py eval/results/..._baseline.json eval/results/..._top_k_8.json
```

**When to run**

Not on every commit — each run costs OpenAI credits and takes ~2–3 minutes. Run it as a checkpoint before/after: changing chunking strategy, changing `top_k`, modifying system prompts, or swapping embedding models. Think of it like running a test suite before merging a significant change.

**Ground truths in dataset.json**

The 8 ground truth answers are written as general-purpose reference answers. They should be validated and refined over time as you learn what the actual Pinecone data says. Inaccurate ground truths will make context_recall and answer_correctness scores misleading. The other two metrics (faithfulness, answer_relevancy) don't depend on ground truth and are always reliable.

**Files created**
- `eval/requirements-eval.txt`
- `eval/dataset.json`
- `eval/run_eval.py`
- `eval/compare.py`
- `eval/results/.gitkeep`
- `.gitignore`: added `eval/results/*.json`

---

### 2026-05-14 — Upgraded Generation Model to Llama 4 Maverick

**What changed**

Switched `LLM_MODEL` in `.env` from `llama-3.3-70b-versatile` to
`meta-llama/llama-4-maverick-17b-128e-instruct`.

**Why**

Groq added Llama 4 models to their free tier. Maverick is a 17B MoE (Mixture of Experts)
model with 128 expert layers — the "17B" parameter count is misleading because MoE
activates many more effective parameters per token than a 70B dense model. Benchmarks
put it near GPT-4o quality.

No code change was needed. `LLM_MODEL` flows through `config.py` → `GroqClient` already,
so updating the env var is the only required change.

**Trade-off**

Free-tier rate limits are the same. Maverick is slightly slower than 70B dense on
simple prompts, but quality is noticeably better for multi-turn reasoning.

---

### 2026-05-14 — RAGAS Eval: Switched from OpenAI to Groq + HuggingFace

**What changed**

Original eval setup required `OPENAI_API_KEY` (paid). Switched to:
- **Evaluator LLM**: Groq `llama-3.3-70b-versatile` via `langchain-groq` (free)
- **Evaluator embeddings**: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
  via `langchain-huggingface` (~90 MB download on first run, cached after)

**Why**

No OpenAI account / free tier only. Using Groq for eval keeps the entire project free.

**Why use llama-3.3-70b for eval instead of the new Maverick generation model?**

Self-scoring bias — using the same model to both generate answers and judge them inflates
scores. A different, well-calibrated model as the judge gives more honest metrics. 70B is
also a well-tested instruction follower for NLI tasks that RAGAS relies on.

**Files changed**
- `eval/requirements-eval.txt`: removed `langchain-openai`, `openai`; added `langchain-groq`,
  `langchain-huggingface`, `sentence-transformers`
- `eval/run_eval.py`: replaced `ChatOpenAI` + `OpenAIEmbeddings` with `ChatGroq` +
  `HuggingFaceEmbeddings`; guard now checks `GROQ_API_KEY`; result JSON records `eval_llm`
  and `eval_embeddings` fields

**Known limitation**

HuggingFace `all-MiniLM-L6-v2` is a much smaller embedding model than OpenAI's
`text-embedding-3-small`. Answer Relevancy scores may differ from what you'd get with
the OpenAI embeddings — scores across runs are still comparable to each other as long as
the same embedding model is used consistently.
