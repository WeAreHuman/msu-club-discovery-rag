# MSU Club Discovery RAG Assistant

A Retrieval-Augmented Generation (RAG) system that helps Michigan State University
students discover clubs and organizations. Ask a natural-language question, get an
answer with citations sourced from actual club data.

**Stack**: Pinecone (llama-text-embed-v2) · Groq (Llama 3.3 70B) · Streamlit · Python

---

## Architecture

```
INGESTION (run once)
  msu_scraper/data/orgs/<slug>/profile.json + documents/*.txt
       |
  ClubDataProcessor  ->  3 chunk types: profile / event / constitution
       |
  Pinecone (llama-text-embed-v2, 1024 dims, integrated inference)

QUERY (real-time)
  User question -> embed -> cosine search -> top-k chunks -> LLM -> answer + citations
```

See [learnings.md](learnings.md) for full architecture decisions and trade-offs.

---

## Setup

### 1. Prerequisites

- Python 3.9+
- [msu_scraper](https://github.com/WeAreHuman/msu_scraper) output at `D:/msu_scraper/data/orgs/`
  (or set `SCRAPER_ORGS_DIR` in `.env`)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Pinecone index

1. Go to [pinecone.io](https://pinecone.io) and create an account
2. Create a new index:
   - **Name**: `msu-clubs-index`
   - **Model**: `llama-text-embed-v2` (Inference mode — Pinecone handles embedding)
   - **Metric**: cosine
3. Copy your API key

### 4. Get a Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) (free, no credit card)
2. Create an API key

### 5. Configure secrets

```bash
cp .env.example .env
```

Edit `.env`:
```env
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=msu-clubs-index
PINECONE_NAMESPACE=msu-clubs

LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key
LLM_MODEL=llama-3.3-70b-versatile

SCRAPER_ORGS_DIR=D:/msu_scraper/data/orgs
```

### 6. Verify setup

```bash
python config.py
```

---

## Ingest Data

Load club data from the scraper output into Pinecone:

```bash
python ingest_data.py
```

This is resumable — if it stops (rate limit, network error), re-run the same command.
Already-processed clubs are skipped automatically via `.ingest_checkpoint.txt`.

```bash
python ingest_data.py --club asa        # ingest one club by slug (for testing)
python ingest_data.py --clear           # wipe Pinecone namespace + checkpoint, start over
```

Expected output:
```
Clubs  : 1399 total  |  0 already done  |  1399 remaining
  Batch 1/1: 5 records  (total: 5)
  Batch 1/1: 3 records  (total: 3)
  ...
```

---

## Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

For Streamlit Cloud deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Test the System

```bash
python test_system.py
```

Runs 5 checks: configuration, document processing, Pinecone connectivity, LLM client,
and end-to-end RAG query.

---

## File Structure

```
msu-club-discovery-rag/
├── app.py                  # Streamlit UI (local + Streamlit Cloud)
├── config.py               # All settings via os.getenv()
├── ingest_data.py          # CLI: scraper output -> Pinecone
├── test_system.py          # Integration test script
├── requirements.txt
├── .env.example            # Secret keys template
├── src/
│   ├── data_processing.py  # ClubDataProcessor: profile / event / constitution chunks
│   ├── vector_store.py     # Pinecone upsert + search
│   ├── rag_engine.py       # RAG orchestration: filter -> retrieve -> generate
│   └── llm_client.py       # Groq / Anthropic LLM abstraction
├── docs/
│   └── sample_inputs.md    # Example MSU social media posts (test data)
├── learnings.md            # Architecture decisions, trade-offs, bugs fixed
└── DEPLOYMENT.md           # Streamlit Cloud + Docker deployment guide
```

---

## Chunk Types

| Type | Source | One per | Key metadata |
|------|--------|---------|--------------|
| `profile` | profile.json | club (may split if long) | org_name, categories, status, contact |
| `event` | profile.json events[] | event | event_id, event_date, event_venue, total_events |
| `constitution` | documents/*.txt | article section | article_num, article_title, doc_id |

All chunk types carry the club anchor (name, categories, status) so the LLM always
knows which club a retrieved chunk belongs to.

---

## Metadata Filters

The sidebar lets you filter results at the Pinecone level (before LLM):
- **Content type**: profile / event / constitution
- **Specific club**: exact `org_name` match

Programmatic example:
```python
rag.query_with_metadata_filter(
    question="when is the kickoff meeting?",
    chunk_type="event",
    org_name="Accessibility Club at MSU"
)
```

---

## Cost

- **Pinecone** free tier: 100,000 vectors (covers ~1,400 clubs)
- **Groq** free tier: ~14,000 requests/day for Llama 3.3 70B
- **Total**: $0 for typical usage
