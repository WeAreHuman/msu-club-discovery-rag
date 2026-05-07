# Deployment Guide

## How configuration works

The app uses a single `config.py` that reads from environment variables via `os.getenv()`.
This works identically in both environments:

| Environment | How secrets are supplied |
|---|---|
| Local | `.env` file loaded by `python-dotenv` |
| Streamlit Cloud | Secrets dashboard → auto-exposed as env vars |

There is one `app.py` entry point for both.

---

## Local Development

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure secrets
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=msu-clubs-index
PINECONE_NAMESPACE=msu-clubs
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
LLM_MODEL=llama-3.3-70b-versatile
SCRAPER_ORGS_DIR=D:/msu_scraper/data/orgs
```

### 3. Ingest data
```bash
python ingest_data.py
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

### 1. Push to GitHub
```bash
git add .
git commit -m "deploy"
git push origin main
```

### 2. Create app on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository and branch
4. Set **Main file** to `app.py`

### 3. Add secrets
In app settings → Secrets, add the same keys as your `.env`:
```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "msu-clubs-index"
PINECONE_NAMESPACE = "msu-clubs"
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-groq-api-key"
LLM_MODEL = "llama-3.3-70b-versatile"
```

Streamlit Cloud exposes these as environment variables, so `config.py` picks them up
via `os.getenv()` — no code changes needed between environments.

---

## Docker (Self-Hosted)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Pass secrets as environment variables:
```bash
docker run -p 8501:8501 \
  -e PINECONE_API_KEY=... \
  -e GROQ_API_KEY=... \
  -e PINECONE_INDEX_NAME=msu-clubs-index \
  -e PINECONE_NAMESPACE=msu-clubs \
  -e LLM_PROVIDER=groq \
  -e LLM_MODEL=llama-3.3-70b-versatile \
  msu-club-app
```

---

## File structure

```
project/
├── app.py                # Single Streamlit entry point (local + cloud)
├── config.py             # Reads env vars — works everywhere
├── ingest_data.py        # CLI: load scraper output into Pinecone
├── requirements.txt
├── .env                  # Local secrets (gitignored)
├── .env.example          # Secrets template
├── src/
│   ├── data_processing.py
│   ├── vector_store.py
│   ├── rag_engine.py
│   └── llm_client.py
└── .streamlit/
    └── config.toml       # Streamlit UI config (theme, etc.) — safe to commit
```

---

## Troubleshooting

**"API key not found" on Streamlit Cloud**
→ Check the Secrets panel in your app settings. Key names must match exactly (case-sensitive).

**Slow search results**
→ Check Pinecone index status and verify the namespace has data (`python ingest_data.py --club asa` to test with one club).

**App crashes on startup**
→ Run locally first: `streamlit run app.py --logger.level=debug`
