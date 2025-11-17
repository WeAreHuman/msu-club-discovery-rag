# Quick Start - 5 Minute Setup

Get the MSU Club Discovery RAG system running in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Get Free API Keys

### Pinecone (30 seconds)
- Go to https://www.pinecone.io/ → Sign up
- Create index: name=`msu-clubs-index`, model=`llama-text-embed-v2`
- Copy API key

### Groq (30 seconds)
- Go to https://console.groq.com/ → Sign up
- Create API key
- Copy it

## 3. Configure

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## 4. Ingest Data

```bash
python ingest_data.py
```

## 5. Run!

```bash
streamlit run app.py
```

Open http://localhost:8501 and try:
- "What is the Accessibility Club?"
- "How much are the dues?"

## Need Help?

See `SETUP_GUIDE.md` for detailed instructions.

## Test Everything

```bash
python test_system.py
```

Should show ✅ for all 5 tests.
