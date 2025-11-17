# Quick Setup Guide - MSU Club Discovery RAG

This guide will help you get the system running in ~10 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Step 1: Install Dependencies (2 minutes)

```bash
cd RAG_Suraj
pip install -r requirements.txt
```

If you encounter any errors, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: Get API Keys (5 minutes)

### 2.1 Pinecone (Vector Database) - FREE

1. Go to https://www.pinecone.io/
2. Click "Sign Up" (use Google/GitHub for quick signup)
3. Once logged in, click "Create Index"
4. Configure:
   - **Name**: `msu-clubs-index`
   - **Dimensions**: Choose "Inference" mode
   - **Model**: Select `llama-text-embed-v2`
   - **Metric**: cosine (default)
5. Click "Create Index"
6. Go to "API Keys" tab
7. Copy your API key

### 2.2 Groq (LLM) - FREE & FAST

1. Go to https://console.groq.com/
2. Sign up (free, no credit card required)
3. Once logged in, go to "API Keys"
4. Click "Create API Key"
5. Give it a name (e.g., "MSU RAG") and create
6. Copy the API key immediately (you won't see it again)

---

## Step 3: Configure Environment (1 minute)

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any text editor
```

Update these lines in `.env`:
```env
PINECONE_API_KEY=your_actual_pinecone_key_here
GROQ_API_KEY=your_actual_groq_key_here
```

Save and close (Ctrl+X, then Y, then Enter if using nano)

---

## Step 4: Verify Configuration (30 seconds)

```bash
python config.py
```

You should see:
```
✓ Configuration validated successfully
  - LLM Provider: groq
  - LLM Model: llama-3.3-70b-versatile
  - Pinecone Index: msu-clubs-index
  - Chunk Size: 300 tokens (overlap: 50)
```

If you see errors, check your API keys in `.env`

---

## Step 5: Ingest Sample Data (1 minute)

```bash
python ingest_data.py
```

You should see:
```
================================================================================
MSU CLUB DISCOVERY - DATA INGESTION
================================================================================

✓ Configuration validated successfully
📁 Input directory: /path/to/data/raw
📄 Found 1 document(s)

--------------------------------------------------------------------------------
PROCESSING DOCUMENTS
--------------------------------------------------------------------------------

📄 Processing: accessibility_club.txt
  ✓ Extracted 7234 characters
  ✓ Metadata: Accessibility Club at Michigan State University
    - Dues: $10.0
  ✓ Created 15 chunks (~300 tokens each)

--------------------------------------------------------------------------------
UPLOADING TO PINECONE
--------------------------------------------------------------------------------

📤 Upserting 15 chunks to Pinecone...
  ✓ Batch 1: 15 chunks upserted
✓ Total upserted: 15 chunks

================================================================================
INGESTION COMPLETE
================================================================================
✓ Documents processed: 1
✓ Chunks created: 15
✓ Chunks uploaded: 15
```

---

## Step 6: Run the Application (30 seconds)

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

---

## Step 7: Test the System

Try these queries in the web interface:

1. **Basic info**: "What is the Accessibility Club?"
2. **Specific details**: "How much are the dues?"
3. **Meeting info**: "When does the Accessibility Club meet?"

Each answer should include:
- ✅ Generated answer
- ✅ Source citations with relevance scores
- ✅ Expandable source details

---

## Troubleshooting

### Error: "Pinecone index not found"

**Solution**: Make sure you created the index in Step 2.1 with the exact name `msu-clubs-index`

### Error: "GROQ_API_KEY is not set"

**Solution**: Check your `.env` file. Make sure the key is after the `=` with no spaces or quotes.

### Error: "No chunks created"

**Solution**: Make sure `data/raw/accessibility_club.txt` exists. If not:
```bash
# The file should already be there, but if not, check the data/raw directory
ls data/raw/
```

### Streamlit won't start

**Solution**: Make sure you installed all dependencies:
```bash
pip install streamlit
streamlit run app.py
```

### Slow responses

**Solution**: Try switching to the faster model. Edit `.env`:
```env
LLM_MODEL=llama-3.1-8b-instant
```

---

## Next Steps

### Add More Club Documents

1. Place PDF or TXT files in `data/raw/`
2. Run ingestion again:
```bash
python ingest_data.py --clear
```
The `--clear` flag removes old data first (optional).

### Test Different Queries

Try queries with filters:
- "What clubs have dues under $15?"
- "Tell me about clubs for beginners"
- "Which clubs meet weekly?"

### Customize the System

Edit `config.py` to adjust:
- Chunk size (default: 300 tokens)
- Chunk overlap (default: 50 tokens)
- Number of results (default: 5)
- LLM temperature (default: 0.3)

---

## Success Checklist

- ✅ Installed dependencies
- ✅ Created Pinecone index with llama-text-embed-v2
- ✅ Got Groq API key
- ✅ Configured .env file
- ✅ Verified configuration
- ✅ Ingested sample data (15 chunks)
- ✅ Ran Streamlit app
- ✅ Tested queries successfully

---

## Getting Help

If you encounter issues:

1. Check the detailed documentation: `README.md`
2. Review the code flow: `FLOW_DOCUMENTATION.md`
3. Look for error messages in the terminal
4. Verify API keys are correct in `.env`

---

## Optional: Using Claude Instead of Groq

If you have an Anthropic API key and want to use Claude:

1. Get API key from https://console.anthropic.com/
2. Edit `.env`:
```env
ANTHROPIC_API_KEY=your_anthropic_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5-20250929
```
3. Restart the app

**Note**: Claude is a paid API, while Groq is free.

---

## Estimated Costs

- **Pinecone Free Tier**: 100,000 vectors (plenty for this project)
- **Groq Free Tier**: ~14,000 requests/day for 70B model
- **Total Monthly Cost**: $0 for typical usage!

You can process hundreds of club documents and handle thousands of queries for free.

---

## Complete! 🎉

You now have a fully functional RAG system for MSU club discovery. Enjoy exploring!
