# MSU Club Discovery RAG Assistant

A Retrieval-Augmented Generation (RAG) system that helps Michigan State University students discover and learn about student clubs and organizations. The system answers questions with verifiable citations, supports metadata filtering, and provides personalized recommendations.

## Features

- **Semantic Search**: Find relevant club information using natural language queries
- **Citation Support**: Every answer includes source citations with relevance scores
- **Metadata Filtering**: Filter clubs by dues, meeting frequency, and other criteria
- **Free & Open Source**: Uses free LLM APIs (Groq with Llama 3.3) and Pinecone free tier
- **Auto-Extraction**: Automatically extracts club metadata from documents
- **Interactive Web UI**: Clean Streamlit interface for easy searching

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Streamlit Web App)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RAG ENGINE                              │
│  - Query Processing                                             │
│  - Filter Extraction                                            │
│  - Context Building                                             │
│  - Response Generation                                          │
└──────────┬─────────────────────────────────┬────────────────────┘
           │                                 │
           ▼                                 ▼
┌──────────────────────────┐    ┌───────────────────────────────┐
│   VECTOR STORE           │    │      LLM CLIENT               │
│   (Pinecone)             │    │   (Groq / Anthropic)          │
│                          │    │                               │
│ - Semantic Search        │    │ - Llama 3.3 (Groq - FREE)    │
│ - Metadata Filtering     │    │ - Claude (Anthropic - PAID)  │
│ - llama-text-embed-v2    │    │                               │
└──────────────────────────┘    └───────────────────────────────┘
           ▲
           │
           │ (Data Ingestion)
           │
┌──────────┴───────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSOR                            │
│  - Text Extraction (PyMuPDF)                                     │
│  - Text Cleaning                                                 │
│  - Chunking (~300 tokens, 50 overlap)                           │
│  - Metadata Extraction                                           │
└──────────────────────────────────────────────────────────────────┘
           ▲
           │
┌──────────┴───────────────────┐
│   RAW DOCUMENTS              │
│   (PDFs, TXT files)          │
│   - Club Constitutions       │
│   - One-Pagers               │
│   - Officer Submissions      │
└──────────────────────────────┘
```

## Project Structure

```
RAG_Suraj/
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Document extraction, cleaning, chunking
│   ├── vector_store.py       # Pinecone vector database operations
│   ├── llm_client.py         # LLM client (Groq/Anthropic)
│   └── rag_engine.py         # RAG orchestration & query processing
│
├── data/
│   ├── raw/                  # Place club documents here (PDF/TXT)
│   └── processed/            # Processed chunks (auto-generated)
│
├── app.py                    # Streamlit web application
├── ingest_data.py            # Data ingestion script
├── config.py                 # Configuration & settings
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
# Required: Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=msu-clubs-index

# Required: Groq API Key (FREE - Recommended)
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Optional: Anthropic Claude (PAID)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-sonnet-4-5-20250929
```

#### Getting API Keys (FREE):

**Pinecone** (Vector Database):
1. Sign up at https://www.pinecone.io/
2. Create a free tier account (generous limits)
3. Create a new index with:
   - **Dimension**: Use Pinecone's hosted model
   - **Model**: `llama-text-embed-v2`
   - **Metric**: cosine
4. Copy your API key from the dashboard

**Groq** (LLM - FREE & Fast):
1. Sign up at https://console.groq.com/
2. Create an API key (free tier with generous limits)
3. Supports Llama 3.3 70B - excellent quality

### 3. Create Pinecone Index

You need to create a Pinecone index with the `llama-text-embed-v2` embedding model:

1. Go to Pinecone dashboard
2. Click "Create Index"
3. Name: `msu-clubs-index`
4. Choose "Inference" mode and select `llama-text-embed-v2`
5. Create the index

### 4. Add Club Documents

Place your club documents (PDF or TXT files) in the `data/raw/` directory:

```bash
# Example: Copy the sample document
# (Already included: data/raw/accessibility_club.txt)

# Add more documents:
cp /path/to/your/club_documents/*.pdf data/raw/
```

### 5. Ingest Data

Process documents and upload to Pinecone:

```bash
python ingest_data.py
```

Options:
- `--clear`: Clear existing data before ingesting
- `--input-dir <path>`: Specify custom input directory
- `--batch-size <n>`: Batch size for uploads (default: 100)

Example:
```bash
python ingest_data.py --clear --input-dir data/raw
```

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Web Interface

1. Open the Streamlit app
2. Enter your question in the text box
3. Optionally apply filters in the sidebar:
   - Maximum dues
   - Specific club name
   - Number of sources to retrieve
4. Click "Search"
5. View the answer with citations

### Example Queries

- "What is the Accessibility Club?"
- "How much are the dues for the Accessibility Club?"
- "What clubs have dues under $15?"
- "Which clubs are good for beginners?"
- "What clubs meet on weekends?"

### Python API Usage

You can also use the RAG engine programmatically:

```python
from src.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine()

# Query
response = rag.query("What is the Accessibility Club?")

print(response["answer"])
print(response["citations"])
```

## Code Flow Documentation

### Data Ingestion Flow

```
1. ingest_data.py (Entry Point)
   ↓
2. DocumentProcessor.process_directory()
   ├─ For each document:
   │  ├─ Extract text (PDF/TXT)
   │  ├─ Clean text (remove artifacts, normalize)
   │  ├─ Extract metadata (club name, dues, etc.)
   │  └─ Chunk text (~300 tokens, 50 overlap)
   ↓
3. VectorStore.upsert_chunks()
   ├─ Generate unique IDs for chunks
   ├─ Flatten metadata (Pinecone compatibility)
   └─ Upload to Pinecone (auto-embedding with llama-text-embed-v2)
   ↓
4. Data Ready for Retrieval!
```

### Query Processing Flow

```
1. User submits query via Streamlit UI
   ↓
2. RAGEngine.query()
   ├─ Extract filters from query (e.g., "under $20" → {"dues": 20})
   ├─ VectorStore.search()
   │  ├─ Embed query using llama-text-embed-v2
   │  ├─ Semantic similarity search
   │  ├─ Apply metadata filters
   │  └─ Return top-k chunks with scores
   ├─ Build context from retrieved chunks
   ├─ Build prompt with system instructions + context + query
   ├─ LLMClient.generate()
   │  └─ Call Groq/Anthropic API
   └─ Return answer + citations
   ↓
3. Display results in Streamlit UI
   ├─ Answer with formatted text
   ├─ Source citations with relevance scores
   └─ Metadata (dues, meeting frequency, etc.)
```

### Module Responsibilities

**1. data_processing.py** (DocumentProcessor)
- Extracts text from PDFs/TXT files using PyMuPDF
- Cleans text (removes page numbers, normalizes whitespace)
- Extracts metadata using regex patterns (club name, dues, dates)
- Chunks text using token-based recursive splitting
- Returns structured chunks with metadata

**2. vector_store.py** (VectorStore)
- Manages Pinecone connection
- Upserts chunks with automatic embedding (llama-text-embed-v2)
- Performs semantic search with optional metadata filters
- Handles batch uploads and error handling

**3. llm_client.py** (BaseLLMClient, GroqClient, AnthropicClient)
- Provides unified interface for multiple LLM providers
- GroqClient: Uses Groq API with Llama 3.3 (FREE)
- AnthropicClient: Uses Anthropic Claude API (PAID)
- Handles prompt formatting and API calls

**4. rag_engine.py** (RAGEngine)
- Orchestrates the entire RAG pipeline
- Extracts filters from natural language queries
- Retrieves relevant chunks from vector store
- Builds context and citations
- Generates answers using LLM
- Returns structured responses with sources

**5. app.py** (Streamlit UI)
- User-friendly web interface
- Filter controls (dues, club name, top-k)
- Query input and example questions
- Result display with expandable citations
- System information and debug mode

**6. config.py** (Configuration)
- Centralized configuration management
- Environment variable loading
- Validation and defaults
- Directory setup

## Technical Details

### Chunking Strategy

- **Chunk Size**: ~300 tokens
- **Overlap**: 50 tokens
- **Method**: Recursive character splitting with token counting (tiktoken)
- **Separators**: Paragraph breaks → sentences → spaces

This ensures:
- Chunks fit in embedding model context
- Semantic coherence maintained
- Overlap prevents information loss at boundaries

### Metadata Extraction

Automatic extraction using regex patterns:
- **Club Name**: From constitution title or filename
- **Dues**: Dollar amounts with $ pattern
- **Meeting Frequency**: Weekly, bi-weekly, monthly patterns
- **Last Updated**: Date patterns
- **Membership Requirements**: Extracted from membership sections

### Retrieval Strategy

1. **Semantic Search**: Uses llama-text-embed-v2 embeddings for similarity
2. **Metadata Filters**: Optional filters for dues, club name
3. **Top-K Selection**: Retrieves top 5 (configurable) most relevant chunks
4. **Relevance Scores**: Cosine similarity scores returned for transparency

### LLM Prompting

**System Prompt**: Instructs LLM to:
- Only use provided context
- Cite sources with [Source X] markers
- Be honest about limitations
- Provide practical information

**User Prompt**: Contains:
- Retrieved context with source markers
- User's question
- Instructions for citation

## Evaluation & Performance

### Retrieval Quality Metrics

To evaluate retrieval precision, you can use the test suite:

```python
from src.rag_engine import RAGEngine

rag = RAGEngine()

# Test queries with expected results
test_cases = [
    ("What is the Accessibility Club?", "Accessibility Club"),
    ("How much are dues?", "$10"),
]

for query, expected in test_cases:
    response = rag.query(query)
    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print(f"Got: {response['answer']}")
    print(f"Relevance: {response['citations'][0]['relevance_score']:.3f}")
```

### Latency

Typical query latency:
- Retrieval: ~200-500ms (Pinecone)
- LLM Generation: ~1-3s (Groq), ~2-5s (Claude)
- Total: ~2-5s end-to-end

## Ethical Considerations

### Privacy & Data

- **Public Data Only**: Only process publicly available club information
- **Opt-In**: Club information should be submitted voluntarily by officers
- **No PII**: Avoid storing personal contact information
- **Transparent Sources**: All answers cite sources for verification

### Terms of Service

- Respect MSU's website terms of service
- No automated scraping of protected pages
- Only use officially sanctioned club documents

### Limitations

- Answers limited to available documents
- May not reflect real-time changes
- Metadata extraction not 100% accurate
- LLM may occasionally misinterpret context

## Troubleshooting

### "Index not found" error

Make sure you created the Pinecone index with the correct name and embedding model.

### "API key not set" error

Check your `.env` file has the required API keys set.

### No results returned

- Verify data was ingested: `python ingest_data.py`
- Check Pinecone dashboard for vector count
- Try simpler queries first

### Import errors

```bash
pip install -r requirements.txt
```

## Future Enhancements

- [ ] Add support for more document formats (DOCX, HTML)
- [ ] Implement LLM-based metadata extraction (more accurate)
- [ ] Add user feedback mechanism for answer quality
- [ ] Support for meeting time/location queries
- [ ] Club recommendation based on user profile
- [ ] Multi-modal support (club logos, photos)
- [ ] LoRA fine-tuning for MSU-specific terminology

## License

This project is for educational purposes as part of an MSU course project.

## Contact

For questions or issues, please contact the project maintainer.

---

**Built with:**
- Pinecone (Vector Database)
- Groq + Llama 3.3 (LLM - FREE)
- Streamlit (Web UI)
- PyMuPDF (PDF Processing)
- LangChain (Text Splitting)
