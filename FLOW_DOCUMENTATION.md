# MSU Club Discovery RAG - Code Flow Documentation

This document provides an intuitive, step-by-step explanation of how the code flows through the system.

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Ingestion Flow](#data-ingestion-flow)
3. [Query Processing Flow](#query-processing-flow)
4. [Module Deep Dive](#module-deep-dive)
5. [Example Walkthrough](#example-walkthrough)

---

## System Overview

The RAG system has two main workflows:

1. **Ingestion Workflow** (One-time setup): Documents → Chunks → Vector Database
2. **Query Workflow** (Real-time): User Question → Retrieval → LLM → Answer + Citations

```
INGESTION WORKFLOW:
Documents → Extract → Clean → Chunk → Embed → Store in Pinecone

QUERY WORKFLOW:
User Query → Search Pinecone → Retrieve Chunks → Build Prompt → LLM → Answer
```

---

## Data Ingestion Flow

### Step-by-Step Process

#### Step 1: User Runs Ingestion Script
```bash
python ingest_data.py --clear
```

**What happens**: Entry point in `ingest_data.py:main()`

**Code path**:
```python
ingest_data.py:main()
  ├─ Parse command line arguments
  ├─ Validate configuration (API keys, directories)
  └─ Call ingest_documents()
```

---

#### Step 2: Initialize Components
**Location**: `ingest_data.py:ingest_documents()`

**What happens**:
```python
# Create document processor
processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)

# Create vector store connection
vector_store = VectorStore(api_key, index_name, namespace)
```

**Under the hood**:
- `DocumentProcessor` initializes tiktoken tokenizer for counting tokens
- `VectorStore` connects to Pinecone index using API key
- Validates that index exists and has correct embedding model

---

#### Step 3: Process Each Document
**Location**: `src/data_processing.py:process_directory()`

**Code flow**:
```python
for each file in data/raw/:
    chunks = process_document(file_path)
    all_chunks.extend(chunks)
```

**For each document**, the following sub-steps occur:

##### Step 3a: Extract Text
**Location**: `src/data_processing.py:extract_text_from_pdf()` or `extract_text_from_txt()`

```python
if file.endswith('.pdf'):
    # Use PyMuPDF (fitz) to extract text
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
elif file.endswith('.txt'):
    # Simply read the file
    text = open(file_path).read()
```

**Output**: Raw text string (may contain artifacts, extra spaces, etc.)

---

##### Step 3b: Clean Text
**Location**: `src/data_processing.py:clean_text()`

```python
# Remove excessive whitespace
text = re.sub(r'\s+', ' ', text)

# Remove page numbers and artifacts
text = re.sub(r'\b\d+\s*Updated\s+\d+\s+\w+\s+\d{4}\b', '', text)

# Normalize and strip
text = text.strip()
```

**Output**: Cleaned text ready for processing

---

##### Step 3c: Extract Metadata
**Location**: `src/data_processing.py:extract_metadata_from_text()`

Uses regex patterns to find:

```python
# Club name
club_name_match = re.search(
    r'name of this organization shall be\s+(?:the\s+)?([^.]+Club[^.]*)',
    text,
    re.IGNORECASE
)

# Dues
dues_match = re.search(r'(?:dues|fee)[^\d]*\$?(\d+(?:\.\d{2})?)', text)

# Meeting frequency
meeting_match = re.search(r'meet(?:ing)?s?\s+(?:every\s+)?(\w+)', text)

# Last updated date
date_match = re.search(r'Updated\s+(\d+\s+\w+\s+\d{4})', text)
```

**Output**: Metadata dictionary
```python
{
    "club_name": "Accessibility Club at Michigan State University",
    "dues": 10.0,
    "meeting_frequency": "every 2 months",
    "last_updated": "10 October 2021",
    "source_file": "accessibility_club.txt"
}
```

---

##### Step 3d: Chunk Text
**Location**: `src/data_processing.py:chunk_text()`

Uses LangChain's `RecursiveCharacterTextSplitter`:

```python
# Initialize splitter with token counting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # tokens
    chunk_overlap=50,        # tokens
    length_function=count_tokens,  # Uses tiktoken
    separators=["\n\n", "\n", ". ", " ", ""]  # Try in order
)

chunks = splitter.split_text(cleaned_text)
```

**How it works**:
1. Tries to split on paragraph breaks (`\n\n`) first
2. If chunks too large, tries sentences (`. `)
3. If still too large, splits on spaces
4. Ensures each chunk ≤ 300 tokens
5. Overlaps chunks by 50 tokens to preserve context

**Output**: List of text chunks
```python
[
    "Updated 10 October 2021 Accessibility Club...",
    "Section 1: The name of this organization...",
    "Subsection 2: Paying a yearly fee of $10...",
    ...
]
```

---

##### Step 3e: Attach Metadata to Chunks
**Location**: `src/data_processing.py:chunk_text()`

```python
for idx, chunk_text in enumerate(chunks):
    chunk_objects.append({
        "text": chunk_text,
        "metadata": {
            **metadata,           # All document metadata
            "chunk_index": idx,   # Position in document
            "total_chunks": len(chunks)
        }
    })
```

**Output**: List of chunk objects
```python
[
    {
        "text": "Updated 10 October 2021...",
        "metadata": {
            "club_name": "Accessibility Club",
            "dues": 10.0,
            "chunk_index": 0,
            "total_chunks": 15,
            ...
        }
    },
    ...
]
```

---

#### Step 4: Upload to Pinecone
**Location**: `src/vector_store.py:upsert_chunks()`

```python
# Process in batches of 100
for batch in chunks[::100]:
    records = []

    for chunk in batch:
        # Generate unique ID
        chunk_id = f"{club_name}_{chunk_idx}_{content_hash}"

        # Prepare record for Pinecone
        records.append({
            "id": chunk_id,
            "text": chunk["text"],  # Pinecone auto-embeds this
            "metadata": flatten(chunk["metadata"])
        })

    # Upload batch
    index.upsert_records(namespace="clubs", records=records)
```

**What happens in Pinecone**:
1. Pinecone receives the text
2. Automatically embeds it using `llama-text-embed-v2` (hosted model)
3. Stores vector + metadata in index
4. Vector is ready for similarity search

**Final state**: All chunks embedded and stored in Pinecone, searchable by semantic similarity

---

## Query Processing Flow

### Step-by-Step Process

#### Step 1: User Submits Query
**Location**: `app.py` (Streamlit UI)

```python
query = st.text_input("Enter your question")

if st.button("Search"):
    response = rag_engine.query(query)
```

**Example query**: "What is the Accessibility Club?"

---

#### Step 2: RAG Engine Processes Query
**Location**: `src/rag_engine.py:query()`

```python
def query(question, top_k=5, apply_filters=True):
    # Step 2a: Extract filters from query
    filters = extract_filters_from_query(question)
    # E.g., "clubs under $20" → {"dues": 20}

    # Step 2b: Search vector store
    chunks = vector_store.search(question, top_k, filters)

    # Step 2c: Build context from chunks
    context, citations = build_context_from_chunks(chunks)

    # Step 2d: Generate answer with LLM
    answer = llm_client.generate(prompt, system_prompt)

    # Step 2e: Return structured response
    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": chunks
    }
```

---

#### Step 2a: Extract Filters (Optional)
**Location**: `src/rag_engine.py:_extract_filters_from_query()`

```python
# Detect dues constraints
if "under $20" in query or "less than $20" in query:
    filters["dues"] = 20.0

# Could extend to detect meeting days, etc.
if "weekend" in query:
    filters["meeting_days"] = "weekend"
```

**Output**: Filter dictionary (may be empty)
```python
{"dues": 20.0}  # or {}
```

---

#### Step 2b: Search Vector Store
**Location**: `src/vector_store.py:search()`

```python
# Build search query
search_params = {
    "namespace": "clubs",
    "query": {
        "inputs": {"text": question},  # Pinecone auto-embeds
        "top_k": 5
    }
}

# Add metadata filters if present
if filters:
    search_params["query"]["filter"] = {
        "dues": {"$lte": 20.0}  # Less than or equal
    }

# Execute search
results = index.search(**search_params)
```

**What happens in Pinecone**:
1. Pinecone embeds the query using `llama-text-embed-v2`
2. Performs cosine similarity search across all vectors
3. Applies metadata filters
4. Returns top 5 most similar chunks

**Output**: List of matches
```python
[
    {
        "id": "Accessibility_Club_0_a1b2c3d4",
        "score": 0.89,  # Similarity score (0-1)
        "metadata": {
            "text": "The purpose of this organization...",
            "club_name": "Accessibility Club",
            "dues": 10.0,
            ...
        }
    },
    ...
]
```

---

#### Step 2c: Build Context for LLM
**Location**: `src/rag_engine.py:_build_context_from_chunks()`

```python
context_parts = []
citations = []

for idx, chunk in enumerate(chunks):
    # Add to context with source marker
    context_parts.append(
        f"[Source {idx + 1}] {chunk['metadata']['club_name']}:\n"
        f"{chunk['text']}\n"
    )

    # Prepare citation info
    citations.append({
        "source_number": idx + 1,
        "club_name": chunk['metadata']['club_name'],
        "relevance_score": chunk['score'],
        "text_snippet": chunk['text'][:150] + "..."
    })

context = "\n".join(context_parts)
```

**Output**: Context string + citations list

**Context example**:
```
[Source 1] Accessibility Club:
The purpose of this organization shall be to serve as a professional organization students with an interest in the field of accessibility.

[Source 2] Accessibility Club:
Paying a yearly fee of $10 per school year. Under no circumstance should dues be reimbursed.

[Source 3] Accessibility Club:
The organization shall host meetings at least once every 2 months.
```

---

#### Step 2d: Build Prompt and Call LLM
**Location**: `src/rag_engine.py:query()`

```python
# System prompt (instructs LLM behavior)
system_prompt = """You are a helpful assistant for MSU students.
Answer based ONLY on the provided context. Cite sources with [Source X]."""

# User prompt (context + question)
user_prompt = f"""Context from MSU club documents:
{context}

Question: {question}

Please answer with citations."""

# Call LLM
answer = llm_client.generate(user_prompt, system_prompt)
```

**LLM Client** (`src/llm_client.py`):
```python
# If using Groq
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3,
    max_tokens=1000
)

answer = response.choices[0].message.content
```

**LLM Output** (example):
```
The Accessibility Club at Michigan State University serves as a professional
organization for students with an interest in the field of accessibility
[Source 1]. The club charges annual dues of $10 per school year [Source 2]
and hosts meetings at least once every 2 months [Source 3].
```

---

#### Step 2e: Return Response
**Location**: `src/rag_engine.py:query()`

```python
return {
    "answer": answer,
    "citations": [
        {
            "source_number": 1,
            "club_name": "Accessibility Club",
            "relevance_score": 0.89,
            "text_snippet": "The purpose of this organization...",
            "metadata": {...}
        },
        ...
    ],
    "retrieved_chunks": chunks,
    "filters_applied": filters
}
```

---

#### Step 3: Display in UI
**Location**: `app.py`

```python
# Display answer
st.markdown(f"### Answer\n{response['answer']}")

# Display citations
for citation in response['citations']:
    with st.expander(f"[{citation['source_number']}] {citation['club_name']}"):
        st.markdown(f"**Relevance**: {citation['relevance_score']:.2%}")
        st.markdown(f"**Snippet**: {citation['text_snippet']}")
        st.markdown(f"**Dues**: ${citation['metadata']['dues']}")
```

---

## Module Deep Dive

### config.py - Configuration Management

**Purpose**: Central place for all configuration

**Key responsibilities**:
- Load environment variables from `.env`
- Provide defaults for missing values
- Validate required configuration (API keys, etc.)
- Define directory paths

**When it runs**: Imported by all other modules at startup

**Example usage**:
```python
import config

api_key = config.PINECONE_API_KEY  # Loaded from .env
chunk_size = config.CHUNK_SIZE      # Default: 300
```

---

### src/data_processing.py - DocumentProcessor

**Purpose**: Convert raw documents into structured, chunked data

**Key methods**:
1. `extract_text_from_pdf()`: Use PyMuPDF to get text from PDF
2. `clean_text()`: Remove artifacts, normalize whitespace
3. `extract_metadata_from_text()`: Regex-based metadata extraction
4. `chunk_text()`: Token-based recursive splitting
5. `process_document()`: Full pipeline for one document
6. `process_directory()`: Process all documents in a folder

**When it runs**: During data ingestion (`python ingest_data.py`)

**Input**: Raw PDF/TXT files in `data/raw/`

**Output**: List of chunks with metadata
```python
[
    {"text": "...", "metadata": {...}},
    ...
]
```

---

### src/vector_store.py - VectorStore

**Purpose**: Manage Pinecone vector database operations

**Key methods**:
1. `__init__()`: Connect to Pinecone index
2. `upsert_chunks()`: Upload chunks to Pinecone (auto-embedding)
3. `search()`: Semantic search with optional filters
4. `delete_namespace()`: Clear all data (for testing)
5. `get_stats()`: Get index statistics

**When it runs**:
- During ingestion (upsert)
- During queries (search)

**Important**: Uses Pinecone's hosted `llama-text-embed-v2` model, so no local embedding needed

---

### src/llm_client.py - LLM Clients

**Purpose**: Unified interface for multiple LLM providers

**Architecture**:
```
BaseLLMClient (abstract)
  ├─ GroqClient (Llama 3.3 - FREE)
  └─ AnthropicClient (Claude - PAID)
```

**Key method**: `generate(prompt, system_prompt, temperature, max_tokens)`

**Factory function**: `get_llm_client(provider)` - Returns appropriate client based on config

**When it runs**: During query processing to generate answers

**Example**:
```python
llm = get_llm_client(provider="groq")
answer = llm.generate("What is AI?", system_prompt="You are helpful.")
```

---

### src/rag_engine.py - RAGEngine

**Purpose**: Orchestrate the entire RAG pipeline

**Key methods**:
1. `query()`: Main entry point for user queries
2. `_extract_filters_from_query()`: Parse natural language for filters
3. `_build_context_from_chunks()`: Format retrieved chunks for LLM
4. `_build_system_prompt()`: Instructions for LLM behavior
5. `_build_user_prompt()`: Combine context + question
6. `query_with_metadata_filter()`: Explicit filter version

**When it runs**: Every time user submits a query

**Workflow**:
```
Query → Extract Filters → Search Vector Store → Build Context → Call LLM → Return Answer + Citations
```

---

### app.py - Streamlit UI

**Purpose**: Web interface for users

**Key functions**:
1. `initialize_rag_engine()`: Cached initialization (runs once)
2. `render_sidebar()`: Filters and settings
3. `main()`: Main app logic

**When it runs**: When user runs `streamlit run app.py`

**Features**:
- Query input
- Filter controls (dues, club name, top-k)
- Example questions
- Answer display with formatted citations
- Expandable debug info

---

## Example Walkthrough

Let's trace a complete query: **"What are the dues for Accessibility Club?"**

### Phase 1: Initialization (happens once)

```python
# app.py
rag_engine = initialize_rag_engine()
  ├─ config.validate_config()  # Check API keys
  ├─ vector_store = VectorStore()  # Connect to Pinecone
  └─ llm_client = get_llm_client()  # Initialize Groq client
```

---

### Phase 2: User Submits Query

```python
# app.py
query = "What are the dues for Accessibility Club?"
response = rag_engine.query(query)
```

---

### Phase 3: RAG Processing

#### 3.1: Extract Filters
```python
# rag_engine.py:query()
filters = _extract_filters_from_query(query)
# No filters extracted (no "under $X" pattern)
# filters = {}
```

#### 3.2: Search Vector Store
```python
# rag_engine.py:query()
chunks = vector_store.search(
    query="What are the dues for Accessibility Club?",
    top_k=5,
    filters={}
)

# Inside vector_store.py:search()
results = index.search(
    namespace="clubs",
    query={"inputs": {"text": query}, "top_k": 5}
)
# Pinecone embeds query and searches
# Returns top 5 similar chunks
```

**Retrieved chunks** (example):
```python
[
    {
        "id": "Accessibility_Club_3_xyz",
        "score": 0.92,
        "text": "Paying a yearly fee of $10 per school year...",
        "metadata": {"club_name": "Accessibility Club", "dues": 10.0, ...}
    },
    {
        "id": "Accessibility_Club_0_abc",
        "score": 0.85,
        "text": "The name of this organization shall be Accessibility Club...",
        "metadata": {"club_name": "Accessibility Club", "dues": 10.0, ...}
    },
    ...
]
```

#### 3.3: Build Context
```python
# rag_engine.py:query()
context, citations = _build_context_from_chunks(chunks)
```

**Context**:
```
[Source 1] Accessibility Club:
Paying a yearly fee of $10 per school year. Under no circumstance should dues be reimbursed.

[Source 2] Accessibility Club:
The name of this organization shall be the Accessibility Club at Michigan State University.

...
```

#### 3.4: Generate Answer
```python
# rag_engine.py:query()
system_prompt = "You are a helpful assistant. Answer based on context. Cite sources."

user_prompt = f"""Context:
{context}

Question: What are the dues for Accessibility Club?"""

answer = llm_client.generate(user_prompt, system_prompt)
```

**LLM processes**:
- Reads all 5 source chunks
- Identifies relevant information
- Generates answer with citations

**LLM output**:
```
The Accessibility Club at Michigan State University charges annual dues of $10 per school year [Source 1]. These dues are non-refundable under any circumstance [Source 1].
```

#### 3.5: Return Response
```python
return {
    "answer": "The Accessibility Club charges annual dues of $10...",
    "citations": [
        {"source_number": 1, "club_name": "Accessibility Club", "score": 0.92, ...},
        {"source_number": 2, "club_name": "Accessibility Club", "score": 0.85, ...},
        ...
    ],
    "retrieved_chunks": chunks,
    "filters_applied": {}
}
```

---

### Phase 4: Display Results

```python
# app.py
st.markdown(f"### Answer\n{response['answer']}")

for citation in response['citations']:
    st.expander(f"[{citation['source_number']}] {citation['club_name']}")
    # Shows relevance score, text snippet, metadata
```

**User sees**:
```
Answer:
The Accessibility Club at Michigan State University charges annual dues of
$10 per school year [Source 1]. These dues are non-refundable [Source 1].

Sources:
▼ [1] Accessibility Club (Relevance: 92%)
  Snippet: "Paying a yearly fee of $10 per school year..."
  Dues: $10
  Source File: accessibility_club.txt

▼ [2] Accessibility Club (Relevance: 85%)
  Snippet: "The name of this organization shall be..."
```

---

## Key Design Decisions

### 1. Why Chunking?

**Problem**: Documents are too long for embedding models (max 512-8192 tokens)

**Solution**: Split into ~300 token chunks with 50 token overlap

**Benefits**:
- Fit in embedding context window
- More precise retrieval (chunk-level vs document-level)
- Overlap prevents context loss at boundaries

---

### 2. Why Metadata?

**Problem**: Semantic search alone may not respect hard constraints (e.g., "under $20 dues")

**Solution**: Attach metadata to chunks, enable hybrid search (semantic + filters)

**Benefits**:
- Support precise filtering ("dues under $20", "specific club")
- Provide structured info in citations
- Enable faceted search

---

### 3. Why Citations?

**Problem**: LLMs can hallucinate; users need to verify answers

**Solution**: Track source chunks, include [Source X] markers, show sources with relevance scores

**Benefits**:
- Transparency and trust
- Users can verify information
- Debugging retrieval quality

---

### 4. Why Multiple LLM Providers?

**Problem**: Different users have different requirements (free vs paid, speed vs quality)

**Solution**: Abstract LLM interface, support Groq (free) and Claude (paid)

**Benefits**:
- Default to free option (Groq)
- Easy to swap providers
- Future-proof (can add more providers)

---

## Performance Characteristics

### Ingestion

- **Speed**: ~2-5 docs/second (depends on doc size, API latency)
- **Bottleneck**: PDF text extraction, Pinecone upload
- **Optimization**: Batch uploads (100 chunks/batch)

### Query

- **Latency breakdown**:
  - Pinecone search: ~200-500ms
  - LLM generation: ~1-3s (Groq), ~2-5s (Claude)
  - Total: ~2-5s end-to-end
- **Bottleneck**: LLM generation
- **Optimization**: Lower top_k, use faster model (Llama 8B vs 70B)

### Cost

- **Pinecone**: Free tier (100K vectors)
- **Groq**: Free tier (~14K requests/day for 70B model)
- **Total**: FREE for typical usage!

---

## Common Issues & Debugging

### Issue: "No results found"

**Possible causes**:
1. Data not ingested
2. Query too specific or uses different terminology
3. Filters too restrictive

**Debug**:
```python
# Check vector count
vector_store.get_stats()

# Try broader query
response = rag.query("Accessibility", top_k=10)

# Disable filters
response = rag.query(query, apply_filters=False)
```

---

### Issue: "Incorrect answers"

**Possible causes**:
1. Relevant chunks not retrieved (retrieval problem)
2. LLM misinterpreting context (generation problem)

**Debug**:
```python
# Check retrieved chunks
response = rag.query(query)
for chunk in response['retrieved_chunks']:
    print(chunk['score'], chunk['text'][:100])

# Check if relevant info is in top results
# If not → retrieval problem (improve chunking, embedding)
# If yes → generation problem (improve prompt, try different LLM)
```

---

### Issue: "Slow responses"

**Possible causes**:
1. Large top_k (retrieving many chunks)
2. Slow LLM model (70B vs 8B)

**Optimize**:
```python
# Reduce top_k
response = rag.query(query, top_k=3)  # Instead of 5

# Use faster model
# In .env: LLM_MODEL=llama-3.1-8b-instant
```

---

## Summary

The RAG system flows through three main stages:

1. **Ingestion** (one-time):
   - Raw documents → Extraction → Cleaning → Metadata → Chunking → Embedding → Pinecone

2. **Retrieval** (per query):
   - User query → Embedding → Semantic search → Metadata filtering → Top-k chunks

3. **Generation** (per query):
   - Retrieved chunks → Context building → LLM prompting → Answer + citations

Each module has a clear responsibility, and the flow is orchestrated by the `RAGEngine`. The system prioritizes transparency (citations), flexibility (multiple LLM providers), and cost-efficiency (free tier options).
