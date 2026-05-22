"""
Configuration for MSU Club Discovery RAG Assistant.
All runtime values come from environment variables (loaded from .env).
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Force UTF-8 stdout/stderr so Unicode in print() works on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

# ============================================================================
# PINECONE
# ============================================================================
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "msu-clubs-index")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "msu-clubs")

# Hosted embedding model (integrated inference — Pinecone embeds for you)
EMBEDDING_MODEL = "llama-text-embed-v2"

# ============================================================================
# LLM
# ============================================================================
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq")   # "groq" | "anthropic"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS  = 1000

# ============================================================================
# CHUNKING
# ============================================================================
CHUNK_SIZE               = int(os.getenv("CHUNK_SIZE", "300"))          # tokens — profile chunks
CHUNK_OVERLAP            = int(os.getenv("CHUNK_OVERLAP", "50"))        # unused by new processor; kept for compat
CONSTITUTION_MAX_TOKENS  = int(os.getenv("CONSTITUTION_MAX_TOKENS", "350"))  # per .knowledge spec

SUPPORTED_FORMATS = [".pdf", ".txt", ".docx"]

# ============================================================================
# RETRIEVAL
# ============================================================================
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# ============================================================================
# LANGSMITH (optional observability — all tracing disabled when key absent)
# ============================================================================
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT  = os.getenv("LANGCHAIN_PROJECT", "msu-club-discovery")
LANGSMITH_ENABLED  = bool(LANGSMITH_API_KEY)

# Activate the @traceable decorator only when the API key is present.
# Setting LANGCHAIN_TRACING_V2 here means callers don't need to export it.
if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

# ============================================================================
# API (headless mode — Streamlit calls this URL)
# ============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Path to the msu_scraper orgs directory (contains one subdir per club)
SCRAPER_ORGS_DIR = Path(os.getenv("SCRAPER_ORGS_DIR", "D:/msu_scraper/data/orgs"))

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

    print("Configuration validated:")
    print(f"  LLM          : {LLM_PROVIDER} / {LLM_MODEL}")
    print(f"  Pinecone     : {PINECONE_INDEX_NAME} / ns={PINECONE_NAMESPACE}")
    print(f"  Chunk sizes  : profile={CHUNK_SIZE}t  constitution={CONSTITUTION_MAX_TOKENS}t")
    print(f"  Scraper dir  : {SCRAPER_ORGS_DIR}")


if __name__ == "__main__":
    validate_config()
