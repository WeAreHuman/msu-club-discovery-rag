"""
Configuration module for MSU Club Discovery RAG Assistant
Loads environment variables and provides centralized configuration
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "msu-clubs-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "clubs")

# Embedding model integrated with Pinecone
EMBEDDING_MODEL = "llama-text-embed-v2"  # Hosted on Pinecone

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "anthropic"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model selection based on provider
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# LLM Generation Parameters
LLM_TEMPERATURE = 0.3  # Lower for more factual responses
LLM_MAX_TOKENS = 1000

# ============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# ============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))  # Tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Token overlap

# Supported file formats
SUPPORTED_FORMATS = [".pdf", ".txt", ".docx"]

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Number of chunks to retrieve

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """
    Validates that required configuration variables are set
    Raises ValueError if critical config is missing
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment variables")

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER is 'groq'")

    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'")

    print(f"✓ Configuration validated successfully")
    print(f"  - LLM Provider: {LLM_PROVIDER}")
    print(f"  - LLM Model: {LLM_MODEL}")
    print(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"  - Chunk Size: {CHUNK_SIZE} tokens (overlap: {CHUNK_OVERLAP})")

if __name__ == "__main__":
    validate_config()
