"""
Configuration module for Streamlit Deployment
Loads configuration from Streamlit secrets instead of .env file
"""

import streamlit as st
from pathlib import Path

# ============================================================================
# PINECONE CONFIGURATION (from Streamlit Secrets)
# ============================================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", None)
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "msu-clubs-index")
PINECONE_NAMESPACE = st.secrets.get("PINECONE_NAMESPACE", "clubs")

# Embedding model integrated with Pinecone
EMBEDDING_MODEL = "llama-text-embed-v2"  # Hosted on Pinecone

# ============================================================================
# LLM CONFIGURATION (from Streamlit Secrets)
# ============================================================================
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "groq")  # "groq" or "anthropic"

# API Keys from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)

# Model selection based on provider
LLM_MODEL = st.secrets.get("LLM_MODEL", "llama-3.3-70b-versatile")

# LLM Generation Parameters
LLM_TEMPERATURE = 0.3  # Lower for more factual responses
LLM_MAX_TOKENS = 1000

# ============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# ============================================================================
CHUNK_SIZE = 300  # Tokens per chunk (not needed for deployment, but kept for compatibility)
CHUNK_OVERLAP = 50  # Token overlap

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RESULTS = 5  # Number of chunks to retrieve

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """
    Validates that required configuration variables are set
    Raises error if critical config is missing
    """
    if not PINECONE_API_KEY:
        st.error("❌ PINECONE_API_KEY not found in Streamlit secrets!")
        st.stop()

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found in Streamlit secrets!")
        st.stop()

    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        st.error("❌ ANTHROPIC_API_KEY not found in Streamlit secrets!")
        st.stop()

    st.success(f"✓ Configuration validated successfully")
    st.caption(f"LLM Provider: {LLM_PROVIDER} | Model: {LLM_MODEL} | Pinecone Index: {PINECONE_INDEX_NAME}")
