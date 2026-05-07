"""
MSU Club Discovery RAG Assistant - Streamlit Web App
Interactive interface for students to discover MSU clubs
"""

import sys

# Workaround: Create a dummy readline module for Windows compatibility
# readline is a Unix-only module but some packages try to import it unconditionally
if 'readline' not in sys.modules:
    import types
    readline_module = types.ModuleType('readline')
    sys.modules['readline'] = readline_module

import streamlit as st
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.rag_engine import RAGEngine
from src.vector_store import VectorStore
import config


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MSU Club Discovery Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
@st.cache_resource
def initialize_rag_engine():
    """
    Initialize RAG engine (cached to avoid re-initialization)
    """
    try:
        # Validate configuration
        config.validate_config()

        # Initialize RAG engine
        rag = RAGEngine()
        return rag, None

    except Exception as e:
        return None, str(e)


# ============================================================================
# SIDEBAR - FILTERS AND SETTINGS
# ============================================================================
def render_sidebar():
    """
    Render sidebar with filters and settings
    """
    st.sidebar.title("🔍 Search Filters")

    st.sidebar.markdown("""
    Use filters to narrow down your search to clubs that match your preferences.
    """)

    # Chunk type filter
    st.sidebar.subheader("📂 Content Type")
    chunk_type = st.sidebar.selectbox(
        "Show results from",
        options=["All", "profile", "event", "constitution"],
        index=0,
        help="Profile = general info, Event = club events, Constitution = bylaws"
    )

    # Club name filter (exact match on org_name)
    st.sidebar.subheader("📋 Specific Club")
    enable_club_filter = st.sidebar.checkbox("Search specific club only", value=False)
    org_name = None
    if enable_club_filter:
        org_name = st.sidebar.text_input("Club name (exact)")

    # Advanced settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Settings")

    top_k = st.sidebar.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=config.TOP_K_RESULTS,
        help="More sources = more comprehensive but slower"
    )

    # System info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")
    st.sidebar.markdown(f"""
    - **LLM Provider**: {config.LLM_PROVIDER.title()}
    - **Model**: {config.LLM_MODEL}
    - **Embedding**: {config.EMBEDDING_MODEL}
    - **Index**: {config.PINECONE_INDEX_NAME}
    """)

    return {
        "chunk_type": chunk_type if chunk_type != "All" else None,
        "org_name": org_name if org_name else None,
        "top_k": top_k
    }


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """
    Main application logic
    """

    # Header
    st.title("🎓 MSU Club Discovery Assistant")
    st.markdown("""
    Welcome to the MSU Club Discovery Assistant! Ask questions about Michigan State University
    student clubs and organizations. Get answers with verifiable sources and citations.
    """)

    # Initialize RAG engine
    rag_engine, error = initialize_rag_engine()

    if error:
        st.error(f"❌ Failed to initialize RAG engine: {error}")
        st.markdown("""
        ### Setup Instructions:
        1. Copy `.env.example` to `.env`
        2. Add your API keys:
           - `PINECONE_API_KEY` (required)
           - `GROQ_API_KEY` (recommended - free tier available)
        3. Restart the app
        """)
        return

    # Render sidebar and get filters
    filters = render_sidebar()

    # Example questions
    st.markdown("### 💡 Example Questions")
    example_cols = st.columns(3)

    example_questions = [
        "What clubs are good for beginners?",
        "Which clubs have low or no dues?",
        "Tell me about the Accessibility Club"
    ]

    for col, question in zip(example_cols, example_questions):
        if col.button(question, use_container_width=True):
            st.session_state.current_question = question

    st.markdown("---")

    # Query input
    st.markdown("### ❓ Ask Your Question")

    # Use session state for query persistence
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    query = st.text_input(
        "Enter your question about MSU clubs:",
        value=st.session_state.current_question,
        placeholder="E.g., What clubs meet on weekends? What is the purpose of the Accessibility Club?",
        label_visibility="collapsed"
    )

    # Search button
    col1, col2, col3 = st.columns([1, 1, 4])
    search_clicked = col1.button("🔍 Search", type="primary", use_container_width=True)
    clear_clicked = col2.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.current_question = ""
        st.rerun()

    # Process query
    if search_clicked and query:
        with st.spinner("🔍 Searching knowledge base and generating answer..."):

            # Execute RAG query
            if filters["org_name"] or filters["chunk_type"]:
                response = rag_engine.query_with_metadata_filter(
                    question=query,
                    org_name=filters["org_name"],
                    chunk_type=filters["chunk_type"],
                    top_k=filters["top_k"]
                )
            else:
                response = rag_engine.query(
                    question=query,
                    top_k=filters["top_k"],
                    apply_filters=True,
                    return_citations=True
                )

        # Display results
        st.markdown("---")
        st.markdown("### 💬 Answer")

        # Answer box
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 20px;
            border-radius: 10px; border-left: 5px solid #1f77b4;'>
            {response['answer']}
            </div>""",
            unsafe_allow_html=True
        )

        # Citations
        if response.get("citations"):
            st.markdown("---")
            st.markdown("### 📚 Sources & Citations")

            for citation in response["citations"]:
                with st.expander(
                    f"[{citation['source_number']}] {citation['org_name']} "
                    f"· {citation['chunk_type']} "
                    f"(Relevance: {citation['relevance_score']:.2%})"
                ):
                    st.markdown(f"> {citation['text_snippet']}")

                    metadata = citation.get('metadata', {})
                    if metadata.get('contact_email'):
                        st.markdown(f"**Email**: {metadata['contact_email']}")
                    if metadata.get('contact_website'):
                        st.markdown(f"**Website**: {metadata['contact_website']}")
                    if metadata.get('categories'):
                        cats = metadata['categories']
                        if isinstance(cats, list):
                            cats = ", ".join(cats)
                        st.markdown(f"**Categories**: {cats}")
                    if metadata.get('org_url'):
                        st.markdown(f"**CampusLabs**: {metadata['org_url']}")

        # Applied filters
        if response.get("filters_applied"):
            st.markdown("---")
            st.info(f"📋 Applied filters: {response['filters_applied']}")

        # Debug info (collapsible)
        with st.expander("🔧 Debug Info"):
            st.json({
                "retrieved_chunks": len(response.get('retrieved_chunks', [])),
                "filters_applied": response.get('filters_applied', {}),
                "top_k": filters["top_k"]
            })

    elif search_clicked and not query:
        st.warning("⚠️ Please enter a question!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    MSU Club Discovery Assistant | Powered by RAG (Retrieval-Augmented Generation)<br>
    Built with Pinecone, Groq/Llama 3.3, and Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
