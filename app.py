"""
MSU Club Discovery RAG Assistant - Streamlit Web App
Interactive interface for students to discover MSU clubs
"""

import streamlit as st
from pathlib import Path
import sys

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

    # Dues filter
    st.sidebar.subheader("💰 Dues/Cost")
    enable_dues_filter = st.sidebar.checkbox("Filter by maximum dues", value=False)
    max_dues = None
    if enable_dues_filter:
        max_dues = st.sidebar.slider(
            "Maximum dues ($)",
            min_value=0,
            max_value=100,
            value=25,
            step=5
        )

    # Club name filter (exact match)
    st.sidebar.subheader("📋 Specific Club")
    enable_club_filter = st.sidebar.checkbox("Search specific club only", value=False)
    club_name = None
    if enable_club_filter:
        club_name = st.sidebar.text_input("Club name")

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
        "max_dues": max_dues,
        "club_name": club_name if club_name else None,
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

            # Build filter dictionary for explicit filters
            explicit_filters = {}
            if filters["club_name"]:
                explicit_filters["club_name"] = filters["club_name"]
            if filters["max_dues"] is not None:
                explicit_filters["max_dues"] = filters["max_dues"]

            # Execute RAG query
            if explicit_filters:
                # Use explicit filter method
                response = rag_engine.query_with_metadata_filter(
                    question=query,
                    club_name=explicit_filters.get("club_name"),
                    max_dues=explicit_filters.get("max_dues"),
                    top_k=filters["top_k"]
                )
            else:
                # Use standard query (with auto-filter extraction)
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
                    f"[{citation['source_number']}] {citation['club_name']} "
                    f"(Relevance: {citation['relevance_score']:.2%})"
                ):
                    st.markdown(f"**Source File**: {citation['source_file']}")
                    st.markdown(f"**Text Snippet**:")
                    st.markdown(f"> {citation['text_snippet']}")

                    # Additional metadata
                    metadata = citation.get('metadata', {})
                    if metadata.get('dues'):
                        st.markdown(f"**Dues**: ${metadata['dues']}")
                    if metadata.get('meeting_frequency'):
                        st.markdown(f"**Meeting Frequency**: {metadata['meeting_frequency']}")
                    if metadata.get('last_updated'):
                        st.markdown(f"**Last Updated**: {metadata['last_updated']}")

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
