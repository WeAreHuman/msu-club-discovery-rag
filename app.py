"""
MSU Club Discovery RAG Assistant - Streamlit Web App
Interactive interface for students to discover MSU clubs
"""

import sys

if 'readline' not in sys.modules:
    import types
    readline_module = types.ModuleType('readline')
    sys.modules['readline'] = readline_module

import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.rag_engine import RAGEngine
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
# GLOBAL CSS — explicit colors so the UI looks right in both light and dark mode
# ============================================================================
st.markdown("""
<style>
    /* Answer box */
    .answer-box {
        background-color: #1e2a1e;
        color: #e8f5e9;
        padding: 22px 26px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    /* Citation snippet blockquote */
    .citation-snippet {
        background-color: #1a1f2e;
        color: #cdd5e0;
        border-left: 3px solid #4c8af5;
        padding: 10px 16px;
        border-radius: 6px;
        font-style: italic;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }

    /* Metadata pill badges */
    .meta-badge {
        display: inline-block;
        background-color: #263238;
        color: #b0bec5;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.82rem;
        margin: 3px 3px 3px 0;
    }

    /* Section headers */
    .section-label {
        color: #90caf9;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 2px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #546e7a;
        font-size: 0.85rem;
        padding-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RAG ENGINE (cached)
# ============================================================================
@st.cache_resource
def initialize_rag_engine():
    try:
        config.validate_config()
        return RAGEngine(), None
    except Exception as e:
        return None, str(e)


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    st.sidebar.title("Search Filters")

    st.sidebar.subheader("Content Type")
    chunk_type = st.sidebar.selectbox(
        "Show results from",
        options=["All", "profile", "event", "constitution"],
        index=0,
        help="Profile = general info | Event = club events | Constitution = bylaws"
    )

    st.sidebar.subheader("Specific Club")
    enable_club_filter = st.sidebar.checkbox("Filter by club name", value=False)
    org_name = None
    if enable_club_filter:
        org_name = st.sidebar.text_input("Club name (exact match)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    top_k = st.sidebar.slider(
        "Sources to retrieve",
        min_value=1, max_value=10, value=config.TOP_K_RESULTS,
        help="More sources = broader but slower"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**LLM**: {config.LLM_PROVIDER.title()} / {config.LLM_MODEL}\n\n"
        f"**Embed**: {config.EMBEDDING_MODEL}\n\n"
        f"**Index**: {config.PINECONE_INDEX_NAME}"
    )

    return {
        "chunk_type": chunk_type if chunk_type != "All" else None,
        "org_name": org_name if org_name else None,
        "top_k": top_k,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("MSU Club Discovery Assistant")
    st.markdown(
        "Ask anything about Michigan State University student clubs and organizations. "
        "Answers are grounded in real club data with citations."
    )

    rag_engine, error = initialize_rag_engine()

    if error:
        st.error(f"Failed to initialize RAG engine: {error}")
        st.markdown("""
        **Setup checklist:**
        1. Copy `.env.example` to `.env`
        2. Set `PINECONE_API_KEY` and `GROQ_API_KEY`
        3. Restart the app
        """)
        return

    filters = render_sidebar()

    # ── Example questions ───────────────────────────────────────────────────
    st.markdown("#### Try an example")
    examples = [
        "What clubs are good for beginners?",
        "Which clubs have low or no dues?",
        "Tell me about the Accessibility Club",
    ]
    cols = st.columns(len(examples))
    for col, q in zip(cols, examples):
        if col.button(q, use_container_width=True):
            st.session_state.current_question = q

    st.markdown("---")

    # ── Query input ─────────────────────────────────────────────────────────
    st.markdown("#### Ask your question")

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    query = st.text_input(
        "Question",
        value=st.session_state.current_question,
        placeholder="e.g. What clubs meet on weekends? What is the purpose of the Robotics Club?",
        label_visibility="collapsed",
    )

    c1, c2, _ = st.columns([1, 1, 5])
    search_clicked = c1.button("Search", type="primary", use_container_width=True)
    if c2.button("Clear", use_container_width=True):
        st.session_state.current_question = ""
        st.rerun()

    # ── Process query ────────────────────────────────────────────────────────
    if search_clicked and query:
        with st.spinner("Searching knowledge base..."):
            if filters["org_name"] or filters["chunk_type"]:
                response = rag_engine.query_with_metadata_filter(
                    question=query,
                    org_name=filters["org_name"],
                    chunk_type=filters["chunk_type"],
                    top_k=filters["top_k"],
                )
            else:
                response = rag_engine.query(
                    question=query,
                    top_k=filters["top_k"],
                    apply_filters=True,
                    return_citations=True,
                )

        # ── Answer ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Answer")
        st.markdown(
            f"<div class='answer-box'>{response['answer']}</div>",
            unsafe_allow_html=True,
        )

        # ── Citations ────────────────────────────────────────────────────────
        if response.get("citations"):
            st.markdown("---")
            st.markdown("#### Sources")

            for cite in response["citations"]:
                meta = cite.get("metadata", {})
                label = (
                    f"[{cite['source_number']}]  {cite['org_name']}  ·  "
                    f"{cite['chunk_type']}  ·  {cite['relevance_score']:.0%} match"
                )
                with st.expander(label):
                    st.markdown(
                        f"<div class='citation-snippet'>{cite['text_snippet']}</div>",
                        unsafe_allow_html=True,
                    )

                    badges = []
                    if meta.get("contact_email"):
                        badges.append(f"✉ {meta['contact_email']}")
                    if meta.get("contact_website"):
                        badges.append(f"🌐 {meta['contact_website']}")
                    if meta.get("org_url"):
                        badges.append(f"🔗 CampusLabs")
                    if meta.get("categories"):
                        cats = meta["categories"]
                        if isinstance(cats, list):
                            cats = " · ".join(cats)
                        badges.append(f"🏷 {cats}")

                    if badges:
                        html = " ".join(
                            f"<span class='meta-badge'>{b}</span>" for b in badges
                        )
                        st.markdown(html, unsafe_allow_html=True)

        # ── Applied filters notice ───────────────────────────────────────────
        if response.get("filters_applied"):
            st.info(f"Filters applied: {response['filters_applied']}")

        # ── Debug ────────────────────────────────────────────────────────────
        with st.expander("Debug info"):
            st.json({
                "chunks_retrieved": len(response.get("retrieved_chunks", [])),
                "filters_applied": response.get("filters_applied", {}),
                "top_k": filters["top_k"],
            })

    elif search_clicked and not query:
        st.warning("Please enter a question first.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='footer'>MSU Club Discovery · Pinecone + Groq/Llama 3.3 + Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
