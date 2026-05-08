"""
MSU Club Discovery RAG Assistant - Streamlit Web App
"""

import sys
if 'readline' not in sys.modules:
    import types
    sys.modules['readline'] = types.ModuleType('readline')

import streamlit as st
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_engine import RAGEngine
import config

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MSU Club Discovery",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
/* ── Hero card ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #18453B 0%, #0f2d20 55%, #071510 100%);
    padding: 40px 36px 18px 36px;
    border-radius: 16px 16px 0 0;
    border: 1px solid #2a6645;
    border-bottom: none;
    margin-bottom: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(76,175,80,0.18);
    color: #a5d6a7;
    border: 1px solid rgba(76,175,80,0.35);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero h1 {
    color: #ffffff;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #81c784;
    font-size: 1.05rem;
    margin: 0 0 22px 0;
}

/* ── Search bar ─────────────────────────────────────────────── */
/* Visually attach input to hero card — negative margins close Streamlit's flex gap */
div[data-testid="stTextInput"] {
    background: #0d2118 !important;
    padding: 12px 36px 12px 36px !important;
    border-left: 1px solid #2a6645 !important;
    border-right: 1px solid #2a6645 !important;
    margin-top: -1rem !important;
    margin-bottom: -1rem !important;
}
div[data-testid="stTextInput"] input {
    font-size: 1.08rem !important;
    padding: 16px 18px !important;
    border-radius: 10px !important;
    border: 1.5px solid #2e7d52 !important;
    background-color: #091a10 !important;
    color: #e8f5e9 !important;
    width: 100% !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76,175,80,0.2) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #3a6649 !important;
}
div[data-testid="stTextInput"] label { display: none; }

/* ── Button row bar ─────────────────────────────────────────── */
.btn-bar {
    background: #0d2118;
    padding: 12px 36px 28px 36px;
    border-radius: 0 0 16px 16px;
    border: 1px solid #2a6645;
    border-top: none;
    margin-top: 0;
    margin-bottom: 0;
}

/* ── Primary Search button ──────────────────────────────────── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d52 0%, #1b5935 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    padding: 10px 0 !important;
    border-radius: 8px !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #388e5c 0%, #2e7d52 100%) !important;
    box-shadow: 0 4px 18px rgba(76,175,80,0.3) !important;
}

/* ── Examples section ───────────────────────────────────────── */
.examples-wrap {
    background: linear-gradient(180deg, #0b1e14 0%, #081409 100%);
    border: 1px solid #1a3d28;
    border-radius: 12px;
    padding: 20px 24px 12px 24px;
    margin-top: 20px;
}
.examples-label {
    color: #66bb6a;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
}

/* Style example buttons as subtle chips */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button[kind="secondary"] {
    background-color: rgba(255,255,255,0.04) !important;
    border: 1px solid #2a5c3a !important;
    color: #a5d6a7 !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    padding: 9px 12px !important;
    transition: all 0.15s ease;
    text-align: left !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: rgba(76,175,80,0.1) !important;
    border-color: #4caf50 !important;
    color: #e8f5e9 !important;
}

/* ── Answer box ─────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg, #0f2a1c 0%, #091a10 100%);
    color: #e8f5e9;
    padding: 26px 30px;
    border-radius: 12px;
    border: 1px solid #2a6645;
    border-left: 5px solid #4caf50;
    font-size: 1.04rem;
    line-height: 1.75;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* ── Citation snippet ───────────────────────────────────────── */
.citation-snippet {
    background: #0d1b26;
    color: #b0bec5;
    border-left: 3px solid #42a5f5;
    padding: 12px 18px;
    border-radius: 8px;
    font-style: italic;
    font-size: 0.94rem;
    margin-bottom: 10px;
    line-height: 1.6;
}

/* ── Meta badges ────────────────────────────────────────────── */
.meta-badge {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    color: #90a4ae;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px 4px 3px 0;
}

/* ── Results divider accent ─────────────────────────────────── */
.results-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px 0;
}
.results-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2e7d52, transparent);
}
.results-header-text {
    color: #66bb6a;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    white-space: nowrap;
}

/* ── Footer ─────────────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #37474f;
    font-size: 0.82rem;
    padding: 16px 0 4px;
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
        help="Profile = general info | Event = club events | Constitution = bylaws",
    )

    st.sidebar.subheader("Specific Club")
    enable_club_filter = st.sidebar.checkbox("Filter by club name", value=False)
    org_name = None
    if enable_club_filter:
        org_name = st.sidebar.text_input("Club name (exact match)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    top_k = st.sidebar.slider(
        "Sources to retrieve", min_value=1, max_value=10,
        value=config.TOP_K_RESULTS,
        help="More sources = broader but slower",
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
    rag_engine, error = initialize_rag_engine()

    if error:
        st.error(f"Failed to initialize: {error}")
        return

    filters = render_sidebar()

    # ── Hero + search ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">1,400+ MSU Student Organizations</div>
        <h1>MSU Club Discovery</h1>
        <p>Ask anything. Get instant answers grounded in real club data.</p>
    </div>
    """, unsafe_allow_html=True)

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "has_result" not in st.session_state:
        st.session_state.has_result = False

    query = st.text_input(
        "search",
        value=st.session_state.current_question,
        placeholder="e.g.  What clubs are good for networking?  Which clubs have no dues?",
        label_visibility="collapsed",
    )

    st.markdown("<div class='btn-bar'>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 6])
    search_clicked = c1.button("Search", type="primary", use_container_width=True)
    if c2.button("Clear", use_container_width=True):
        st.session_state.current_question = ""
        st.session_state.has_result = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Examples (hidden once a result is showing) ────────────────────────────
    if not st.session_state.has_result:
        st.markdown("""
        <div class="examples-wrap">
            <div class="examples-label">Quick examples — click to search</div>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            "What clubs are good for beginners?",
            "Which clubs have low or no dues?",
            "Tell me about the Accessibility Club",
        ]
        cols = st.columns(3)
        for col, q in zip(cols, examples):
            if col.button(q, use_container_width=True):
                st.session_state.current_question = q
                st.rerun()

    # ── Query execution ───────────────────────────────────────────────────────
    if search_clicked and query:
        st.session_state.has_result = True
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
        st.markdown("""
        <div class="results-header">
            <div class="results-header-line"></div>
            <div class="results-header-text">Answer</div>
            <div class="results-header-line" style="background:linear-gradient(90deg,transparent,#2e7d52);"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<div class='answer-box'>{response['answer']}</div>",
            unsafe_allow_html=True,
        )

        # ── Citations ────────────────────────────────────────────────────────
        if response.get("citations"):
            st.markdown("""
            <div class="results-header" style="margin-top:24px;">
                <div class="results-header-line"></div>
                <div class="results-header-text">Sources</div>
                <div class="results-header-line" style="background:linear-gradient(90deg,transparent,#2e7d52);"></div>
            </div>
            """, unsafe_allow_html=True)

            for cite in response["citations"]:
                meta = cite.get("metadata", {})
                label = (
                    f"[{cite['source_number']}]  {cite['org_name']}"
                    f"  ·  {cite['chunk_type']}"
                    f"  ·  {cite['relevance_score']:.0%} match"
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
                        badges.append("🔗 CampusLabs page")
                    if meta.get("categories"):
                        cats = meta["categories"]
                        if isinstance(cats, list):
                            cats = " · ".join(cats)
                        badges.append(f"🏷 {cats}")
                    if badges:
                        st.markdown(
                            " ".join(f"<span class='meta-badge'>{b}</span>" for b in badges),
                            unsafe_allow_html=True,
                        )

        if response.get("filters_applied"):
            st.info(f"Filters applied: {response['filters_applied']}")

        with st.expander("Debug info"):
            st.json({
                "chunks_retrieved": len(response.get("retrieved_chunks", [])),
                "filters_applied": response.get("filters_applied", {}),
                "top_k": filters["top_k"],
            })

    elif search_clicked and not query:
        st.warning("Please enter a question first.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>MSU Club Discovery &nbsp;·&nbsp; Pinecone + Groq/Llama 3.3 + Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
