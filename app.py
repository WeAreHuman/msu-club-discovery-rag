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
    padding: 28px 36px 20px 36px;
    border-radius: 16px;
    border: 1px solid #2a6645;
    margin-bottom: 20px;
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
    font-size: 2.0rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #81c784;
    font-size: 1.0rem;
    margin: 0;
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

/* ── Vibe badge ─────────────────────────────────────────────── */
.vibe-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 10px;
    border: 1px solid;
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


VIBE_META = {
    "scholar":  {"icon": "🎓", "label": "Scholar Mode",      "color": "#42a5f5"},
    "buddy":    {"icon": "🤝", "label": "Spartan Buddy",     "color": "#66bb6a"},
    "nofilter": {"icon": "🔥", "label": "No Filter Spartan", "color": "#ffa726"},
}


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    if st.sidebar.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Response Style")
    vibe_options = {
        "Scholar Mode": "scholar",
        "Spartan Buddy": "buddy",
        "No Filter": "nofilter",
    }
    vibe_label = st.sidebar.radio(
        "Pick a vibe",
        list(vibe_options.keys()),
        index=0,
        label_visibility="collapsed",
    )
    vibe = vibe_options[vibe_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    chunk_type = st.sidebar.selectbox(
        "Content type",
        options=["All", "profile", "event", "constitution"],
        index=0,
        help="Profile = general info | Event = club events | Constitution = bylaws",
    )

    enable_club_filter = st.sidebar.checkbox("Filter by specific club", value=False)
    org_name = None
    if enable_club_filter:
        org_name = st.sidebar.text_input("Club name (exact match)")

    st.sidebar.markdown("---")
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
        "vibe": vibe,
    }


# ============================================================================
# RENDER HELPERS
# ============================================================================
def render_citations(citations):
    st.markdown("""
    <div class="results-header" style="margin-top:24px;">
        <div class="results-header-line"></div>
        <div class="results-header-text">Sources</div>
        <div class="results-header-line" style="background:linear-gradient(90deg,transparent,#2e7d52);"></div>
    </div>
    """, unsafe_allow_html=True)

    for cite in citations:
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


def render_assistant_message(msg):
    vibe_key = msg.get("vibe", "scholar")
    vm = VIBE_META[vibe_key]
    st.markdown(
        f"<div class='vibe-badge' style='color:{vm['color']};border-color:{vm['color']};background:rgba(0,0,0,0.25);'>"
        f"{vm['icon']} {vm['label']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='answer-box'>{msg['content']}</div>",
        unsafe_allow_html=True,
    )
    if msg.get("citations"):
        render_citations(msg["citations"])
    if msg.get("filters_applied"):
        st.info(f"Filters applied: {msg['filters_applied']}")
    with st.expander("Debug info"):
        st.json({
            "chunks_retrieved": msg.get("num_chunks", 0),
            "filters_applied": msg.get("filters_applied", {}),
        })


# ============================================================================
# MAIN
# ============================================================================
def main():
    rag_engine, error = initialize_rag_engine()

    if error:
        st.error(f"Failed to initialize: {error}")
        return

    filters = render_sidebar()

    # ── Session state ────────────────────────────────────────────────────────
    # messages: full display state — {role, content, citations, vibe, ...}
    # chat_history: plain {role, content} pairs passed to the LLM each turn
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">1,400+ MSU Student Organizations</div>
        <h1>MSU Club Discovery</h1>
        <p>Ask anything — I remember what we've talked about.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Examples (shown only when chat is empty) ──────────────────────────────
    if not st.session_state.messages:
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
                st.session_state.pending_prompt = q
                st.rerun()

    # ── Render conversation history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant_message(msg)

    # ── Get new input (chat box or example chip) ──────────────────────────────
    prompt = st.session_state.pop("pending_prompt", None)
    chat_input = st.chat_input("Ask about MSU clubs...")
    if chat_input:
        prompt = chat_input

    # ── Process prompt ────────────────────────────────────────────────────────
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                response = rag_engine.chat(
                    question=prompt,
                    conversation_history=st.session_state.chat_history,
                    top_k=filters["top_k"],
                    apply_filters=True,
                    return_citations=True,
                    vibe=filters["vibe"],
                    org_name=filters["org_name"],
                    chunk_type=filters["chunk_type"],
                )

            assistant_msg = {
                "role": "assistant",
                "content": response["answer"],
                "citations": response.get("citations", []),
                "filters_applied": response.get("filters_applied", {}),
                "vibe": filters["vibe"],
                "num_chunks": len(response.get("retrieved_chunks", [])),
            }
            render_assistant_message(assistant_msg)

        # Persist to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(assistant_msg)

        # LLM history uses plain question/answer — no context blob in user turn
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>MSU Club Discovery &nbsp;·&nbsp; Pinecone + Groq/Llama 3.3 + Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
