"""
MSU Club Discovery RAG Assistant - Streamlit Deployment
Interactive interface for students to discover MSU clubs
Streamlit deployment version without data ingestion functionality
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.rag_engine import RAGEngine
from src.vector_store import VectorStore
import config_streamlit as config


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MSU Club Discovery Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .club-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .score-badge {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)


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
# MAIN APP
# ============================================================================
def main():
    # Initialize RAG Engine
    rag_engine, init_error = initialize_rag_engine()

    if init_error:
        st.error(f"❌ Initialization Error: {init_error}")
        st.info("Please check your Streamlit secrets configuration.")
        return

    # Header
    st.title("🎓 MSU Club Discovery Assistant")
    st.markdown("### Find the perfect student club at Michigan State University!")
    st.divider()

    # Main search interface
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_query = st.text_input(
                "What are you looking for in a club?",
                placeholder="e.g., 'Good for beginners', 'AI and Machine Learning', 'Creative activities'",
                key="search_input"
            )
        
        with col2:
            st.write("")  # Spacing
            search_button = st.button("🔍 Search", use_container_width=True)

    st.divider()

    # Process search
    if search_button and user_query:
        with st.spinner("🔍 Searching for clubs..."):
            try:
                # Get RAG response
                response = rag_engine.query(user_query)

                # Display results
                st.subheader("📚 Search Results")
                
                if response.get("relevant_chunks"):
                    st.success(f"✓ Found {len(response['relevant_chunks'])} relevant clubs")
                    
                    # Display relevant chunks
                    with st.expander("📖 Relevant Information", expanded=True):
                        for idx, chunk in enumerate(response["relevant_chunks"], 1):
                            st.markdown(f"**Result {idx}:**")
                            st.info(f"📌 Score: {chunk['score']:.2%}")
                            
                            # Display metadata
                            metadata = chunk.get("metadata", {})
                            if metadata.get("club_name"):
                                st.markdown(f"**Club:** {metadata['club_name']}")
                            if metadata.get("meeting_frequency"):
                                st.markdown(f"**Meeting Frequency:** {metadata['meeting_frequency']}")
                            if metadata.get("dues"):
                                st.markdown(f"**Dues:** ${metadata['dues']:.2f}")
                            
                            # Display the text
                            st.markdown(f"**Details:** {chunk['text'][:300]}...")
                            st.divider()
                else:
                    st.warning("⚠️ No relevant clubs found. Try a different search query.")

                # Display AI-generated response
                st.subheader("🤖 Assistant Response")
                if response.get("answer"):
                    st.markdown(response["answer"])
                else:
                    st.info("Could not generate a response. Please try again.")

            except Exception as e:
                st.error(f"❌ Search Error: {str(e)}")
                st.info("Please try a different search query or refresh the page.")

    elif search_button and not user_query:
        st.warning("⚠️ Please enter a search query")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you discover student clubs at Michigan State University 
        that match your interests and goals.
        
        **How to use:**
        1. Enter what you're looking for in a club
        2. The assistant will search our club database
        3. Review the relevant information about clubs
        4. Get personalized recommendations
        
        **Features:**
        - 🔍 Semantic search across club descriptions
        - 🤖 AI-powered recommendations
        - 📊 Club details (fees, meeting times, etc.)
        """)
        
        st.divider()
        st.subheader("🎯 Quick Suggestions")
        suggestions = [
            "Good for beginners",
            "Creative and artistic clubs",
            "Technology and AI",
            "Professional development",
            "Low membership fees",
            "Meets weekly"
        ]
        
        for suggestion in suggestions:
            if st.button(f"🔗 {suggestion}", use_container_width=True):
                st.session_state.search_input = suggestion
                st.rerun()
        
        st.divider()
        st.caption("🔐 Powered by RAG & LLM | Version 1.0 - Deployment")


if __name__ == "__main__":
    main()
