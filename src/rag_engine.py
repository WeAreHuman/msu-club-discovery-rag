"""
RAG Engine Module
Orchestrates retrieval-augmented generation pipeline:
1. Retrieve relevant chunks from vector store
2. Generate response with LLM using retrieved context
3. Provide citations and sources
"""

from typing import List, Dict, Optional, Tuple
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.llm_client import get_llm_client, BaseLLMClient

# Try to import config_streamlit first (for Streamlit deployment), fall back to config
try:
    import config_streamlit as config
except ImportError:
    import config


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for MSU club discovery
    Combines semantic search with LLM generation to answer questions with citations
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        llm_client: BaseLLMClient = None
    ):
        """
        Initialize RAG engine

        Args:
            vector_store: Vector store instance (defaults to new instance)
            llm_client: LLM client instance (defaults to configured provider)
        """
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or get_llm_client()

        print("✓ RAG Engine initialized")

    def _build_context_from_chunks(self, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Build context string from retrieved chunks for LLM prompt
        Also prepare citation information

        Args:
            chunks: List of retrieved chunks with metadata

        Returns:
            Tuple of (context_string, citations_list)
        """
        if not chunks:
            return "", []

        context_parts = []
        citations = []

        for idx, chunk in enumerate(chunks):
            # Extract information
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            club_name = metadata.get("club_name", "Unknown Club")
            source_file = metadata.get("source_file", "Unknown Source")
            score = chunk.get("score", 0)

            # Build context entry with citation marker
            context_parts.append(
                f"[Source {idx + 1}] {club_name}:\n{text}\n"
            )

            # Build citation entry
            citations.append({
                "source_number": idx + 1,
                "club_name": club_name,
                "source_file": source_file,
                "relevance_score": score,
                "text_snippet": text[:150] + "..." if len(text) > 150 else text,
                "metadata": metadata
            })

        context = "\n".join(context_parts)
        return context, citations

    def _extract_filters_from_query(self, query: str) -> Dict:
        """
        Extract metadata filters from natural language query
        Examples:
        - "clubs under $20" -> {"dues": 20}
        - "weekend clubs" -> (would need more sophisticated NLP)

        Args:
            query: User query string

        Returns:
            Dictionary of filters
        """
        filters = {}

        # Extract dues/cost constraints
        dues_patterns = [
            r'(?:under|less than|below|max)\s*\$?(\d+)',
            r'\$(\d+)\s*(?:or less|max|maximum)',
        ]

        for pattern in dues_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters["dues"] = float(match.group(1))
                break

        return filters

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for LLM

        Returns:
            System prompt string
        """
        return """You are a helpful assistant for Michigan State University (MSU) students looking for student clubs and organizations.

Your role is to:
1. Answer questions about MSU student clubs based ONLY on the provided context
2. Be specific and cite your sources using [Source X] markers
3. If the context doesn't contain relevant information, say so honestly
4. Provide practical information like meeting times, dues, and membership requirements when available
5. Be concise but informative
6. If asked about fit/recommendations, consider the student's stated preferences and constraints

Remember: Only use information from the provided context. Do not make up information."""

    def _build_user_prompt(self, query: str, context: str) -> str:
        """
        Build user prompt with query and context

        Args:
            query: User's question
            context: Retrieved context from vector store

        Returns:
            Formatted user prompt
        """
        return f"""Context from MSU club documents:
{context}

Question: {query}

Please answer the question based on the context above. Cite your sources using [Source X] format when referencing specific information. If you cannot answer based on the context, say so."""

    def query(
        self,
        question: str,
        top_k: int = None,
        apply_filters: bool = True,
        return_citations: bool = True
    ) -> Dict:
        """
        Main RAG query method: retrieve relevant chunks and generate answer

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (defaults to config)
            apply_filters: Whether to extract and apply filters from query
            return_citations: Whether to include citation details in response

        Returns:
            Dictionary containing:
            - answer: Generated answer
            - citations: List of source citations (if return_citations=True)
            - retrieved_chunks: Raw retrieved chunks
            - filters_applied: Filters extracted from query
        """
        print(f"\n🔍 Processing query: '{question}'")

        # Step 1: Extract filters from query if enabled
        filters = {}
        if apply_filters:
            filters = self._extract_filters_from_query(question)
            if filters:
                print(f"   📋 Extracted filters: {filters}")

        # Step 2: Retrieve relevant chunks from vector store
        top_k = top_k or config.TOP_K_RESULTS
        chunks = self.vector_store.search(
            query=question,
            top_k=top_k,
            filters=filters if filters else None
        )

        print(f"   ✓ Retrieved {len(chunks)} relevant chunks")

        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the club database to answer your question. Please try rephrasing or ask about a specific club.",
                "citations": [],
                "retrieved_chunks": [],
                "filters_applied": filters
            }

        # Step 3: Build context and citations from chunks
        context, citations = self._build_context_from_chunks(chunks)

        # Step 4: Generate answer using LLM
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, context)

        print(f"   🤖 Generating answer with {config.LLM_PROVIDER}...")

        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )

        print(f"   ✓ Answer generated")

        # Step 5: Build response
        response = {
            "answer": answer,
            "retrieved_chunks": chunks,
            "filters_applied": filters
        }

        if return_citations:
            response["citations"] = citations

        return response

    def query_with_metadata_filter(
        self,
        question: str,
        club_name: str = None,
        max_dues: float = None,
        top_k: int = None
    ) -> Dict:
        """
        Query with explicit metadata filters

        Args:
            question: User's question
            club_name: Filter by specific club name
            max_dues: Maximum dues amount
            top_k: Number of results

        Returns:
            Response dictionary
        """
        filters = {}
        if club_name:
            filters["club_name"] = club_name
        if max_dues is not None:
            filters["dues"] = max_dues

        print(f"\n🔍 Query with filters: {filters}")

        # Retrieve chunks with filters
        top_k = top_k or config.TOP_K_RESULTS
        chunks = self.vector_store.search(
            query=question,
            top_k=top_k,
            filters=filters
        )

        if not chunks:
            return {
                "answer": f"No clubs found matching your criteria: {filters}",
                "citations": [],
                "retrieved_chunks": [],
                "filters_applied": filters
            }

        # Build context and generate answer
        context, citations = self._build_context_from_chunks(chunks)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, context)

        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": chunks,
            "filters_applied": filters
        }


# ============================================================================
# TESTING / STANDALONE EXECUTION
# ============================================================================
if __name__ == "__main__":
    """Test RAG engine with sample queries"""

    print("\n" + "="*80)
    print("RAG ENGINE TEST")
    print("="*80)

    # Initialize RAG engine
    rag = RAGEngine()

    # Test queries
    test_queries = [
        "What is the Accessibility Club?",
        "How much are the dues for Accessibility Club?",
        "When does the Accessibility Club meet?",
        "What clubs have dues under $15?",
    ]

    for query in test_queries:
        print("\n" + "-"*80)
        response = rag.query(query)

        print(f"\n❓ Question: {query}")
        print(f"\n💬 Answer:\n{response['answer']}")

        if response.get('citations'):
            print(f"\n📚 Citations:")
            for citation in response['citations']:
                print(f"   [{citation['source_number']}] {citation['club_name']} "
                      f"(relevance: {citation['relevance_score']:.3f})")

        if response.get('filters_applied'):
            print(f"\n📋 Filters: {response['filters_applied']}")
