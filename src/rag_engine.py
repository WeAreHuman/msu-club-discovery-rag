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

            org_name = metadata.get("org_name", "Unknown Club")
            chunk_type = metadata.get("chunk_type", "")
            score = chunk.get("score", 0)

            context_parts.append(
                f"[Source {idx + 1}] {org_name}:\n{text}\n"
            )

            citations.append({
                "source_number": idx + 1,
                "org_name": org_name,
                "chunk_type": chunk_type,
                "relevance_score": score,
                "text_snippet": text[:150] + "..." if len(text) > 150 else text,
                "metadata": metadata
            })

        context = "\n".join(context_parts)
        return context, citations

    def _extract_filters_from_query(self, query: str) -> Dict:
        """
        Extract Pinecone metadata filters from natural language query.
        Currently supports chunk_type steering only; extend as schema evolves.
        """
        filters = {}

        # Steer toward event chunks when the question is clearly event-focused
        event_keywords = r'\b(event|events|when|upcoming|schedule|meeting|meets)\b'
        if re.search(event_keywords, query, re.IGNORECASE):
            filters["chunk_type"] = {"$eq": "event"}

        return filters

    def _build_system_prompt(self, vibe: str = "scholar") -> str:
        prompts = {
    "scholar": """You are a helpful assistant for Michigan State University (MSU) students looking for student clubs and organizations.

Your role is to:
1. Answer questions about MSU student clubs based ONLY on the provided context
2. Be specific and cite your sources using [Source X] markers
3. If the context doesn't contain relevant information, say so honestly
4. Provide practical information like meeting times, dues, and membership requirements when available
5. Be concise but informative
6. If asked about fit/recommendations, consider the student's stated preferences and constraints

After your answer, always add one short empathetic follow-up — a genuine clarifying question that would help you give a more personalized answer next time. Examples: asking what area or domain they're interested in, what "beginner-friendly" means to them, whether they're looking for social vs academic clubs, their schedule, or budget. Keep it natural and conversational, not a formal heading or bullet. One question only.

Remember: Only use information from the provided context. Do not make up information.""",

    "buddy": """You're a chill MSU senior texting a first-year about clubs. Here's how you actually talk:

BAD: "I would recommend considering Her Campus, which is a publication focused on women's issues."
GOOD: "Her Campus is honestly a solid pick — they do events, workshops, the whole thing. Not just a writing club vibe [Source 1]."

BAD: "If you are interested in community service, you may want to explore the following options."
GOOD: "If you're into giving back, there's actually a few decent options in here."

Rules:
- Lead with the answer, not a disclaimer
- Weave in [Source X] citations naturally mid-sentence
- Short paragraphs. No bullet walls.
- If the context doesn't cover it: "honestly couldn't find much on that one, might be worth DMing them directly"
- Only use info from the provided context. Never make things up.

After your answer, drop one casual follow-up question — something that'd help you give a better rec next time. Like asking what field they're into, what vibe they want (chill hangout vs career-focused), or what they mean by something vague. One question, woven in naturally at the end, like you're texting a friend.""",

    "nofilter": """You're an MSU student giving your friend the real, unfiltered club breakdown. You actually read between the lines of what people are asking.

Here's the difference between you and a boring assistant:

QUESTION: "I'm new to MSU, looking to find a girl for myself, what clubs should I join?"
BORING: "I must advise that joining a club solely to find a romantic partner may not be the best approach..."
YOU: "okay okay, I see the vision 👀 clubs are honestly one of the best places to meet people at MSU so smart move. Her Campus is surprisingly open to everyone and their events are actually fun [Source 1] — way better than joining some club where everyone just stares at a whiteboard together."

QUESTION: "which clubs have no dues?"
BORING: "Several clubs do not charge membership fees."
YOU: "keeping it budget-friendly, respect. here's what the data shows..."

Your rules:
- Read the actual intent behind the question and acknowledge it (meeting people, saving money, career moves — whatever it is)
- Keep it short and punchy. No long intros, no "it's important to note that"
- Cite with [Source X] dropped naturally, not in a footnote-y way
- Never objectifying, never creepy — you're being real, not weird
- If the context doesn't cover it: "bro the data doesn't say, hit them up directly"
- Only use info from the provided context. Never make things up.

End every response with one real follow-up question — the kind a friend would actually ask to narrow it down. Something like "what's your major tho?" or "are you more into casual vibes or building your resume?" One question, no more, totally natural.""",
}
        return prompts.get(vibe, prompts["scholar"])

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
        return_citations: bool = True,
        vibe: str = "scholar"
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
        system_prompt = self._build_system_prompt(vibe)
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

    def chat(
        self,
        question: str,
        conversation_history: List[Dict] = None,
        top_k: int = None,
        apply_filters: bool = True,
        return_citations: bool = True,
        vibe: str = "scholar",
        org_name: str = None,
        chunk_type: str = None,
    ) -> Dict:
        """
        Multi-turn RAG query. Retrieves fresh context for the current question
        and passes prior conversation history to the LLM so it can reference
        earlier turns.

        Args:
            question: Current user question
            conversation_history: Plain [{role, content}] pairs from prior turns
                                  (user content = raw question, not context-injected)
            org_name: Optional explicit club filter
            chunk_type: Optional explicit chunk type filter
        """
        print(f"\n🔍 Chat query: '{question}'")

        # Build filters — explicit overrides take priority over auto-extraction
        filters = {}
        if org_name:
            filters["org_name"] = {"$eq": org_name}
        if chunk_type:
            filters["chunk_type"] = {"$eq": chunk_type}
        if not filters and apply_filters:
            filters = self._extract_filters_from_query(question)
            if filters:
                print(f"   📋 Extracted filters: {filters}")

        top_k = top_k or config.TOP_K_RESULTS
        chunks = self.vector_store.search(
            query=question,
            top_k=top_k,
            filters=filters if filters else None,
        )

        print(f"   ✓ Retrieved {len(chunks)} relevant chunks")

        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the club database. Try rephrasing or ask about a specific club.",
                "citations": [],
                "retrieved_chunks": [],
                "filters_applied": filters,
            }

        context, citations = self._build_context_from_chunks(chunks)

        # Build message list: history + current user message with injected context
        current_user_content = self._build_user_prompt(question, context)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": current_user_content})

        system_prompt = self._build_system_prompt(vibe)
        print(f"   🤖 Generating answer with {config.LLM_PROVIDER}...")

        answer = self.llm_client.generate_with_history(
            messages=messages,
            system_prompt=system_prompt,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        print(f"   ✓ Answer generated")

        response = {
            "answer": answer,
            "retrieved_chunks": chunks,
            "filters_applied": filters,
        }
        if return_citations:
            response["citations"] = citations
        return response

    def query_with_metadata_filter(
        self,
        question: str,
        org_name: str = None,
        chunk_type: str = None,
        top_k: int = None,
        vibe: str = "scholar"
    ) -> Dict:
        """
        Query with explicit metadata filters.

        Args:
            question  : user's question
            org_name  : restrict to a specific club (exact match on org_name)
            chunk_type: restrict to "profile", "event", or "constitution"
            top_k     : number of results
        """
        filters = {}
        if org_name:
            filters["org_name"] = {"$eq": org_name}
        if chunk_type:
            filters["chunk_type"] = {"$eq": chunk_type}

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
        system_prompt = self._build_system_prompt(vibe)
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
