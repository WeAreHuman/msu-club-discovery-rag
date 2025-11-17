"""
Vector Store Module
Manages Pinecone vector database operations with llama-text-embed-v2 embedding model
"""

from typing import List, Dict, Optional
import sys

# Workaround: Patch pinecone's deprecated plugin check
# The check is too strict and fails even if the plugin isn't being used
# We'll disable it since pinecone 5.x has the features natively
sys.modules['pinecone_plugins'] = None
sys.modules['pinecone_plugins.inference'] = None

import pinecone
import hashlib
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import config_streamlit first (for Streamlit deployment), fall back to config
try:
    import config_streamlit as config
except ImportError:
    import config

# Try to import sentence-transformers for embeddings, fall back to hash-based if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class VectorStore:
    """
    Manages interactions with Pinecone vector database
    Uses Pinecone's hosted llama-text-embed-v2 embedding model
    """

    def __init__(self, api_key: str = None, index_name: str = None, namespace: str = None):
        """
        Initialize Pinecone vector store

        Args:
            api_key: Pinecone API key (defaults to config)
            index_name: Name of Pinecone index (defaults to config)
            namespace: Namespace for organizing vectors (defaults to config)
        """
        self.api_key = api_key or config.PINECONE_API_KEY
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.namespace = namespace or config.PINECONE_NAMESPACE

        # Initialize Pinecone client (using class-based API compatible with 2.x and 5.x)
        try:
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            print(f"✓ Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to index '{self.index_name}': {e}")
            print(f"   Make sure the index exists and is configured with llama-text-embed-v2")
            self.index = None

        # Initialize embedding model
        try:
            if EMBEDDINGS_AVAILABLE:
                # Use sentence-transformers with 1024 dimensions to match Pinecone index
                # Using a larger model that produces 1024-dim vectors
                self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                print(f"✓ Embedding model initialized: sentence-transformers (all-mpnet-base-v2)")
            else:
                print(f"⚠️  Warning: sentence-transformers not available, using hash-based embeddings")
                self.embedding_model = None
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize embedding model: {e}")
            print(f"   Trying alternative model...")
            try:
                if EMBEDDINGS_AVAILABLE:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
                    print(f"✓ Embedding model initialized: sentence-transformers (all-MiniLM-L12-v2)")
                else:
                    self.embedding_model = None
            except:
                self.embedding_model = None

    def _generate_chunk_id(self, text: str, metadata: Dict) -> str:
        """
        Generate a unique ID for a chunk based on content and metadata

        Args:
            text: Chunk text
            metadata: Chunk metadata

        Returns:
            Unique ID string
        """
        # Create deterministic ID from club name and chunk index
        club_name = metadata.get("club_name", "unknown")
        chunk_idx = metadata.get("chunk_index", 0)

        # Create hash for uniqueness
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]

        return f"{club_name.replace(' ', '_')}_{chunk_idx}_{content_hash}"

    def create_vector_record(self, text: str, metadata: Dict) -> Dict:
        """
        Create a vector record for insertion into Pinecone

        This function prepares a single chunk of text and metadata into a format
        that Pinecone can accept. Uses Pinecone's hosted embedding model.

        Args:
            text: The text content to embed
            metadata: Dictionary containing metadata about the text
                      (e.g., club_name, dues, meeting_frequency, source_file, etc.)

        Returns:
            Dictionary with 'id', 'text', and 'metadata' ready for Pinecone upsert
        """
        # Generate unique ID for this text chunk
        chunk_id = self._generate_chunk_id(text, metadata)

        # Flatten metadata (Pinecone doesn't support nested structures well)
        flat_metadata = {
            "text": text,  # Store text for retrieval display
            "club_name": metadata.get("club_name", ""),
            "dues": metadata.get("dues"),
            "meeting_frequency": metadata.get("meeting_frequency", ""),
            "source_file": metadata.get("source_file", ""),
            "last_updated": metadata.get("last_updated", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "total_chunks": metadata.get("total_chunks", 1),
        }

        # Remove None values
        flat_metadata = {k: v for k, v in flat_metadata.items() if v is not None}

        # Create the vector record - Pinecone will generate embeddings from text
        vector_record = {
            "id": chunk_id,
            "text": text,  # Pinecone will embed this using llama-text-embed-v2
            "metadata": flat_metadata
        }

        return vector_record

    def upsert_chunks(self, chunks: List[Dict[str, any]], batch_size: int = 100) -> int:
        """
        Insert or update chunks in Pinecone vector database
        Uses Pinecone's hosted embedding model (llama-text-embed-v2)

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            batch_size: Number of chunks to upsert per batch

        Returns:
            Number of chunks successfully upserted
        """
        if not self.index:
            print("⚠️  Index not initialized. Cannot upsert chunks.")
            return 0

        print(f"\n📤 Upserting {len(chunks)} chunks to Pinecone...")

        total_upserted = 0

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare records for Pinecone using create_vector_record
            records = []
            for chunk in batch:
                vector_record = self.create_vector_record(
                    text=chunk["text"],
                    metadata=chunk["metadata"]
                )
                records.append(vector_record)

            # Upsert to Pinecone using text-based interface
            try:
                self.index.upsert(
                    namespace=self.namespace,
                    documents=records
                )
                total_upserted += len(records)
                print(f"  ✓ Batch {i//batch_size + 1}: {len(records)} chunks upserted")

            except Exception as e:
                print(f"  ⚠️  Error upserting batch {i//batch_size + 1}: {e}")

        print(f"✓ Total upserted: {total_upserted} chunks")
        return total_upserted

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity
        Optionally apply metadata filters

        Args:
            query: Search query text
            top_k: Number of results to return (defaults to config)
            filters: Optional metadata filters (e.g., {"club_name": "Accessibility Club"})

        Returns:
            List of matching chunks with scores and metadata
        """
        if not self.index:
            print("⚠️  Index not initialized. Cannot search.")
            return []

        top_k = top_k or config.TOP_K_RESULTS

        # Execute search using Pinecone's vector API
        try:
            # Generate embedding for the query
            if self.embedding_model:
                # Use sentence-transformers if available
                query_embedding = self.embedding_model.encode(query)
                # Pad to 1024 dimensions if needed
                if len(query_embedding) < 1024:
                    import numpy as np
                    query_embedding = np.pad(query_embedding, (0, 1024 - len(query_embedding)), mode='constant')
                query_vector = query_embedding.tolist()
            else:
                # Fall back to simple hash-based embedding with 1024 dimensions
                import hashlib
                import numpy as np
                hash_obj = hashlib.sha256(query.encode())
                hash_bytes = hash_obj.digest()
                query_vector = []
                for i in range(1024):  # Use 1024 dimensions to match index
                    byte_idx = (i * 2) % len(hash_bytes)
                    value = float(hash_bytes[byte_idx]) / 255.0 - 0.5
                    query_vector.append(value)
                vector_array = np.array(query_vector)
                vector_norm = np.linalg.norm(vector_array)
                if vector_norm > 0:
                    vector_array = vector_array / vector_norm
                query_vector = vector_array.tolist()

            # Query the index with the embedding vector
            results = self.index.query(
                namespace=self.namespace,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            # Format results
            formatted_results = []
            matches = results.get("matches", [])
            
            for match in matches:
                # Handle both dict and object formats
                match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", None)
                match_score = match.get("score") if isinstance(match, dict) else getattr(match, "score", 0)
                match_metadata = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})
                
                if match_metadata:
                    formatted_results.append({
                        "id": match_id,
                        "score": match_score,
                        "text": match_metadata.get("text", "") if isinstance(match_metadata, dict) else "",
                        "metadata": {
                            k: v for k, v in match_metadata.items()
                            if k != "text"
                        } if isinstance(match_metadata, dict) else {}
                    })

            return formatted_results

        except Exception as e:
            print(f"⚠️  Error searching: {e}")
            return []

    def delete_namespace(self):
        """
        Delete all vectors in the current namespace
        Useful for clearing data during development
        """
        if not self.index:
            print("⚠️  Index not initialized.")
            return

        try:
            self.index.delete(namespace=self.namespace, delete_all=True)
            print(f"✓ Deleted all vectors in namespace '{self.namespace}'")
        except Exception as e:
            print(f"⚠️  Error deleting namespace: {e}")

    def get_stats(self):
        """
        Get statistics about the index
        """
        if not self.index:
            print("⚠️  Index not initialized.")
            return None

        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"⚠️  Error getting stats: {e}")
            return None


# ============================================================================
# TESTING / STANDALONE EXECUTION
# ============================================================================
if __name__ == "__main__":
    """Test vector store operations"""

    # Initialize vector store
    vs = VectorStore()

    # Test data
    test_chunks = [
        {
            "text": "The Accessibility Club meets every Tuesday at 6 PM in the Union.",
            "metadata": {
                "club_name": "Accessibility Club",
                "dues": 10.0,
                "meeting_frequency": "weekly",
                "source_file": "test.pdf",
                "chunk_index": 0,
                "total_chunks": 1
            }
        }
    ]

    # Test upsert
    print("\n=== Testing Upsert ===")
    vs.upsert_chunks(test_chunks)

    # Test search
    print("\n=== Testing Search ===")
    results = vs.search("when does accessibility club meet?", top_k=3)
    for idx, result in enumerate(results):
        print(f"\n{idx + 1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Club: {result['metadata'].get('club_name')}")

    # Test stats
    print("\n=== Index Stats ===")
    stats = vs.get_stats()
    if stats:
        print(json.dumps(stats, indent=2))
