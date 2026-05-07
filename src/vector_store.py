"""
Vector Store — Pinecone with llama-text-embed-v2 (integrated inference).

Upsert : text is embedded server-side by Pinecone (documents= interface).
Search : query is embedded via pc.inference.embed() then queried with index.query(),
         ensuring the same model is used for storage and retrieval.
"""

import sys
import json
from typing import List, Dict, Optional

# Windows compatibility — some packages import readline unconditionally
if "readline" not in sys.modules:
    import types
    sys.modules["readline"] = types.ModuleType("readline")

import pinecone
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


class VectorStore:
    """
    Manages Pinecone interactions for the MSU Club Discovery RAG system.

    Chunk types stored: "profile", "event", "constitution"
    Embedding model   : llama-text-embed-v2 (1024 dims, hosted by Pinecone)
    """

    def __init__(self, api_key: str = None, index_name: str = None, namespace: str = None):
        self.api_key = api_key or config.PINECONE_API_KEY
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.namespace = namespace or config.PINECONE_NAMESPACE

        self.pc = pinecone.Pinecone(api_key=self.api_key)
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Warning: could not connect to index '{self.index_name}': {e}")
            self.index = None

    # -------------------------------------------------------------------------
    # Upsert
    # -------------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict], batch_size: int = 96) -> int:
        """
        Insert or update chunks in Pinecone. Text is embedded by Pinecone's
        integrated llama-text-embed-v2 model.

        Args:
            chunks    : list of {"text": str, "metadata": dict}
            batch_size: records per upsert call (keep ≤ 96 for the inference API)

        Returns:
            Number of chunks successfully upserted.
        """
        if not self.index:
            print("Index not initialized — cannot upsert.")
            return 0

        total = 0
        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            records = [self._make_record(c) for c in batch]
            try:
                self.index.upsert_records(records=records, namespace=self.namespace)
                total += len(records)
                print(f"  Batch {batch_num}/{len(batches)}: {len(records)} records  (total: {total})")
            except Exception as e:
                print(f"  Batch {batch_num} failed: {e}")
                return total  # stop and report how many made it

        return total

    def _make_record(self, chunk: Dict) -> Dict:
        """
        Build a flat Pinecone record for upsert_records().

        upsert_records expects:
          { "_id": str, <embedded_field>: str, ...flat metadata fields... }
        No nested "metadata" dict — everything is top-level.
        """
        text = chunk["text"]
        meta = chunk.get("metadata", {})

        # _id is the Pinecone record identifier
        record: Dict = {"_id": self._chunk_id(meta, text), "text": text}

        # Flatten metadata — Pinecone scalar: str/int/float/bool; list: list[str]
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                record[k] = v
            elif isinstance(v, list):
                record[k] = [str(x) for x in v] if v else []
            # None and nested dicts are omitted

        return record

    @staticmethod
    def _chunk_id(meta: Dict, text: str) -> str:
        """Deterministic ID so repeated ingestion is idempotent."""
        slug = meta.get("org_slug", "unknown")
        chunk_type = meta.get("chunk_type", "chunk")

        if chunk_type == "event":
            event_id = meta.get("event_id", "")
            return f"{slug}_event_{event_id}"
        if chunk_type == "constitution":
            article_num = meta.get("article_num", "")
            idx = meta.get("chunk_index", 0)
            return f"{slug}_constitution_{article_num}_{idx}"
        # profile (and any future type)
        idx = meta.get("chunk_index", 0)
        return f"{slug}_profile_{idx}"

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search using llama-text-embed-v2 via Pinecone inference API.

        Args:
            query  : natural-language question
            top_k  : number of results (default from config)
            filters: Pinecone metadata filter dict, e.g.
                     {"chunk_type": {"$eq": "event"}}
                     {"categories": {"$in": ["Academic"]}}

        Returns:
            List of {"id", "score", "text", "metadata"} dicts.
        """
        if not self.index:
            print("Index not initialized — cannot search.")
            return []

        top_k = top_k or config.TOP_K_RESULTS

        try:
            # Embed the query with the same model used at index time
            embeddings = self.pc.inference.embed(
                model=config.EMBEDDING_MODEL,
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"},
            )
            query_vector = embeddings[0].values

            results = self.index.query(
                namespace=self.namespace,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filters,
            )
        except Exception as e:
            print(f"Search error: {e}")
            return []

        formatted = []
        for match in results.get("matches") or []:
            match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", "")
            score = match.get("score") if isinstance(match, dict) else getattr(match, "score", 0.0)
            raw_meta = (
                match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})
            )
            if not isinstance(raw_meta, dict):
                raw_meta = {}

            # text was stored inside metadata for display; separate it out
            raw_meta = dict(raw_meta)  # don't mutate the original
            text = raw_meta.pop("text", "")

            formatted.append({
                "id": match_id,
                "score": score,
                "text": text,
                "metadata": raw_meta,
            })

        return formatted

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def delete_namespace(self):
        """Delete all vectors in the current namespace."""
        if not self.index:
            return
        try:
            self.index.delete(namespace=self.namespace, delete_all=True)
            print(f"Deleted all vectors in namespace '{self.namespace}'")
        except Exception as e:
            print(f"Error deleting namespace: {e}")

    def get_stats(self) -> Optional[Dict]:
        """Return index statistics dict."""
        if not self.index:
            return None
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    vs = VectorStore()
    print("\n=== Search test ===")
    results = vs.search("volunteer community service clubs", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['metadata'].get('chunk_type')}] score={r['score']:.4f}")
        print(f"   Club : {r['metadata'].get('org_name')}")
        print(f"   Text : {r['text'][:120]}...")

    print("\n=== Index stats ===")
    stats = vs.get_stats()
    if stats:
        print(json.dumps(dict(stats), indent=2, default=str))
