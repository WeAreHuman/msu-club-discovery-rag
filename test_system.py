"""
System Test Script
Tests all components of the RAG system to ensure proper setup.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from src.data_processing import ClubDataProcessor
from src.vector_store import VectorStore
from src.llm_client import get_llm_client
from src.rag_engine import RAGEngine


def test_configuration():
    print("\n" + "=" * 70)
    print("TEST 1: CONFIGURATION")
    print("=" * 70)
    try:
        config.validate_config()
        print("PASS: Configuration is valid")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_document_processing():
    print("\n" + "=" * 70)
    print("TEST 2: DOCUMENT PROCESSING (ClubDataProcessor)")
    print("=" * 70)
    try:
        processor = ClubDataProcessor()

        orgs_dir = config.SCRAPER_ORGS_DIR
        if not orgs_dir.exists():
            print(f"SKIP: Scraper orgs directory not found: {orgs_dir}")
            print("      Set SCRAPER_ORGS_DIR in .env and run the scraper first.")
            return False

        # Find first club dir that has a profile.json
        sample_dir = next(
            (d for d in sorted(orgs_dir.iterdir())
             if d.is_dir() and (d / "profile.json").exists()),
            None
        )
        if sample_dir is None:
            print("SKIP: No club directories with profile.json found.")
            return False

        chunks = processor.process_club_dir(sample_dir)
        types = {}
        for c in chunks:
            t = c["metadata"].get("chunk_type", "?")
            types[t] = types.get(t, 0) + 1

        print(f"PASS: Processed '{sample_dir.name}' -> {len(chunks)} chunks: {types}")
        if chunks:
            print(f"      Sample text: {chunks[0]['text'][:120]}...")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_vector_store():
    print("\n" + "=" * 70)
    print("TEST 3: VECTOR STORE (PINECONE)")
    print("=" * 70)
    try:
        vs = VectorStore()
        if vs.index is None:
            print("FAIL: Could not connect to Pinecone index")
            print("  1. Check PINECONE_API_KEY in .env")
            print(f"  2. Verify index '{config.PINECONE_INDEX_NAME}' exists")
            return False

        stats = vs.get_stats()
        if stats:
            total = getattr(stats, "total_vector_count", None) or stats.get("total_vector_count", 0)
            print(f"PASS: Connected to '{config.PINECONE_INDEX_NAME}'  vectors={total}")
            return True

        print("WARN: Connected but could not fetch stats")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_llm_client():
    print("\n" + "=" * 70)
    print("TEST 4: LLM CLIENT")
    print("=" * 70)
    try:
        llm = get_llm_client()
        print(f"     Provider: {config.LLM_PROVIDER} / {config.LLM_MODEL}")

        response = llm.generate(
            prompt="Reply with exactly: Hello from MSU RAG!",
            temperature=0.0,
            max_tokens=30,
        )

        if response:
            print(f"PASS: '{response.strip()}'")
            return True
        print("FAIL: Empty response from LLM")
        return False

    except Exception as e:
        print(f"FAIL: {e}")
        key_hint = "GROQ_API_KEY" if config.LLM_PROVIDER == "groq" else "ANTHROPIC_API_KEY"
        print(f"  Check {key_hint} in .env")
        import traceback; traceback.print_exc()
        return False


def test_rag_engine():
    print("\n" + "=" * 70)
    print("TEST 5: RAG ENGINE (END-TO-END)")
    print("=" * 70)
    try:
        rag = RAGEngine()
        query = "What volunteer clubs are there at MSU?"
        print(f"     Query: '{query}'")

        response = rag.query(query, top_k=3)

        if response.get("answer"):
            chunks = response.get("retrieved_chunks", [])
            citations = response.get("citations", [])
            print(f"PASS: Answer generated ({len(chunks)} chunks, {len(citations)} citations)")
            print(f"      Answer: {response['answer'][:200]}...")
            if citations:
                c = citations[0]
                print(f"      Top citation: {c.get('org_name')} (score {c.get('relevance_score', 0):.2%})")
            return True

        print("FAIL: No answer generated")
        return False

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    print("\n" + "#" * 70)
    print("# MSU CLUB DISCOVERY RAG — SYSTEM TEST")
    print("#" * 70)

    results = {
        "Configuration":       test_configuration(),
        "Document Processing": test_document_processing(),
        "Vector Store":        test_vector_store(),
        "LLM Client":          test_llm_client(),
        "RAG Engine":          test_rag_engine(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")

    passed_count = sum(results.values())
    print(f"\n{passed_count}/{len(results)} tests passed")

    if passed_count == len(results):
        print("\nAll tests passed. Run: streamlit run app.py")
    else:
        print("\nFix the failures above, then re-run this script.")

    return passed_count == len(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
