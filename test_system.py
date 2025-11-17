"""
System Test Script
Tests all components of the RAG system to ensure proper setup
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from src.data_processing import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_client import get_llm_client
from src.rag_engine import RAGEngine


def test_configuration():
    """Test 1: Configuration validation"""
    print("\n" + "="*80)
    print("TEST 1: CONFIGURATION")
    print("="*80)

    try:
        config.validate_config()
        print("✅ Configuration is valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_document_processing():
    """Test 2: Document processing"""
    print("\n" + "="*80)
    print("TEST 2: DOCUMENT PROCESSING")
    print("="*80)

    try:
        processor = DocumentProcessor()

        # Check if sample document exists
        sample_doc = config.RAW_DATA_DIR / "accessibility_club.txt"
        if not sample_doc.exists():
            print(f"⚠️  Sample document not found: {sample_doc}")
            print("   Please ensure data/raw/accessibility_club.txt exists")
            return False

        # Process document
        chunks = processor.process_document(sample_doc)

        if chunks:
            print(f"✅ Successfully processed document into {len(chunks)} chunks")
            print(f"   Sample chunk: {chunks[0]['text'][:100]}...")
            return True
        else:
            print("❌ No chunks created")
            return False

    except Exception as e:
        print(f"❌ Document processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test 3: Vector store connection"""
    print("\n" + "="*80)
    print("TEST 3: VECTOR STORE (PINECONE)")
    print("="*80)

    try:
        vs = VectorStore()

        if vs.index is None:
            print("❌ Failed to connect to Pinecone index")
            print("   Make sure:")
            print("   1. PINECONE_API_KEY is set in .env")
            print("   2. Index 'msu-clubs-index' exists")
            print("   3. Index is configured with llama-text-embed-v2")
            return False

        # Get stats
        stats = vs.get_stats()
        if stats:
            print("✅ Successfully connected to Pinecone")
            print(f"   Index: {config.PINECONE_INDEX_NAME}")
            print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
            return True
        else:
            print("⚠️  Connected but couldn't get stats")
            return True

    except Exception as e:
        print(f"❌ Vector store error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_client():
    """Test 4: LLM client"""
    print("\n" + "="*80)
    print("TEST 4: LLM CLIENT")
    print("="*80)

    try:
        llm = get_llm_client()
        print(f"✅ LLM client initialized ({config.LLM_PROVIDER})")

        # Test generation
        print("   Testing generation with simple prompt...")
        response = llm.generate(
            prompt="Say 'Hello from MSU RAG!'",
            temperature=0.5,
            max_tokens=50
        )

        if response and len(response) > 0:
            print(f"✅ LLM response received: '{response[:100]}...'")
            return True
        else:
            print("❌ Empty response from LLM")
            return False

    except Exception as e:
        print(f"❌ LLM client error: {e}")
        print("   Make sure your API key is set:")
        if config.LLM_PROVIDER == "groq":
            print("   GROQ_API_KEY in .env")
        else:
            print("   ANTHROPIC_API_KEY in .env")
        import traceback
        traceback.print_exc()
        return False


def test_rag_engine():
    """Test 5: RAG engine end-to-end"""
    print("\n" + "="*80)
    print("TEST 5: RAG ENGINE (END-TO-END)")
    print("="*80)

    try:
        rag = RAGEngine()
        print("✅ RAG engine initialized")

        # Test query
        test_query = "What is the Accessibility Club?"
        print(f"\n   Testing query: '{test_query}'")

        response = rag.query(test_query, top_k=3)

        if response.get('answer'):
            print(f"✅ Query successful!")
            print(f"   Answer: {response['answer'][:200]}...")
            print(f"   Retrieved chunks: {len(response.get('retrieved_chunks', []))}")
            print(f"   Citations: {len(response.get('citations', []))}")

            if response.get('citations'):
                print(f"\n   Top citation:")
                citation = response['citations'][0]
                print(f"   - Club: {citation.get('club_name')}")
                print(f"   - Relevance: {citation.get('relevance_score', 0):.2%}")

            return True
        else:
            print("❌ No answer generated")
            return False

    except Exception as e:
        print(f"❌ RAG engine error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# MSU CLUB DISCOVERY RAG - SYSTEM TEST")
    print("#"*80)

    results = {
        "Configuration": test_configuration(),
        "Document Processing": test_document_processing(),
        "Vector Store": test_vector_store(),
        "LLM Client": test_llm_client(),
        "RAG Engine": test_rag_engine()
    }

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open browser to http://localhost:8501")
        print("  3. Start asking questions about MSU clubs!")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the app.")
        print("\nCommon fixes:")
        print("  - Check API keys in .env file")
        print("  - Verify Pinecone index exists and is configured correctly")
        print("  - Make sure data/raw/accessibility_club.txt exists")
        print("  - Run: python ingest_data.py")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
