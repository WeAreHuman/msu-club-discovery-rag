"""
Data Ingestion Script
Processes club documents and uploads them to Pinecone vector database
"""

import argparse
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import DocumentProcessor
from src.vector_store import VectorStore
import config


def ingest_documents(
    input_dir: Path = None,
    clear_existing: bool = False,
    batch_size: int = 100
):
    """
    Main ingestion pipeline

    Args:
        input_dir: Directory containing club documents (defaults to config.RAW_DATA_DIR)
        clear_existing: Whether to clear existing data in namespace before ingesting
        batch_size: Number of chunks to upload per batch
    """

    input_dir = input_dir or config.RAW_DATA_DIR

    print("="*80)
    print("MSU CLUB DISCOVERY - DATA INGESTION")
    print("="*80)

    # Validate configuration
    try:
        config.validate_config()
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nPlease set up your .env file with required API keys:")
        print("  - PINECONE_API_KEY")
        print(f"  - {config.LLM_PROVIDER.upper()}_API_KEY")
        return

    # Check input directory
    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        print(f"\nPlease create the directory and add club documents:")
        print(f"  mkdir -p {input_dir}")
        print(f"  # Add your PDF/TXT files to {input_dir}")
        return

    # Count files
    file_count = len([f for f in input_dir.rglob('*') if f.suffix.lower() in config.SUPPORTED_FORMATS])
    if file_count == 0:
        print(f"\n⚠️  No supported documents found in {input_dir}")
        print(f"   Supported formats: {', '.join(config.SUPPORTED_FORMATS)}")
        return

    print(f"\n📁 Input directory: {input_dir}")
    print(f"📄 Found {file_count} document(s)")

    # Initialize components
    print("\n" + "-"*80)
    print("INITIALIZING COMPONENTS")
    print("-"*80)

    processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    vector_store = VectorStore(
        api_key=config.PINECONE_API_KEY,
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE
    )

    # Clear existing data if requested
    if clear_existing:
        print(f"\n🗑️  Clearing existing data in namespace '{config.PINECONE_NAMESPACE}'...")
        confirm = input("   Are you sure? This cannot be undone. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            vector_store.delete_namespace()
        else:
            print("   Skipped deletion.")

    # Process documents
    print("\n" + "-"*80)
    print("PROCESSING DOCUMENTS")
    print("-"*80)

    chunks = processor.process_directory(input_dir)

    if not chunks:
        print("\n⚠️  No chunks created. Please check your documents.")
        return

    # Upload to Pinecone
    print("\n" + "-"*80)
    print("UPLOADING TO PINECONE")
    print("-"*80)

    upserted_count = vector_store.upsert_chunks(chunks, batch_size=batch_size)

    # Summary
    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80)
    print(f"✓ Documents processed: {file_count}")
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Chunks uploaded: {upserted_count}")
    print(f"✓ Namespace: {config.PINECONE_NAMESPACE}")
    print(f"✓ Index: {config.PINECONE_INDEX_NAME}")

    # Display index stats
    print("\n📊 Index Statistics:")
    stats = vector_store.get_stats()
    if stats:
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        namespaces = stats.get('namespaces', {})
        if config.PINECONE_NAMESPACE in namespaces:
            ns_count = namespaces[config.PINECONE_NAMESPACE].get('vector_count', 0)
            print(f"   Vectors in '{config.PINECONE_NAMESPACE}': {ns_count}")

    print("\n✓ Data ingestion successful! You can now run the Streamlit app.")
    print(f"  Run: streamlit run app.py")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """
    Command line interface for data ingestion
    """

    parser = argparse.ArgumentParser(
        description="Ingest MSU club documents into Pinecone vector database"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=f"Directory containing club documents (default: {config.RAW_DATA_DIR})"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data in namespace before ingesting"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks to upload per batch (default: 100)"
    )

    args = parser.parse_args()

    # Convert input_dir to Path if provided
    input_dir = Path(args.input_dir) if args.input_dir else None
    # if args.input_dir:
    #     input_dir=Path(args.input_dir)
    # else:
    #     input_dir=None


    # Run ingestion
    ingest_documents(
        input_dir=input_dir,
        clear_existing=args.clear,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
