"""
Document Processing Module
Handles extraction, cleaning, chunking, and metadata extraction from club documents
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class DocumentProcessor:
    """
    Processes club documents: extraction, cleaning, chunking, and metadata extraction
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document processor

        Args:
            chunk_size: Number of tokens per chunk (default from config)
            chunk_overlap: Number of overlapping tokens between chunks (default from config)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # Initialize tiktoken encoder for token counting (using cl100k_base for approximation)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize text splitter with token-based splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file using PyMuPDF

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        
        try:
            doc = fitz.open(str(file_path))
            text = ""

            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()

            doc.close()
            return text

        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    def extract_text_from_txt(self, file_path: Path) -> str:
        """
        Extract text from TXT file

        Args:
            file_path: Path to TXT file

        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and common artifacts
        text = re.sub(r'\b\d+\s*Updated\s+\d+\s+\w+\s+\d{4}\b', '', text)

        # Normalize line breaks
        text = text.replace('\n ', '\n')

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def extract_metadata_from_text(self, text: str, file_name: str) -> Dict[str, any]:
        """
        Extract metadata from document text using rule-based patterns
        Falls back to providing basic metadata if extraction fails

        Args:
            text: Document text
            file_name: Name of the source file

        Returns:
            Dictionary containing metadata fields
        """
        metadata = {
            "club_name": "",
            "dues": None,
            "meeting_frequency": "",
            "membership_requirements": [],
            "source_file": file_name,
            "last_updated": None,
            "contact_info": "",
        }

        # Extract club name (usually in title or first Article)
        club_name_match = re.search(
            r'(?:name of this organization shall be|organization:)\s+(?:the\s+)?([^.]+(?:Club|Organization|Society)[^.]*)',
            text,
            re.IGNORECASE
        )
        if club_name_match:
            metadata["club_name"] = club_name_match.group(1).strip()
        else:
            # Fallback: use filename
            metadata["club_name"] = file_name.replace('.pdf', '').replace('_', ' ').title()

        # Extract dues/fees
        dues_match = re.search(
            r'(?:dues|fee|cost)[^\d]*\$?(\d+(?:\.\d{2})?)',
            text,
            re.IGNORECASE
        )
        if dues_match:
            metadata["dues"] = float(dues_match.group(1))

        # Extract meeting frequency
        meeting_patterns = [
            r'meet(?:ing)?s?\s+(?:every\s+)?(\w+(?:\s+\w+)?)',
            r'(?:bi-?weekly|monthly|weekly|daily)',
        ]
        for pattern in meeting_patterns:
            meeting_match = re.search(pattern, text, re.IGNORECASE)
            if meeting_match:
                metadata["meeting_frequency"] = meeting_match.group(0).strip()
                break

        # Extract last updated date
        date_match = re.search(
            r'Updated\s+(\d+\s+\w+\s+\d{4})',
            text,
            re.IGNORECASE
        )
        if date_match:
            metadata["last_updated"] = date_match.group(1)

        # Extract membership requirements (look for eligibility section)
        if "membership" in text.lower():
            # Extract paragraph containing membership info
            membership_section = re.search(
                r'(membership[^:]*:.*?)(?=Article|Section|$)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if membership_section:
                membership_text = membership_section.group(1)[:200]  # First 200 chars
                metadata["membership_requirements"] = [membership_text.strip()]

        return metadata

    def chunk_text(self, text: str, metadata: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split text into chunks with metadata

        Args:
            text: Cleaned document text
            metadata: Document-level metadata

        Returns:
            List of chunks with metadata
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Create chunk objects with metadata
        chunk_objects = []
        for idx, chunk in enumerate(chunks):
            chunk_obj = {
                "text": chunk,
                "metadata": {
                    **metadata,  # Include all document metadata
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def process_document(self, file_path: Path) -> List[Dict[str, any]]:
        """
        Complete pipeline: extract, clean, chunk, and add metadata

        Args:
            file_path: Path to document file

        Returns:
            List of processed chunks with metadata
        """
        print(f"\n📄 Processing: {file_path.name}")

        # Step 1: Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            raw_text = self.extract_text_from_txt(file_path)
        else:
            print(f"  ⚠️  Unsupported file format: {file_path.suffix}")
            return []

        if not raw_text:
            print(f"  ⚠️  No text extracted from {file_path.name}")
            return []

        # Step 2: Clean text
        cleaned_text = self.clean_text(raw_text)
        print(f"  ✓ Extracted {len(cleaned_text)} characters")

        # Step 3: Extract metadata
        metadata = self.extract_metadata_from_text(cleaned_text, file_path.name)
        print(f"  ✓ Metadata: {metadata.get('club_name', 'Unknown')}")
        if metadata.get('dues'):
            print(f"    - Dues: ${metadata['dues']}")

        # Step 4: Chunk text
        chunks = self.chunk_text(cleaned_text, metadata)
        print(f"  ✓ Created {len(chunks)} chunks (~{self.chunk_size} tokens each)")

        return chunks

    def process_directory(self, directory: Path) -> List[Dict[str, any]]:
        """
        Process all documents in a directory

        Args:
            directory: Path to directory containing documents

        Returns:
            List of all processed chunks from all documents
        """
        all_chunks = []

        # Find all supported document files
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in config.SUPPORTED_FORMATS:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)

        print(f"\n✓ Total: {len(all_chunks)} chunks from {directory}")
        return all_chunks


# ============================================================================
# TESTING / STANDALONE EXECUTION
# ============================================================================
if __name__ == "__main__":
    """Test document processing with sample files"""

    # Initialize processor
    processor = DocumentProcessor()

    # Test with raw data directory
    chunks = processor.process_directory(config.RAW_DATA_DIR)

    # Display sample chunk
    if chunks:
        print("\n" + "="*80)
        print("SAMPLE CHUNK:")
        print("="*80)
        sample = chunks[0]
        print(f"Text: {sample['text'][:300]}...")
        print(f"\nMetadata: {sample['metadata']}")
