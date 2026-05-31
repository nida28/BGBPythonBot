"""Document parser for PDF and DOCX files with intelligent chunking."""

import re
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 512  # tokens (approximate)
OVERLAP = 50  # tokens


def validate_file(file_path: str, filename: str) -> None:
    """Validate file exists, size, and extension."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {filename}")
    
    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Maximum size: 10MB. Received: {file_size / 1024 / 1024:.1f}MB")
    
    ext = path.suffix.lower()
    if ext not in ['.pdf', '.docx']:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx")


def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text: {str(e)}")


def extract_docx_text(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {str(e)}")


def extract_text(file_path: str, filename: str) -> str:
    """Extract text from PDF or DOCX file."""
    ext = Path(filename).suffix.lower()
    if ext == '.pdf':
        return extract_pdf_text(file_path)
    elif ext == '.docx':
        return extract_docx_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_text(text: str) -> str:
    """Normalize text: remove extra whitespace, standardize line breaks."""
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Clean up common artifacts
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenation splits
    return text.strip()


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters for English)."""
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving structure."""
    # Split on periods, exclamation marks, question marks, but preserve abbreviations
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove trailing punctuation from sentences (.,!,?) for normalized output
    cleaned = [re.sub(r'[.!?]+$', '', s.strip()) for s in sentences if s.strip()]
    return cleaned


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Create overlapping chunks from text.
    
    Args:
        text: Input text to chunk
        chunk_size: Approximate tokens per chunk
        overlap: Approximate tokens of overlap between chunks
    
    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        
        # If adding this sentence exceeds chunk_size, start new chunk
        if current_token_count + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap (keep last ~overlap tokens worth of sentences)
            overlap_tokens = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                overlap_tokens += estimate_token_count(s)
                overlap_sentences.insert(0, s)
                if overlap_tokens >= overlap:
                    break
            
            current_chunk = overlap_sentences
            current_token_count = overlap_tokens
        
        current_chunk.append(sentence)
        current_token_count += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def parse_document(file_path: str, filename: str) -> List[Dict]:
    """Parse document and return chunks with metadata.
    
    Args:
        file_path: Path to temporary file
        filename: Original filename
    
    Returns:
        List of chunk dictionaries with keys: text, source, page, filename
    """
    # Validate file
    validate_file(file_path, filename)
    
    # Extract text
    text = extract_text(file_path, filename)
    
    if not text.strip():
        raise ValueError("Document is empty or contains no extractable text")
    
    # Normalize
    text = normalize_text(text)
    
    # Create chunks
    chunks_text = create_chunks(text)
    
    if not chunks_text:
        raise ValueError("Failed to create chunks from document")
    
    # Add metadata to chunks
    chunks = [
        {
            "text": chunk,
            "source": "uploaded",
            "filename": filename,
            "page": i + 1,  # 1-indexed for display
            "embedding": None  # Will be filled by document_store
        }
        for i, chunk in enumerate(chunks_text)
    ]
    
    return chunks
