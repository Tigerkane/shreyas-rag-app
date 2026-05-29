"""
loader.py - Document loading and chunking logic
Supports PDF, TXT, and MD files
"""

import os
from pathlib import Path
from typing import List, Dict, Any


def load_document(file_path: str) -> str:
    """Load raw text from a document file."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return _load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md")


def _load_pdf(file_path: str) -> str:
    """Load text from a PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page_num, page in enumerate(doc):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")


def _load_text(file_path: str) -> str:
    """Load text from a plain text or markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str = "unknown"
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: Raw document text
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        source: Source filename for metadata
    
    Returns:
        List of dicts with 'text', 'chunk_id', 'source', 'start_char'
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (period, newline)
        if end < len(text):
            # Look for a good break point within the last 100 chars of the chunk
            break_point = _find_break_point(text, end, lookback=100)
            if break_point:
                end = break_point

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append({
                "text": chunk_text_content,
                "chunk_id": chunk_id,
                "source": source,
                "start_char": start,
            })
            chunk_id += 1

        # Move forward with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break

    return chunks


def _find_break_point(text: str, position: int, lookback: int = 100) -> int:
    """Find the nearest sentence/paragraph break before a position."""
    search_start = max(0, position - lookback)
    segment = text[search_start:position]

    # Prefer paragraph breaks, then sentence endings
    for delimiter in ["\n\n", "\n", ". ", "! ", "? "]:
        idx = segment.rfind(delimiter)
        if idx != -1:
            return search_start + idx + len(delimiter)

    return position  # fallback to hard cut


def load_and_chunk(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """Convenience function: load a document and return chunks."""
    source = Path(file_path).name
    text = load_document(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source=source)
    return chunks
