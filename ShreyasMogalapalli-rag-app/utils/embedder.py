"""
embedder.py - Embedding generation using Ollama (nomic-embed-text)
Generates vector embeddings for document chunks and queries.
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    retries: int = 3,
    delay: float = 1.0
) -> Optional[List[float]]:
    """
    Generate an embedding vector for a single text string.

    Args:
        text: Input text to embed
        model: Ollama embedding model name
        retries: Number of retry attempts on failure
        delay: Seconds to wait between retries

    Returns:
        List of floats (embedding vector), or None on failure
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": model, "prompt": text}

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding")
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                raise ConnectionError(
                    "Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
                )
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise TimeoutError(f"Ollama embedding request timed out after {retries} attempts.")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise RuntimeError(f"Embedding failed: {e}")

    return None


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model: str = DEFAULT_EMBED_MODEL,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks: List of chunk dicts (must have 'text' key)
        model: Ollama embedding model name
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Same list of chunks, each with an added 'embedding' key
    """
    embedded_chunks = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"], model=model)
        if embedding is not None:
            chunk_with_embedding = {**chunk, "embedding": embedding}
            embedded_chunks.append(chunk_with_embedding)

        if progress_callback:
            progress_callback(i + 1, total)

    return embedded_chunks


def embed_query(query: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    """
    Generate an embedding for a user query.

    Args:
        query: User's question or search string
        model: Ollama embedding model name

    Returns:
        Embedding vector as list of floats
    """
    embedding = get_embedding(query, model=model)
    if embedding is None:
        raise RuntimeError("Failed to generate query embedding.")
    return embedding


def check_ollama_connection() -> bool:
    """Check if Ollama server is reachable."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_available_models() -> List[str]:
    """Return a list of locally available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
