"""
embedder.py - Embedding generation using sentence-transformers
Generates vector embeddings for document chunks and queries.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
_model_cache = {}

def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
) -> Optional[List[float]]:
    """Generate an embedding vector for a single text string."""
    try:
        transformer = _get_model(model)
        # return as list of floats
        return transformer.encode(text).tolist()
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def embed_chunks(
    chunks: List[Dict[str, Any]],
    model: str = DEFAULT_EMBED_MODEL,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """Generate embeddings for a list of text chunks."""
    embedded_chunks = []
    total = len(chunks)
    
    if total == 0:
        return []

    try:
        transformer = _get_model(model)
        texts = [chunk["text"] for chunk in chunks]
        # Batch encode is much faster
        embeddings = transformer.encode(texts)
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_with_embedding = {**chunk, "embedding": emb.tolist()}
            embedded_chunks.append(chunk_with_embedding)
            if progress_callback:
                progress_callback(i + 1, total)
                
    except Exception as e:
        print(f"Batch embedding failed: {e}")
        # fallback to single embedding
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk["text"], model=model)
            if emb is not None:
                embedded_chunks.append({**chunk, "embedding": emb})
            if progress_callback:
                progress_callback(i + 1, total)

    return embedded_chunks

def embed_query(query: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    """Generate an embedding for a user query."""
    embedding = get_embedding(query, model=model)
    if embedding is None:
        raise RuntimeError("Failed to generate query embedding.")
    return embedding

def check_groq_api_key(api_key: str) -> bool:
    """Check if the provided Groq API key is valid."""
    if not api_key:
        return False
    from groq import Groq
    try:
        client = Groq(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def list_available_chat_models() -> List[str]:
    """Return a list of available Groq chat models."""
    return [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "qwen-2.5-coder-32b"
    ]

def check_ollama_connection() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def list_available_ollama_models() -> List[str]:
    """Return a list of locally available Ollama models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
