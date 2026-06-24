"""
retriever.py - Vector store and retrieval logic
Uses an in-memory numpy-based vector store for ephemeral cloud deployments.
"""

import numpy as np
from typing import List, Dict, Any, Optional

DEFAULT_TOP_K = 5


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorStore:
    """
    In-memory vector store using numpy for retrieval.
    Perfect for ephemeral cloud deployments where local disk is wiped on restart.
    """

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self._chunks: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        self._chunks.extend(chunks)
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = DEFAULT_TOP_K,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        candidates = self._chunks
        if source_filter:
            candidates = [c for c in candidates if c.get("source") == source_filter]

        scored = []
        for chunk in candidates:
            sim = _cosine_similarity(query_embedding, chunk["embedding"])
            scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "text": c["text"],
                "source": c.get("source", "unknown"),
                "chunk_id": c.get("chunk_id", i),
                "score": round(sim, 4),
            }
            for i, (sim, c) in enumerate(scored[:top_k])
        ]

    def get_chunk_count(self) -> int:
        return len(self._chunks)

    def get_sources(self) -> List[str]:
        return sorted(list({c.get("source", "unknown") for c in self._chunks}))

    def clear(self):
        self._chunks = []

    def is_ready(self) -> bool:
        return len(self._chunks) > 0


def build_context(retrieved_chunks: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    """
    Combine retrieved chunks into a context string for the LLM prompt.
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(retrieved_chunks):
        header = f"[Source: {chunk['source']} | Relevance: {chunk['score']:.2f}]"
        body = chunk["text"]
        part = f"{header}\n{body}"

        if total_chars + len(part) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                context_parts.append(part[:remaining] + "...")
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n---\n\n".join(context_parts)
