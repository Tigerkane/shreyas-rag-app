"""
retriever.py - Vector store and retrieval logic
Uses ChromaDB as a local persistent vector database.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


CHROMA_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION = "rag_documents"
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
    Local vector store using ChromaDB for persistent storage and retrieval.
    Falls back to in-memory numpy search if ChromaDB is unavailable.
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION, persist_dir: str = CHROMA_DB_DIR):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._use_chroma = False

        # In-memory fallback
        self._chunks: List[Dict[str, Any]] = []

        self._init_chroma()

    def _init_chroma(self):
        """Try to initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._use_chroma = True
        except ImportError:
            self._use_chroma = False
        except Exception:
            self._use_chroma = False

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add embedded chunks to the vector store.

        Args:
            chunks: List of dicts with keys: text, embedding, chunk_id, source

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        if self._use_chroma:
            return self._add_to_chroma(chunks)
        else:
            return self._add_to_memory(chunks)

    def _add_to_chroma(self, chunks: List[Dict[str, Any]]) -> int:
        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk in chunks:
            chunk_id = f"{chunk.get('source', 'doc')}_{chunk.get('chunk_id', 0)}"
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk.get("source", "unknown"),
                "chunk_id": chunk.get("chunk_id", 0),
                "start_char": chunk.get("start_char", 0),
            })

        # Add in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
        return len(ids)

    def _add_to_memory(self, chunks: List[Dict[str, Any]]) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = DEFAULT_TOP_K,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Embedding vector of the user query
            top_k: Number of results to return
            source_filter: Optional — only return chunks from this source file

        Returns:
            List of result dicts with: text, source, score, chunk_id
        """
        if self._use_chroma:
            return self._search_chroma(query_embedding, top_k, source_filter)
        else:
            return self._search_memory(query_embedding, top_k, source_filter)

    def _search_chroma(self, query_embedding, top_k, source_filter):
        where_filter = {"source": source_filter} if source_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i, doc in enumerate(results["documents"][0]):
            score = 1 - results["distances"][0][i]  # ChromaDB returns distance
            meta = results["metadatas"][0][i]
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", i),
                "score": round(score, 4),
            })
        return output

    def _search_memory(self, query_embedding, top_k, source_filter):
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
        """Return total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._chunks)

    def get_sources(self) -> List[str]:
        """Return list of unique source documents in the store."""
        if self._use_chroma and self._collection:
            results = self._collection.get(include=["metadatas"])
            sources = list({m["source"] for m in results["metadatas"]})
            return sorted(sources)
        else:
            return sorted(list({c.get("source", "unknown") for c in self._chunks}))

    def clear(self):
        """Remove all documents from the vector store."""
        if self._use_chroma and self._client:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        self._chunks = []

    def is_ready(self) -> bool:
        """Check if the vector store has any documents."""
        return self.get_chunk_count() > 0


def build_context(retrieved_chunks: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    """
    Combine retrieved chunks into a context string for the LLM prompt.

    Args:
        retrieved_chunks: List of result dicts from search()
        max_chars: Maximum total character length of context

    Returns:
        Formatted context string
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
