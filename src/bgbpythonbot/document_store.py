"""In-memory vector store for uploaded document chunks."""

from typing import List, Dict, Callable
import numpy as np


class DocumentStore:
    """In-memory vector store for uploaded document chunks."""
    
    def __init__(self, cosine_similarity_fn: Callable):
        """Initialize store.
        
        Args:
            cosine_similarity_fn: Function to compute cosine similarity between embeddings
        """
        self.chunks = []  # List of chunks with embeddings
        self.cosine_similarity = cosine_similarity_fn
    
    def add_chunks(self, chunks: List[Dict], embedding_fn: Callable) -> None:
        """Add chunks and generate their embeddings.
        
        Args:
            chunks: List of chunk dicts with 'text' key and optional metadata
            embedding_fn: Function to generate embedding for a text string
        """
        self.clear()  # Only one file at a time
        
        for chunk in chunks:
            embedding = embedding_fn(chunk["text"])
            chunk["embedding"] = embedding
            self.chunks.append(chunk)
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Search chunks by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
        
        Returns:
            List of chunks ranked by similarity score (highest first)
        """
        if not self.chunks:
            return []
        
        ranked = sorted(
            [
                {
                    "chunk": chunk,
                    "score": self.cosine_similarity(query_embedding, chunk["embedding"])
                }
                for chunk in self.chunks
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]
        
        return ranked
    
    def clear(self) -> None:
        """Clear all stored chunks."""
        self.chunks = []
    
    def get_chunk_count(self) -> int:
        """Get number of stored chunks."""
        return len(self.chunks)
    
    def get_filename(self) -> str:
        """Get filename of currently stored document, or empty string if none."""
        if self.chunks:
            return self.chunks[0].get("filename", "Unknown")
        return ""
    
    def has_documents(self) -> bool:
        """Check if any documents are stored."""
        return len(self.chunks) > 0
