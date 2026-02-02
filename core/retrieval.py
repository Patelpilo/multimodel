"""
Retrieval engine for multi-modal RAG
"""
import os
import sys
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.vector_store import FAISSVectorStore, SearchResult
from models.embedding_model import MultiModalEmbedder

logger = logging.getLogger(__name__)

class MultiModalRetrievalEngine:
    """Retrieval engine that combines multiple modalities"""

    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.embedder = MultiModalEmbedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve relevant document chunks for a query

        Args:
            query: User query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects
        """
        try:
            # Check if vector store has any chunks
            if self.vector_store is None or len(self.vector_store.chunks) == 0:
                logger.warning("Vector store is empty - no documents have been ingested")
                return []

            # Generate embedding for the query
            query_embedding = self.embedder.embed_text(query)[0]

            # Search vector store
            results = self.vector_store.search(query_embedding, k=top_k)

            logger.info(f"Retrieved {len(results)} chunks for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    def retrieve_with_filter(self, query: str, modality_filter: str = None,
                           page_filter: int = None, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve chunks with filtering options

        Args:
            query: User query
            modality_filter: Filter by modality ('text', 'table', 'image')
            page_filter: Filter by page number
            top_k: Number of top results to return

        Returns:
            Filtered list of retrieved chunks
        """
        # Get all results first
        results = self.retrieve(query, top_k=top_k * 2)  # Get more to filter

        # Apply filters
        filtered_results = []
        for result in results:
            if modality_filter and result.chunk.modality != modality_filter:
                continue
            if page_filter is not None and result.chunk.page != page_filter:
                continue
            filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        return filtered_results
