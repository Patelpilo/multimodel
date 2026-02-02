"""
Vector database for storing and retrieving embeddings
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import os
from dataclasses import dataclass
import logging
import torch

from core.ingestion import DocumentChunk
from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk: DocumentChunk
    score: float
    rank: int

class FAISSVectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def create_index(self, use_gpu: bool = False):
        """Create FAISS index"""
        if use_gpu and torch.cuda.is_available():
            # Try to use GPU
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            # CPU index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        logger.info(f"Created FAISS index with dimension {self.embedding_dim}")
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]):
        """
        Add chunks and their embeddings to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of corresponding embeddings
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
            return
        
        if self.index is None:
            self.create_index()
        
        # Convert embeddings to numpy array
        embedding_array = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_array)
        
        # Add to index
        self.index.add(embedding_array)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self,
               query_embedding: np.ndarray,
               k: int = 5,
               score_threshold: float = None) -> List[SearchResult]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []

        # Use config threshold if none provided
        if score_threshold is None:
            score_threshold = config.SIMILARITY_THRESHOLD

        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):  # Invalid index
                continue

            if distance < score_threshold:
                continue

            result = SearchResult(
                chunk=self.chunks[idx],
                score=float(distance),
                rank=i + 1
            )
            results.append(result)

        return results
    
    def hybrid_search(self, 
                      query_embedding: np.ndarray, 
                      query_text: str,
                      k: int = 5,
                      alpha: float = 0.7) -> List[SearchResult]:
        """
        Hybrid search combining vector and keyword matching
        
        Args:
            query_embedding: Vector embedding of query
            query_text: Text query for keyword matching
            k: Number of results
            alpha: Weight for vector vs keyword (0-1)
            
        Returns:
            Combined search results
        """
        # Vector search
        vector_results = self.search(query_embedding, k * 2)
        
        # Simple keyword matching (can be enhanced with BM25)
        keyword_results = []
        query_keywords = set(query_text.lower().split())
        
        for i, chunk in enumerate(self.chunks):
            chunk_keywords = set(chunk.content.lower().split())
            common_keywords = query_keywords.intersection(chunk_keywords)
            
            if common_keywords:
                # Simple Jaccard similarity
                similarity = len(common_keywords) / len(query_keywords.union(chunk_keywords))
                
                result = SearchResult(
                    chunk=chunk,
                    score=similarity,
                    rank=0
                )
                keyword_results.append(result)
        
        # Sort keyword results
        keyword_results.sort(key=lambda x: x.score, reverse=True)
        keyword_results = keyword_results[:k]
        
        # Combine results
        combined_results = {}
        
        # Add vector results with weighted scores
        for result in vector_results:
            combined_results[result.chunk.chunk_id] = {
                'chunk': result.chunk,
                'score': result.score * alpha,
                'source': 'vector'
            }
        
        # Add keyword results with weighted scores
        for result in keyword_results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in combined_results:
                # Combine scores
                combined_results[chunk_id]['score'] += result.score * (1 - alpha)
                combined_results[chunk_id]['source'] = 'hybrid'
            else:
                combined_results[chunk_id] = {
                    'chunk': result.chunk,
                    'score': result.score * (1 - alpha),
                    'source': 'keyword'
                }
        
        # Convert to list and sort
        final_results = []
        for data in combined_results.values():
            final_results.append(SearchResult(
                chunk=data['chunk'],
                score=data['score'],
                rank=0
            ))
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Re-rank
        for i, result in enumerate(final_results[:k]):
            result.rank = i + 1
        
        return final_results[:k]
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")

        # Save chunks and metadata
        with open(f"{filepath}.chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        if not os.path.exists(f"{filepath}.index"):
            logger.error(f"Index file not found: {filepath}.index")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load chunks
            with open(f"{filepath}.chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Loaded vector store from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False