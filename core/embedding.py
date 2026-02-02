"""
Embedding utilities for multi-modal content
"""
import os
import sys
import logging
from typing import List, Union
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedding_model import MultiModalEmbedder

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings"""

    def __init__(self):
        self.embedder = MultiModalEmbedder()

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text content

        Args:
            texts: Single text or list of texts

        Returns:
            Numpy array of embeddings
        """
        return self.embedder.embed_text(texts)

    def generate_batch_embeddings(self, chunks: List) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of chunks

        Args:
            chunks: List of DocumentChunk objects or text strings

        Returns:
            List of embeddings
        """
        return self.embedder.embed_batch(chunks)
