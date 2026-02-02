"""
Offline embedding models for multi-modal content
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class MultiModalEmbedder:
    """Generate embeddings for all modalities"""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device or config.EMBEDDING_DEVICE

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device=self.device
            )

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text

        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty texts
        texts = [text if text and text.strip() else " " for text in texts]

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_dim))

    def embed_batch(self, chunks: List, batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of chunks

        Args:
            chunks: List of DocumentChunk objects or text strings
            batch_size: Batch size for embedding

        Returns:
            List of embeddings
        """
        if not chunks:
            return []

        # Extract text from chunks if needed
        if hasattr(chunks[0], 'content'):
            texts = [chunk.content for chunk in chunks]
        else:
            texts = chunks

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embed_text(batch_texts)
            all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            return list(embeddings)
        else:
            return []
