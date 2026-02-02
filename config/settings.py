"""
Configuration settings for the Multi-Modal RAG system
"""
import os
from pathlib import Path

class Config:
    """Configuration class for the RAG system"""

    # Vector Store Settings
    VECTOR_STORE_PATH = "data/vector_store"
    VECTOR_STORE_DIMENSION = 384  # Dimension for sentence-transformers models

    # Embedding Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
    EMBEDDING_DEVICE = "cpu"  # Use 'cuda' for GPU

    # LLM Settings
    LLM_MODEL = "llama3.2:3b"  # Smaller, more memory-efficient Ollama model
    MAX_TOKENS = 512
    TEMPERATURE = 0.7

    # Retrieval Settings
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.1

    # Chunking Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # PDF Processing Settings
    PDF_DPI = 300

    # Logging
    LOG_LEVEL = "INFO"

    # Data Paths
    DATA_DIR = Path("data")
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"

    def __init__(self):
        # Create necessary directories
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)

# Global config instance
config = Config()
