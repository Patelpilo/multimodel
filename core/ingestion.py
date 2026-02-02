"""
Document ingestion pipeline for multi-modal content
"""
import os
import sys
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_utils import PDFProcessor
from core.chunking import TextChunker, TableChunker
from config.settings import config

logger = logging.getLogger(__name__)

class DocumentChunk:
    """Represents a chunk of document content"""

    def __init__(self, content: str, page: int, modality: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.page = page
        self.modality = modality  # 'text', 'table', 'image'
        self.metadata = metadata or {}
        self.embedding = None
        self.chunk_id = f"{modality}_{page}_{hash(content) % 1000000}"

class DocumentIngestionPipeline:
    """Pipeline for ingesting and processing multi-modal documents"""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.table_chunker = TableChunker()

    def process_document(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Process a PDF document and extract chunks

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of DocumentChunk objects
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Processing document: {pdf_path}")

        chunks = []

        try:
            # Extract text content
            text_pages = self.pdf_processor.extract_text(pdf_path)
            for page_num, text in text_pages.items():
                if text.strip():
                    # Chunk the text
                    text_chunks = self.text_chunker.chunk_text(text)
                    for chunk_text in text_chunks:
                        chunk = DocumentChunk(
                            content=chunk_text,
                            page=page_num,
                            modality='text',
                            metadata={'source': pdf_path}
                        )
                        chunks.append(chunk)

            # Extract tables
            tables = self.pdf_processor.extract_tables(pdf_path)
            for page_num, page_tables in tables.items():
                for table_idx, table in enumerate(page_tables):
                    if table is not None and not table.empty:
                        # Convert table to text chunks
                        table_chunks = self.table_chunker.chunk_table(table)
                        for chunk_text in table_chunks:
                            chunk = DocumentChunk(
                                content=chunk_text,
                                page=page_num,
                                modality='table',
                                metadata={
                                    'source': pdf_path,
                                    'table_index': table_idx
                                }
                            )
                            chunks.append(chunk)

            # Extract images (if needed)
            # This would require OCR processing
            # For now, we'll skip image extraction

            logger.info(f"Extracted {len(chunks)} chunks from {len(text_pages)} pages")

        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {e}")
            raise

        return chunks
