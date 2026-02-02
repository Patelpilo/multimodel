"""
Chunking strategies for different modalities
"""
import re
from typing import List, Union
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """Semantic chunking for text content"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks"""
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks

class TableChunker:
    """Row-wise chunking for tables"""
    
    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
    
    def chunk_table(self, table: pd.DataFrame) -> List[str]:
        """Convert table to text chunks by rows"""
        if table is None or table.empty:
            return []
        
        chunks = []
        rows = []
        
        # Add header
        header_text = " | ".join(str(col) for col in table.columns)
        rows.append(f"Header: {header_text}")
        
        # Add rows in chunks
        for i, (_, row) in enumerate(table.iterrows()):
            row_text = " | ".join(str(val) for val in row.values)
            rows.append(f"Row {i+1}: {row_text}")
            
            # Create chunk every chunk_size rows
            if (i + 1) % self.chunk_size == 0:
                chunk_text = "\n".join(rows)
                chunks.append(chunk_text)
                rows = []
        
        # Add remaining rows
        if rows:
            chunk_text = "\n".join(rows)
            chunks.append(chunk_text)
        
        return chunks
    
    def table_to_text(self, table: pd.DataFrame) -> str:
        """Convert entire table to text format"""
        if table is None or table.empty:
            return ""
        
        # Create markdown-like representation
        lines = []
        
        # Header
        lines.append("| " + " | ".join(str(col) for col in table.columns) + " |")
        lines.append("|" + " --- |" * len(table.columns))
        
        # Rows
        for _, row in table.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
        
        return "\n".join(lines)