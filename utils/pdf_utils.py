"""
PDF processing utilities
"""
import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents to extract text, tables, and images"""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def extract_text(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text from all pages of a PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping page numbers to text content
        """
        text_pages = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        text_pages[page_num + 1] = text  # 1-indexed pages
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")

        return text_pages

    def extract_tables(self, pdf_path: str) -> Dict[int, List[pd.DataFrame]]:
        """
        Extract tables from PDF pages

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping page numbers to list of tables
        """
        tables = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    if page_tables:
                        # Convert to pandas DataFrames
                        df_tables = []
                        for table in page_tables:
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                                df_tables.append(df)
                        if df_tables:
                            tables[page_num + 1] = df_tables
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")

        return tables

    def extract_images(self, pdf_path: str, output_dir: str = None) -> Dict[int, List[str]]:
        """
        Extract images from PDF pages

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images

        Returns:
            Dictionary mapping page numbers to list of image paths
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        os.makedirs(output_dir, exist_ok=True)
        images = {}

        try:
            # Convert PDF to images
            pdf_images = convert_from_path(pdf_path, dpi=self.dpi)

            for page_num, image in enumerate(pdf_images):
                image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
                image.save(image_path, "PNG")
                if page_num + 1 not in images:
                    images[page_num + 1] = []
                images[page_num + 1].append(image_path)

        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")

        return images

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        try:
            with fitz.open(pdf_path) as pdf:
                return len(pdf)
        except Exception as e:
            logger.error(f"Error getting page count from {pdf_path}: {e}")
            return 0

    def extract_text_with_layout(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extract text with layout information

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping page numbers to list of text elements with bbox
        """
        layout_text = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    elements = []
                    for char in page.chars:
                        elements.append({
                            'text': char['text'],
                            'bbox': char['bbox'],
                            'size': char['size'],
                            'font': char['fontname']
                        })
                    if elements:
                        layout_text[page_num + 1] = elements
        except Exception as e:
            logger.error(f"Error extracting layout text from {pdf_path}: {e}")

        return layout_text
