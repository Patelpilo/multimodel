"""
OCR utilities for image text extraction
"""
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import torch

# Check if PaddleOCR is available
try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handle OCR for scanned documents and images"""
    
    def __init__(self, engine: str = "paddle", language: str = "en"):
        self.engine = engine
        self.language = language

        if engine == "paddle":
            if PADDLE_AVAILABLE:
                # Initialize PaddleOCR
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=language,
                    show_log=False,
                    use_gpu=torch.cuda.is_available()
                )
            else:
                logger.warning("PaddleOCR not available, falling back to Tesseract")
                self.engine = "tesseract"
                engine = "tesseract"

        if engine == "tesseract":
            # Check if tesseract is installed
            try:
                pytesseract.get_tesseract_version()
            except:
                logger.warning("Tesseract not found. Please install: sudo apt-get install tesseract-ocr")
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, List[dict]]:
        """
        Extract text from image using selected OCR engine
        
        Args:
            image: numpy array of image
            
        Returns:
            Tuple of (full_text, bboxes_with_text)
        """
        if self.engine == "paddle":
            return self._extract_with_paddle(image)
        else:
            return self._extract_with_tesseract(image)
    
    def _extract_with_paddle(self, image: np.ndarray) -> Tuple[str, List[dict]]:
        """Extract text using PaddleOCR"""
        try:
            result = self.ocr.ocr(image, cls=True)
            if result[0] is None:
                return "", []
            
            full_text = ""
            bboxes = []
            
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                full_text += text + " "
                bboxes.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(confidence)
                })
            
            return full_text.strip(), bboxes
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return "", []
    
    def _extract_with_tesseract(self, image: np.ndarray) -> Tuple[str, List[dict]]:
        """Extract text using Tesseract"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                pil_image, 
                output_type=pytesseract.Output.DICT,
                lang=self.language
            )
            
            full_text = ""
            bboxes = []
            
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                if int(ocr_data['conf'][i]) > 60:  # Confidence threshold
                    text = ocr_data['text'][i].strip()
                    if text:
                        full_text += text + " "
                        bboxes.append({
                            "bbox": (
                                ocr_data['left'][i],
                                ocr_data['top'][i],
                                ocr_data['left'][i] + ocr_data['width'][i],
                                ocr_data['top'][i] + ocr_data['height'][i]
                            ),
                            "text": text,
                            "confidence": int(ocr_data['conf'][i])
                        })
            
            return full_text.strip(), bboxes
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return "", []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        denoised = cv2.medianBlur(thresh, 3)
        
        return denoised