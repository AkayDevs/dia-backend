from typing import List, Dict, Any
import logging
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io

from app.services.ml.base import BaseTableDetector
from .image_detector import ImageTableDetector

logger = logging.getLogger(__name__)

class PDFTableDetector(BaseTableDetector):
    """Service for detecting tables in PDF documents."""
    
    def __init__(self):
        """Initialize the PDF table detection service."""
        logger.debug("Initializing PDF Table Detection Service")
        self.image_detector = ImageTableDetector()
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        return ["pdf"]

    def detect_tables(
        self, 
        file_path: str, 
        confidence_threshold: float = 0.5,
        min_row_count: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect tables in a PDF document.
        
        Args:
            file_path: Path to the PDF file
            confidence_threshold: Minimum confidence score for detections
            min_row_count: Minimum number of rows to consider a valid table
            
        Returns:
            List of detected tables with their properties
        """
        logger.debug(f"Detecting tables in PDF: {file_path}")
        
        try:
            # Open PDF
            doc = fitz.open(file_path)
            all_tables = []
            
            # Process each page
            for page_num, page in enumerate(doc):
                logger.debug(f"Processing page {page_num + 1}")
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
                img_data = pix.tobytes()
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                # Save temporary image
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Detect tables in the page image
                page_tables = self.image_detector.detect_tables(
                    img_bytes,
                    confidence_threshold=confidence_threshold,
                    min_row_count=min_row_count
                )
                
                # Add page number to results
                for table in page_tables:
                    table["page_number"] = page_num + 1
                    
                all_tables.extend(page_tables)
            
            logger.info(f"Detected {len(all_tables)} tables in PDF")
            return all_tables
            
        except Exception as e:
            logger.error(f"PDF table detection failed: {str(e)}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Boolean indicating if file is valid
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"PDF file not found: {file_path}")
                return False
            
            # Check if file is a valid PDF
            try:
                doc = fitz.open(file_path)
                doc.close()
                return True
            except Exception:
                logger.error(f"Invalid PDF file: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False 