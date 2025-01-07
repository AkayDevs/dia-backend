from typing import Dict, Any, List
import cv2
import numpy as np
import pytesseract
import logging
from .base import BaseTableDetector, BaseTextExtractor

logger = logging.getLogger(__name__)

class ImageTableDetector(BaseTableDetector):
    """Detector for tables in image documents."""
    
    async def detect_tables(self, file_path: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and extract tables from image."""
        try:
            # Read image
            image = cv2.imread(file_path)
            confidence_threshold = parameters.get("confidence_threshold", 0.5)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                # Get rectangle bounding contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and aspect ratio
                area = cv2.contourArea(contour)
                rect_area = w * h
                if rect_area == 0:
                    continue
                    
                fill_ratio = area / rect_area
                confidence = fill_ratio if 0.1 <= fill_ratio <= 0.9 else 0
                
                if confidence < confidence_threshold:
                    continue
                
                # Extract table region
                table_region = image[y:y+h, x:x+w]
                
                # Use OCR to get text from table region
                table_text = pytesseract.image_to_data(table_region, output_type=pytesseract.Output.DICT)
                
                # Process OCR results to get table structure
                table_data = self._process_ocr_results(table_text)
                
                table_info = {
                    "bbox": [x, y, x+w, y+h],
                    "confidence": confidence,
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0,
                    "data": table_data
                }
                
                tables.append(table_info)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error detecting tables in image: {str(e)}")
            raise
            
    def _process_ocr_results(self, ocr_data: Dict[str, Any]) -> List[List[str]]:
        """Process OCR results to get table structure."""
        # Group text by lines based on top coordinate
        lines = {}
        for i in range(len(ocr_data['text'])):
            if not ocr_data['text'][i].strip():
                continue
                
            top = ocr_data['top'][i]
            if top not in lines:
                lines[top] = []
            
            lines[top].append({
                'text': ocr_data['text'][i],
                'left': ocr_data['left'][i],
                'conf': float(ocr_data['conf'][i])
            })
        
        # Sort lines by top coordinate
        sorted_lines = sorted(lines.items())
        
        # Convert to table data
        table_data = []
        for _, line in sorted_lines:
            # Sort cells in line by left coordinate
            sorted_cells = sorted(line, key=lambda x: x['left'])
            row_data = [cell['text'] for cell in sorted_cells]
            table_data.append(row_data)
        
        return table_data

class ImageTextExtractor(BaseTextExtractor):
    """Extractor for text from image documents."""
    
    async def extract_text(self, file_path: str, parameters: Dict[str, Any]) -> str:
        """Extract text from image."""
        try:
            # Read image
            image = cv2.imread(file_path)
            confidence_threshold = parameters.get("confidence_threshold", 0.5)
            
            # Use Tesseract OCR with confidence
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Process results
            text_parts = []
            current_line = []
            current_top = None
            
            for i in range(len(ocr_data['text'])):
                if float(ocr_data['conf'][i]) < confidence_threshold:
                    continue
                    
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                top = ocr_data['top'][i]
                
                # New line detection
                if current_top is None:
                    current_top = top
                elif abs(top - current_top) > 5:  # New line if y-coordinate differs significantly
                    text_parts.append(" ".join(current_line))
                    current_line = []
                    current_top = top
                
                current_line.append(text)
            
            # Add last line
            if current_line:
                text_parts.append(" ".join(current_line))
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            raise 