from typing import Dict, Any, List
import fitz  # PyMuPDF
import logging
from .base import BaseTableDetector, BaseTextExtractor

logger = logging.getLogger(__name__)

class PDFTableDetector(BaseTableDetector):
    """Detector for tables in PDF documents."""
    
    async def detect_tables(self, file_path: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and extract tables from PDF document."""
        try:
            doc = fitz.open(file_path)
            confidence_threshold = parameters.get("confidence_threshold", 0.5)
            min_row_count = parameters.get("min_row_count", 2)
            
            tables = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get tables using PyMuPDF's table detection
                table_finder = page.find_tables()
                for table in table_finder:
                    if table.confidence < confidence_threshold:
                        continue
                        
                    if len(table.rows) < min_row_count:
                        continue
                    
                    # Extract table data
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row]
                        table_data.append(row_data)
                    
                    # Create table info
                    table_info = {
                        "page": page_num + 1,
                        "bbox": list(table.bbox),  # Convert rect to list
                        "confidence": table.confidence,
                        "rows": len(table.rows),
                        "columns": len(table.cols),
                        "data": table_data
                    }
                    
                    tables.append(table_info)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error detecting tables in PDF document: {str(e)}")
            raise

class PDFTextExtractor(BaseTextExtractor):
    """Extractor for text from PDF documents."""
    
    async def extract_text(self, file_path: str, parameters: Dict[str, Any]) -> str:
        """Extract text from PDF document."""
        try:
            doc = fitz.open(file_path)
            confidence_threshold = parameters.get("confidence_threshold", 0.5)
            extract_layout = parameters.get("extract_layout", True)
            detect_lists = parameters.get("detect_lists", True)
            
            text_parts = []
            
            for page in doc:
                if extract_layout:
                    # Extract text blocks preserving layout
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[5] > confidence_threshold:  # Check text confidence
                            text = block[4].strip()
                            if text:
                                # Detect bullet points if enabled
                                if detect_lists and (text.startswith("•") or text.startswith("-")):
                                    text_parts.append(f"• {text[1:].strip()}")
                                else:
                                    text_parts.append(text)
                else:
                    # Simple text extraction
                    text = page.get_text("text")
                    if text.strip():
                        text_parts.append(text.strip())
            
            # Join with appropriate separator
            separator = "\n" if extract_layout else " "
            return separator.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF document: {str(e)}")
            raise 