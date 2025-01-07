from typing import Dict, Any, List
import fitz  # PyMuPDF
import logging
from pathlib import Path
from .base import BaseTableDetector, BaseTextExtractor
from app.core.config import settings

logger = logging.getLogger(__name__)

class PDFTableDetector(BaseTableDetector):
    """Detector for tables in PDF documents."""
    
    async def detect_tables(self, file_path: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and extract tables from PDF document."""
        try:
            # Convert to absolute path and check existence
            abs_path = Path(settings.UPLOAD_DIR) / file_path.replace("/uploads/", "")
            if not abs_path.exists():
                raise FileNotFoundError(f"Document not found at path: {abs_path}")
                
            doc = fitz.open(abs_path)
            min_row_count = parameters.get("min_row_count", 2)
            
            tables = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get tables using PyMuPDF's table detection
                found_tables = page.find_tables()
                if not found_tables:
                    continue
                    
                # Process each table
                for table_data in found_tables:
                    # table_data is a list where:
                    # table_data[0] is table bbox (x0, y0, x1, y1)
                    # table_data[1] is number of rows
                    # table_data[2] is number of columns
                    # table_data[3] is actual table content
                    if not isinstance(table_data, (list, tuple)) or len(table_data) < 4:
                        continue
                        
                    bbox, n_rows, n_cols, content = table_data
                    
                    # Extract and clean table data
                    cleaned_data = []
                    for row in content:
                        row_data = [cell.strip() if cell else "" for cell in row]
                        if any(cell for cell in row_data):  # Only add non-empty rows
                            cleaned_data.append(row_data)
                    
                    # Skip tables with too few rows
                    if len(cleaned_data) < min_row_count:
                        continue
                    
                    # Create table info
                    table_info = {
                        "page": page_num + 1,
                        "bbox": list(bbox),  # Convert tuple to list
                        "rows": len(cleaned_data),
                        "columns": len(cleaned_data[0]) if cleaned_data else 0,
                        "data": cleaned_data
                    }
                    
                    tables.append(table_info)
            
            doc.close()  # Close the document
            return tables
            
        except FileNotFoundError as e:
            logger.error(f"Document not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error detecting tables in PDF document: {str(e)}")
            raise

class PDFTextExtractor(BaseTextExtractor):
    """Extractor for text from PDF documents."""
    
    async def extract_text(self, file_path: str, parameters: Dict[str, Any]) -> str:
        """Extract text from PDF document."""
        try:
            # Convert to absolute path and check existence
            abs_path = Path(settings.UPLOAD_DIR) / file_path.replace("/uploads/", "")
            if not abs_path.exists():
                raise FileNotFoundError(f"Document not found at path: {abs_path}")
                
            doc = fitz.open(abs_path)
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
            doc.close()  # Close the document
            return separator.join(text_parts)
            
        except FileNotFoundError as e:
            logger.error(f"Document not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF document: {str(e)}")
            raise 