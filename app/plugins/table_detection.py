from typing import Dict, Any, List
import logging
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io

from app.core.analysis import AnalysisPlugin
from app.db.models.document import DocumentType

logger = logging.getLogger(__name__)

class TableDetectionBasic(AnalysisPlugin):
    """Basic table detection using OpenCV contour detection."""
    
    VERSION = "1.0.0"
    SUPPORTED_DOCUMENT_TYPES = [DocumentType.PDF, DocumentType.IMAGE]
    PARAMETERS = [
        {
            "name": "page_range",
            "description": "Range of pages to process (e.g., '1-5' or '1,3,5')",
            "type": "string",
            "required": False,
            "default": "all"
        },
        {
            "name": "max_tables",
            "description": "Maximum number of tables to detect per page",
            "type": "integer",
            "required": False,
            "default": 10,
            "min_value": 1,
            "max_value": 50
        },
        {
            "name": "min_table_size",
            "description": "Minimum table size as percentage of page size",
            "type": "float",
            "required": False,
            "default": 0.05,
            "min_value": 0.01,
            "max_value": 1.0
        }
    ]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters."""
        # Set defaults for missing parameters
        if "max_tables" not in parameters:
            parameters["max_tables"] = 10
        if "min_table_size" not in parameters:
            parameters["min_table_size"] = 0.05
        if "page_range" not in parameters:
            parameters["page_range"] = "all"
            
        # Validate values
        if parameters["max_tables"] < 1 or parameters["max_tables"] > 50:
            raise ValueError("max_tables must be between 1 and 50")
        if parameters["min_table_size"] < 0.01 or parameters["min_table_size"] > 1.0:
            raise ValueError("min_table_size must be between 0.01 and 1.0")
            
        # Validate page range format
        if parameters["page_range"] != "all":
            try:
                pages = []
                for part in parameters["page_range"].split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        pages.extend(range(start, end + 1))
                    else:
                        pages.append(int(part))
                parameters["_parsed_pages"] = sorted(set(pages))
            except:
                raise ValueError("Invalid page_range format")
    
    def _detect_tables_in_image(
        self,
        image: np.ndarray,
        max_tables: int,
        min_table_size: float
    ) -> List[Dict[str, Any]]:
        """Detect tables in an image using contour detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and sort contours
            min_area = image.shape[0] * image.shape[1] * min_table_size
            table_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    table_contours.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": min(0.95, area / (image.shape[0] * image.shape[1]))
                    })
            
            # Sort by area (largest first) and limit number
            table_contours.sort(key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
            return table_contours[:max_tables]
            
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            return []
    
    async def execute(self, document_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute table detection on the document."""
        try:
            results = []
            full_path = Path(document_path)
            
            if not full_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            if full_path.suffix.lower() in [".pdf"]:
                # Process PDF
                doc = fitz.open(full_path)
                
                # Determine pages to process
                if parameters.get("page_range") == "all":
                    pages = range(doc.page_count)
                else:
                    pages = [p - 1 for p in parameters["_parsed_pages"] if p <= doc.page_count]
                
                # Process each page
                for page_num in pages:
                    try:
                        page = doc[page_num]
                        pix = page.get_pixmap()
                        
                        # Convert to numpy array
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_array = np.array(img)
                        
                        # Detect tables
                        tables = self._detect_tables_in_image(
                            img_array,
                            parameters["max_tables"],
                            parameters["min_table_size"]
                        )
                        
                        if tables:
                            results.append({
                                "page": page_num + 1,
                                "tables": tables
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
                        
                doc.close()
                
            else:
                # Process image
                img = cv2.imread(str(full_path))
                if img is None:
                    raise ValueError("Failed to load image")
                
                tables = self._detect_tables_in_image(
                    img,
                    parameters["max_tables"],
                    parameters["min_table_size"]
                )
                
                if tables:
                    results.append({
                        "page": 1,
                        "tables": tables
                    })
            
            return {
                "tables_found": sum(len(r["tables"]) for r in results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            raise

class TableDetectionML(AnalysisPlugin):
    """Table detection using a pre-trained deep learning model."""
    
    VERSION = "1.0.0"
    SUPPORTED_DOCUMENT_TYPES = [DocumentType.PDF, DocumentType.IMAGE]
    PARAMETERS = [
        {
            "name": "page_range",
            "description": "Range of pages to process (e.g., '1-5' or '1,3,5')",
            "type": "string",
            "required": False,
            "default": "all"
        },
        {
            "name": "max_tables",
            "description": "Maximum number of tables to detect per page",
            "type": "integer",
            "required": False,
            "default": 10,
            "min_value": 1,
            "max_value": 50
        },
        {
            "name": "confidence_threshold",
            "description": "Minimum confidence score for table detection",
            "type": "float",
            "required": False,
            "default": 0.5,
            "min_value": 0.1,
            "max_value": 1.0
        }
    ]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters."""
        # Set defaults for missing parameters
        if "max_tables" not in parameters:
            parameters["max_tables"] = 10
        if "confidence_threshold" not in parameters:
            parameters["confidence_threshold"] = 0.5
        if "page_range" not in parameters:
            parameters["page_range"] = "all"
            
        # Validate values
        if parameters["max_tables"] < 1 or parameters["max_tables"] > 50:
            raise ValueError("max_tables must be between 1 and 50")
        if parameters["confidence_threshold"] < 0.1 or parameters["confidence_threshold"] > 1.0:
            raise ValueError("confidence_threshold must be between 0.1 and 1.0")
            
        # Validate page range format
        if parameters["page_range"] != "all":
            try:
                pages = []
                for part in parameters["page_range"].split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        pages.extend(range(start, end + 1))
                    else:
                        pages.append(int(part))
                parameters["_parsed_pages"] = sorted(set(pages))
            except:
                raise ValueError("Invalid page_range format")
    
    async def execute(self, document_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute table detection using ML model."""
        # TODO: Implement ML-based table detection
        # This is a placeholder that returns empty results
        return {
            "tables_found": 0,
            "results": [],
            "note": "ML-based table detection not implemented yet"
        } 