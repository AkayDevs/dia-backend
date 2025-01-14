from typing import Dict, Any, List
import logging
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime

from app.core.analysis import AnalysisPlugin
from app.db.models.document import DocumentType
from app.core.config import settings
from app.schemas.analysis_results import (
    TableDetectionOutput,
    TableDetectionResult,
    TableLocation,
    BoundingBox,
    Confidence,
    PageInfo
)

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
    
    def _create_bbox(self, x: int, y: int, w: int, h: int) -> BoundingBox:
        """Create a bounding box from pixel coordinates."""
        return BoundingBox(
            x1=x,
            y1=y,
            x2=x + w,
            y2=y + h
        )

    def _detect_tables_in_image(
        self,
        image: np.ndarray,
        max_tables: int,
        min_table_size: float
    ) -> List[TableLocation]:
        """Detect tables in an image using contour detection."""
        try:
            height, width = image.shape[:2]
            
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
            table_locations = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence_score = min(0.95, area / (image.shape[0] * image.shape[1]))
                    
                    table_locations.append(TableLocation(
                        bbox=self._create_bbox(x, y, w, h),
                        confidence=Confidence(
                            score=confidence_score,
                            method="contour_area_ratio"
                        ),
                        table_type="bordered" if confidence_score > 0.7 else "borderless"
                    ))
            
            # Sort by confidence and limit number
            table_locations.sort(key=lambda x: x.confidence.score, reverse=True)
            return table_locations[:max_tables]
            
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            return []

    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table detection on the document."""
        try:
            results = []
            # Convert URL-style path to filesystem path
            relative_path = document_path.replace("/uploads/", "")
            full_path = Path(settings.UPLOAD_DIR) / relative_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            if full_path.suffix.lower() in [".pdf"]:
                doc = fitz.open(str(full_path))
                
                # Get pages to process
                if parameters["page_range"] == "all":
                    pages = range(1, doc.page_count + 1)
                else:
                    pages = parameters["_parsed_pages"]
                    # Validate page numbers
                    if any(p > doc.page_count for p in pages):
                        raise ValueError(f"Page number exceeds document length ({doc.page_count} pages)")
                
                # Process each page
                for page_num in pages:
                    try:
                        page = doc[page_num - 1]
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
                            results.append(TableDetectionResult(
                                page_info=PageInfo(
                                    page_number=page_num,
                                    width=pix.width,
                                    height=pix.height
                                ),
                                tables=tables,
                                processing_info={
                                    "parameters": parameters
                                }
                            ))
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        continue
                    
                doc.close()
                
            else:  # Process image
                img = cv2.imread(str(full_path))
                if img is None:
                    raise ValueError("Failed to load image")
                
                height, width = img.shape[:2]
                tables = self._detect_tables_in_image(
                    img,
                    parameters["max_tables"],
                    parameters["min_table_size"]
                )
                
                if tables:
                    results.append(TableDetectionResult(
                        page_info=PageInfo(
                            page_number=1,
                            width=width,
                            height=height
                        ),
                        tables=tables,
                        processing_info={
                            "parameters": parameters
                        }
                    ))
            
            # Create standardized output
            output = TableDetectionOutput(
                total_pages_processed=len(results),
                total_tables_found=sum(len(result.tables) for result in results),
                results=results,
                metadata={
                    "document_type": "pdf" if full_path.suffix.lower() == ".pdf" else "image",
                    "plugin_version": self.VERSION,
                    "processing_timestamp": str(datetime.utcnow())
                }
            )
            
            return output.dict()
            
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