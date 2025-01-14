from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime
import pytesseract
import re
import uuid

from app.core.analysis import AnalysisPlugin
from app.db.models.document import DocumentType
from app.core.config import settings
from app.schemas.analysis_results import (
    TableDataOutput,
    TableDataResult,
    TableData,
    CellContent,
    BoundingBox,
    Confidence,
    PageInfo
)

logger = logging.getLogger(__name__)

class TableDataBasic(AnalysisPlugin):
    """Basic table data extraction using OCR."""
    
    VERSION = "1.0.0"
    SUPPORTED_DOCUMENT_TYPES = [DocumentType.PDF, DocumentType.IMAGE]
    PARAMETERS = [
        {
            "name": "ocr_lang",
            "description": "Language for OCR (e.g., 'eng', 'fra')",
            "type": "string",
            "required": False,
            "default": "eng"
        },
        {
            "name": "data_types",
            "description": "List of data types to detect",
            "type": "array",
            "required": False,
            "default": ["text", "number", "date"]
        },
        {
            "name": "confidence_threshold",
            "description": "Minimum confidence threshold for OCR",
            "type": "float",
            "required": False,
            "default": 0.6,
            "min_value": 0.1,
            "max_value": 1.0
        }
    ]

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters."""
        if "ocr_lang" not in parameters:
            parameters["ocr_lang"] = "eng"
        if "data_types" not in parameters:
            parameters["data_types"] = ["text", "number", "date"]
        if "confidence_threshold" not in parameters:
            parameters["confidence_threshold"] = 0.6

        if not (0.1 <= parameters["confidence_threshold"] <= 1.0):
            raise ValueError("confidence_threshold must be between 0.1 and 1.0")

    def _detect_data_type(
        self,
        text: str,
        data_types: List[str]
    ) -> Tuple[str, Optional[Any]]:
        """Detect the data type of cell content and normalize if possible."""
        text = text.strip()
        if not text:
            return "text", None
            
        if "number" in data_types:
            try:
                # Try to convert to number
                if '.' in text:
                    value = float(text.replace(',', ''))
                    return "number", value
                else:
                    value = int(text.replace(',', ''))
                    return "number", value
            except ValueError:
                pass
                
        if "date" in data_types:
            try:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        value = datetime.strptime(text, fmt)
                        return "date", value.isoformat()
                    except ValueError:
                        continue
            except Exception:
                pass
                
        return "text", text

    def _normalize_value(self, text: str, data_type: str) -> Any:
        """Normalize the value based on its data type."""
        if data_type == "number":
            try:
                return float(text.strip())
            except:
                return None
        elif data_type == "date":
            # Could add date parsing here
            return text.strip()
        return text.strip()

    def _extract_cell_content(
        self,
        image: np.ndarray,
        cell_bbox: BoundingBox,
        parameters: Dict[str, Any]
    ) -> CellContent:
        """Extract content from a cell using OCR."""
        try:
            height, width = image.shape[:2]
            
            # Ensure bbox coordinates are within image boundaries
            x1 = max(0, min(cell_bbox.x1, width - 1))
            y1 = max(0, min(cell_bbox.y1, height - 1))
            x2 = max(0, min(cell_bbox.x2, width))
            y2 = max(0, min(cell_bbox.y2, height))
            
            # Skip if cell area is too small
            if x2 <= x1 or y2 <= y1:
                return CellContent(
                    text="",
                    confidence=Confidence(score=0.0, method="none"),
                    data_type="text",
                    normalized_value=None
                )
            
            # Crop cell region
            cell_img = image[y1:y2, x1:x2]
            
            # Convert to grayscale for OCR
            if len(cell_img.shape) == 3:
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            
            # Save temporary image for OCR
            temp_path = Path(settings.TEMP_DIR) / f"cell_{uuid.uuid4()}.png"
            cv2.imwrite(str(temp_path), cell_img)
            
            try:
                # Perform OCR
                text = pytesseract.image_to_string(
                    str(temp_path),
                    lang=parameters.get("ocr_lang", "eng"),
                    config='--psm 6'  # Assume uniform block of text
                ).strip()
                
                # Get confidence score
                confidence_data = pytesseract.image_to_data(
                    str(temp_path),
                    lang=parameters.get("ocr_lang", "eng"),
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [float(conf) / 100 for conf in confidence_data['conf'] if conf != '-1']
                confidence_score = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Detect data type
                data_type, normalized_value = self._detect_data_type(
                    text,
                    parameters.get("data_types", ["text", "number", "date"])
                )
                
                return CellContent(
                    text=text,
                    confidence=Confidence(
                        score=confidence_score,
                        method="tesseract_ocr"
                    ),
                    data_type=data_type,
                    normalized_value=normalized_value
                )
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            logger.error(f"Error extracting cell content: {str(e)}")
            return CellContent(
                text="",
                confidence=Confidence(score=0.0, method="error"),
                data_type="text",
                normalized_value=None
            )

    def _extract_table_data(
        self,
        image: np.ndarray,
        table_structure: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> TableData:
        """Extract data from a table using its structure."""
        try:
            bbox = BoundingBox(**table_structure["bbox"])
            cells = [BoundingBox(**cell["bbox"]) for cell in table_structure["cells"]]
            num_rows = table_structure["num_rows"]
            num_cols = table_structure["num_cols"]
            
            # Create 2D array for cell contents
            cell_contents = []
            cell_idx = 0
            
            for i in range(num_rows):
                row = []
                for j in range(num_cols):
                    if cell_idx < len(cells):
                        content = self._extract_cell_content(
                            image,
                            cells[cell_idx],
                            parameters
                        )
                        row.append(content)
                        cell_idx += 1
                    else:
                        # Handle missing cells
                        row.append(CellContent(
                            text="",
                            confidence=Confidence(score=0.0, method="none"),
                            data_type="text",
                            normalized_value=None
                        ))
                cell_contents.append(row)
            
            return TableData(
                bbox=bbox,
                cells=cell_contents,
                confidence=Confidence(
                    score=sum(cell.confidence.score for row in cell_contents for cell in row) / (num_rows * num_cols),
                    method="average_cell_confidence"
                )
            )
            
        except Exception as e:
            logger.error(f"Error extracting table data: {str(e)}")
            # Return empty table data with error confidence
            return TableData(
                bbox=bbox,
                cells=[[CellContent(
                    text="",
                    confidence=Confidence(score=0.0, method="error"),
                    data_type="text",
                    normalized_value=None
                )] * num_cols for _ in range(num_rows)],
                confidence=Confidence(score=0.0, method="error")
            )

    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table data extraction on the document."""
        try:
            results = []
            # Convert URL-style path to filesystem path
            relative_path = document_path.replace("/uploads/", "")
            full_path = Path(settings.UPLOAD_DIR) / relative_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Get table structures from previous step
            table_structure_results = previous_results.get("table_structure_recognition", {})
            if not table_structure_results:
                raise ValueError("No table structure results found")
            
            if full_path.suffix.lower() in [".pdf"]:
                doc = fitz.open(str(full_path))
                
                for result in table_structure_results.get("results", []):
                    page_num = result["page_info"]["page_number"]
                    page = doc[page_num - 1]
                    pix = page.get_pixmap()
                    
                    # Convert to numpy array
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_array = np.array(img)
                    
                    # Process each table
                    tables = []
                    for table_struct in result["tables"]:
                        table_data = self._extract_table_data(
                            img_array,
                            table_struct,
                            parameters
                        )
                        tables.append(table_data)
                    
                    if tables:
                        results.append(TableDataResult(
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
                
                doc.close()
                
            else:  # Process image
                img = cv2.imread(str(full_path))
                if img is None:
                    raise ValueError("Failed to load image")
                
                height, width = img.shape[:2]
                tables = []
                
                # Process each table from previous step
                for result in table_structure_results.get("results", []):
                    for table_struct in result["tables"]:
                        table_data = self._extract_table_data(
                            img,
                            table_struct,
                            parameters
                        )
                        tables.append(table_data)
                
                if tables:
                    results.append(TableDataResult(
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
            output = TableDataOutput(
                total_pages_processed=len(results),
                total_tables_processed=sum(len(result.tables) for result in results),
                results=results,
                metadata={
                    "document_type": "pdf" if full_path.suffix.lower() == ".pdf" else "image",
                    "plugin_version": self.VERSION,
                    "processing_timestamp": str(datetime.utcnow())
                }
            )
            
            return output.dict()
            
        except Exception as e:
            logger.error(f"Error in table data extraction: {str(e)}")
            raise 