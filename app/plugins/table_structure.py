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
    TableStructureOutput,
    TableStructureResult,
    TableStructure,
    Cell,
    BoundingBox,
    Confidence,
    PageInfo
)

logger = logging.getLogger(__name__)

class TableStructureBasic(AnalysisPlugin):
    """Basic table structure recognition using grid detection."""
    
    VERSION = "1.0.0"
    SUPPORTED_DOCUMENT_TYPES = [DocumentType.PDF, DocumentType.IMAGE]
    PARAMETERS = [
        {
            "name": "min_line_length",
            "description": "Minimum line length as percentage of image dimension",
            "type": "float",
            "required": False,
            "default": 0.1,
            "min_value": 0.05,
            "max_value": 0.5
        },
        {
            "name": "line_threshold",
            "description": "Threshold for line detection",
            "type": "integer",
            "required": False,
            "default": 50,
            "min_value": 10,
            "max_value": 100
        },
        {
            "name": "header_row_count",
            "description": "Number of header rows to detect",
            "type": "integer",
            "required": False,
            "default": 1,
            "min_value": 0,
            "max_value": 5
        }
    ]

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters."""
        if "min_line_length" not in parameters:
            parameters["min_line_length"] = 0.1
        if "line_threshold" not in parameters:
            parameters["line_threshold"] = 50
        if "header_row_count" not in parameters:
            parameters["header_row_count"] = 1

        if not (0.05 <= parameters["min_line_length"] <= 0.5):
            raise ValueError("min_line_length must be between 0.05 and 0.5")
        if not (10 <= parameters["line_threshold"] <= 100):
            raise ValueError("line_threshold must be between 10 and 100")
        if not (0 <= parameters["header_row_count"] <= 5):
            raise ValueError("header_row_count must be between 0 and 5")

    def _create_bbox(self, x: int, y: int, w: int, h: int) -> BoundingBox:
        """Create a bounding box from pixel coordinates."""
        return BoundingBox(
            x1=x,
            y1=y,
            x2=x + w,
            y2=y + h
        )

    def _detect_lines(self, image: np.ndarray, min_length: float, threshold: int) -> tuple[np.ndarray, np.ndarray]:
        """Detect horizontal and vertical lines in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[1] * min_length), 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image.shape[0] * min_length)))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        return horizontal_lines, vertical_lines

    def _find_intersections(self, h_lines: np.ndarray, v_lines: np.ndarray) -> List[tuple[int, int]]:
        """Find intersection points of horizontal and vertical lines."""
        intersections = cv2.bitwise_and(h_lines, v_lines)
        return list(zip(*np.where(intersections > 0)))

    def _create_grid(self, intersections: List[tuple[int, int]], tolerance: int = 5) -> tuple[List[int], List[int]]:
        """Create a grid from intersection points."""
        x_coords = sorted(set(x for x, _ in intersections))
        y_coords = sorted(set(y for _, y in intersections))
        
        # Merge nearby coordinates
        merged_x = [x_coords[0]]
        merged_y = [y_coords[0]]
        
        for x in x_coords[1:]:
            if x - merged_x[-1] > tolerance:
                merged_x.append(x)
        
        for y in y_coords[1:]:
            if y - merged_y[-1] > tolerance:
                merged_y.append(y)
                
        return merged_x, merged_y

    def _detect_table_structure(
        self,
        image: np.ndarray,
        table_bbox: BoundingBox,
        parameters: Dict[str, Any]
    ) -> TableStructure:
        """Detect the structure of a table in the image."""
        # Crop the table region
        table_img = image[table_bbox.y1:table_bbox.y2, table_bbox.x1:table_bbox.x2]
        
        # Detect lines
        h_lines, v_lines = self._detect_lines(
            table_img,
            parameters["min_line_length"],
            parameters["line_threshold"]
        )
        
        # Find intersections and create grid
        intersections = self._find_intersections(h_lines, v_lines)
        x_coords, y_coords = self._create_grid(intersections)
        
        # Create cells
        cells = []
        header_rows = parameters["header_row_count"]
        
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                # Basic cell detection (can be improved with span detection)
                cell = Cell(
                    bbox=self._create_bbox(
                        x_coords[j] + table_bbox.x1,
                        y_coords[i] + table_bbox.y1,
                        x_coords[j+1] - x_coords[j],
                        y_coords[i+1] - y_coords[i]
                    ),
                    row_span=1,
                    col_span=1,
                    is_header=i < header_rows,
                    confidence=Confidence(
                        score=0.9,  # Can be improved based on line strength
                        method="grid_detection"
                    )
                )
                cells.append(cell)
        
        return TableStructure(
            bbox=table_bbox,
            cells=cells,
            num_rows=len(y_coords) - 1,
            num_cols=len(x_coords) - 1,
            confidence=Confidence(
                score=0.9,  # Can be improved based on overall grid quality
                method="grid_detection"
            )
        )

    async def execute(self, document_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute table structure recognition on the document."""
        try:
            results = []
            # Convert URL-style path to filesystem path
            relative_path = document_path.replace("/uploads/", "")
            full_path = Path(settings.UPLOAD_DIR) / relative_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Get table locations from previous step
            prev_step_result = parameters.get("previous_step_result", {})
            if not prev_step_result:
                raise ValueError("No table detection results found")
            
            if full_path.suffix.lower() in [".pdf"]:
                doc = fitz.open(str(full_path))
                
                for result in prev_step_result.get("results", []):
                    page_num = result["page_info"]["page_number"]
                    page = doc[page_num - 1]
                    pix = page.get_pixmap()
                    
                    # Convert to numpy array
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_array = np.array(img)
                    
                    # Process each table
                    tables = []
                    for table_loc in result["tables"]:
                        table_structure = self._detect_table_structure(
                            img_array,
                            BoundingBox(**table_loc["bbox"]),
                            parameters
                        )
                        tables.append(table_structure)
                    
                    if tables:
                        results.append(TableStructureResult(
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
                for result in prev_step_result.get("results", []):
                    for table_loc in result["tables"]:
                        table_structure = self._detect_table_structure(
                            img,
                            BoundingBox(**table_loc["bbox"]),
                            parameters
                        )
                        tables.append(table_structure)
                
                if tables:
                    results.append(TableStructureResult(
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
            output = TableStructureOutput(
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
            logger.error(f"Error in table structure recognition: {str(e)}")
            raise 