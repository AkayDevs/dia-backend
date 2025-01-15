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
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding with refined parameters
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Larger block size for better adaptation
            3
        )

        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Calculate dynamic kernel sizes based on image dimensions
        h_kernel_length = max(int(image.shape[1] * min_length), 20)
        v_kernel_length = max(int(image.shape[0] * min_length), 20)

        # Detect horizontal lines with dynamic parameters
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (h_kernel_length, max(int(image.shape[0] * 0.001), 1))
        )
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(
            horizontal_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
            iterations=1
        )

        # Detect vertical lines with dynamic parameters
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(int(image.shape[1] * 0.001), 1), v_kernel_length)
        )
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(
            vertical_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
            iterations=1
        )

        return horizontal_lines, vertical_lines

    def _find_intersections(self, h_lines: np.ndarray, v_lines: np.ndarray) -> List[tuple[int, int]]:
        """Find intersection points of horizontal and vertical lines."""
        intersections = cv2.bitwise_and(h_lines, v_lines)
        return list(zip(*np.where(intersections > 0)))

    def _create_grid(self, intersections: List[tuple[int, int]], tolerance: int = 5) -> tuple[List[int], List[int]]:
        """Create a grid from intersection points with improved clustering."""
        if not intersections:
            return [], []

        # Separate x and y coordinates
        x_coords = np.array([x for x, _ in intersections])
        y_coords = np.array([y for _, y in intersections])

        # Function to cluster coordinates
        def cluster_coordinates(coords: np.ndarray) -> List[int]:
            if len(coords) == 0:
                return []

            # Sort coordinates
            sorted_coords = np.sort(coords)
            
            # Initialize clusters
            clusters = [[sorted_coords[0]]]
            
            # Cluster coordinates that are within tolerance
            for coord in sorted_coords[1:]:
                if coord - clusters[-1][-1] <= tolerance:
                    clusters[-1].append(coord)
                else:
                    clusters.append([coord])
            
            # Take median of each cluster
            return [int(np.median(cluster)) for cluster in clusters]

        # Cluster coordinates
        merged_x = cluster_coordinates(x_coords)
        merged_y = cluster_coordinates(y_coords)

        return merged_x, merged_y

    def _detect_table_structure(
        self,
        image: np.ndarray,
        table_bbox: BoundingBox,
        parameters: Dict[str, Any]
    ) -> TableStructure:
        """Detect the structure of a table in the image with improved accuracy."""
        # Crop the table region with a small margin
        margin = 5
        y1 = max(table_bbox.y1 - margin, 0)
        y2 = min(table_bbox.y2 + margin, image.shape[0])
        x1 = max(table_bbox.x1 - margin, 0)
        x2 = min(table_bbox.x2 + margin, image.shape[1])
        table_img = image[y1:y2, x1:x2]
        
        # Detect lines
        h_lines, v_lines = self._detect_lines(
            table_img,
            parameters["min_line_length"],
            parameters["line_threshold"]
        )
        
        # Find intersections and create grid
        intersections = self._find_intersections(h_lines, v_lines)
        
        # Adjust intersection coordinates to account for cropping
        adjusted_intersections = [(x + x1, y + y1) for x, y in intersections]
        x_coords, y_coords = self._create_grid(adjusted_intersections)
        
        # Create cells with confidence based on line strength
        cells = []
        header_rows = parameters["header_row_count"]
        
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                # Calculate cell confidence based on line presence
                cell_region_h = h_lines[
                    max(0, y_coords[i] - y1 - 2):min(h_lines.shape[0], y_coords[i+1] - y1 + 2),
                    max(0, x_coords[j] - x1):min(h_lines.shape[1], x_coords[j+1] - x1)
                ]
                cell_region_v = v_lines[
                    max(0, y_coords[i] - y1):min(v_lines.shape[0], y_coords[i+1] - y1),
                    max(0, x_coords[j] - x1 - 2):min(v_lines.shape[1], x_coords[j+1] - x1 + 2)
                ]
                
                # Calculate confidence based on line presence
                h_confidence = np.sum(cell_region_h > 0) / cell_region_h.size
                v_confidence = np.sum(cell_region_v > 0) / cell_region_v.size
                confidence_score = min(0.95, (h_confidence + v_confidence) / 2)
                
                cell = Cell(
                    bbox=self._create_bbox(
                        x_coords[j],
                        y_coords[i],
                        x_coords[j+1] - x_coords[j],
                        y_coords[i+1] - y_coords[i]
                    ),
                    row_span=1,
                    col_span=1,
                    is_header=i < header_rows,
                    confidence=Confidence(
                        score=confidence_score,
                        method="grid_detection"
                    )
                )
                cells.append(cell)
        
        # Calculate overall table confidence
        table_confidence = np.mean([cell.confidence.score for cell in cells])
        
        return TableStructure(
            bbox=table_bbox,
            cells=cells,
            num_rows=len(y_coords) - 1,
            num_cols=len(x_coords) - 1,
            confidence=Confidence(
                score=table_confidence,
                method="grid_detection"
            )
        )

    def _descale_table_structure(self, table: TableStructure, zoom: float) -> TableStructure:
        """Descale a table structure back to original coordinates."""
        descaled_bbox = BoundingBox(
            x1=int(table.bbox.x1 / zoom),
            y1=int(table.bbox.y1 / zoom),
            x2=int(table.bbox.x2 / zoom),
            y2=int(table.bbox.y2 / zoom)
        )
        
        descaled_cells = []
        for cell in table.cells:
            descaled_cell = Cell(
                bbox=BoundingBox(
                    x1=int(cell.bbox.x1 / zoom),
                    y1=int(cell.bbox.y1 / zoom),
                    x2=int(cell.bbox.x2 / zoom),
                    y2=int(cell.bbox.y2 / zoom)
                ),
                row_span=cell.row_span,
                col_span=cell.col_span,
                is_header=cell.is_header,
                confidence=cell.confidence
            )
            descaled_cells.append(descaled_cell)
        
        return TableStructure(
            bbox=descaled_bbox,
            cells=descaled_cells,
            num_rows=table.num_rows,
            num_cols=table.num_cols,
            confidence=table.confidence
        )

    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table structure recognition on the document."""
        try:
            results = []
            # Convert URL-style path to filesystem path
            relative_path = document_path.replace("/uploads/", "")
            full_path = Path(settings.UPLOAD_DIR) / relative_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Get table locations from previous step
            table_detection_results = previous_results.get("table_detection", {})
            if not table_detection_results:
                raise ValueError("No table detection results found")
            
            if full_path.suffix.lower() in [".pdf"]:
                doc = fitz.open(str(full_path))
                
                for result in table_detection_results.get("results", []):
                    page_num = result["page_info"]["page_number"]
                    page = doc[page_num - 1]
                    
                    # Set a consistent DPI for PDF rendering
                    zoom = 2  # Increase resolution for better line detection
                    matrix = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrix)
                    
                    # Convert to numpy array
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_array = np.array(img)
                    
                    # Process each table
                    tables = []
                    for table_loc in result["tables"]:
                        # Scale the input bounding box according to zoom factor
                        scaled_bbox = BoundingBox(
                            x1=int(table_loc["bbox"]["x1"] * zoom),
                            y1=int(table_loc["bbox"]["y1"] * zoom),
                            x2=int(table_loc["bbox"]["x2"] * zoom),
                            y2=int(table_loc["bbox"]["y2"] * zoom)
                        )
                        
                        table_structure = self._detect_table_structure(
                            img_array,
                            scaled_bbox,
                            parameters
                        )
                        
                        # Scale back the detected table structure
                        descaled_structure = self._descale_table_structure(table_structure, zoom)
                        tables.append(descaled_structure)
                    
                    if tables:
                        results.append(TableStructureResult(
                            page_info=PageInfo(
                                page_number=page_num,
                                width=int(pix.width / zoom),
                                height=int(pix.height / zoom)
                            ),
                            tables=tables,
                            processing_info={
                                "parameters": parameters,
                                "dpi_scale": zoom
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
                for result in table_detection_results.get("results", []):
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