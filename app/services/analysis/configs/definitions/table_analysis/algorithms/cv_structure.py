from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
from app.services.analysis.configs.base import BaseAlgorithm
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase, AlgorithmParameter, AlgorithmParameterValue
from app.enums.document import DocumentType
from app.schemas.analysis.results.table_structure import (
    TableStructureResult,
    PageTableStructureResult,
    TableStructure,
    Cell
)
from app.schemas.analysis.results.table_shared import BoundingBox, Confidence, PageInfo

class CVTableStructureAlgorithm(BaseAlgorithm):
    """Advanced table structure detection using OpenCV and image processing techniques"""
    
    def get_info(self) -> AlgorithmDefinitionBase:
        return AlgorithmDefinitionBase(
            code="cv_structure",
            name="OpenCV Table Structure Detection",
            version="1.0.0",
            description="Advanced table structure detection using OpenCV and image processing",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOCX],
            parameters=[
                AlgorithmParameter(
                    name="min_line_length",
                    description="Minimum line length as percentage of table dimension",
                    type="float",
                    required=False,
                    default=0.3,
                    constraints={
                        "min": 0.1,
                        "max": 1.0
                    }
                ),
                AlgorithmParameter(
                    name="line_threshold",
                    description="Threshold for line detection",
                    type="integer",
                    required=False,
                    default=30,
                    constraints={
                        "min": 10,
                        "max": 100
                    }
                ),
                AlgorithmParameter(
                    name="header_row_count",
                    description="Number of header rows to detect",
                    type="integer",
                    required=False,
                    default=1,
                    constraints={
                        "min": 0,
                        "max": 5
                    }
                ),
                AlgorithmParameter(
                    name="detect_merged_cells",
                    description="Whether to detect merged cells",
                    type="boolean",
                    required=False,
                    default=True
                )
            ],
            implementation_path="app.services.analysis.configs.definitions.table_analysis.algorithms.cv_structure.CVTableStructureAlgorithm",
            is_active=True
        )

    def get_default_parameters(self) -> List[AlgorithmParameterValue]:
        """Get default parameters for the algorithm"""
        return [
            AlgorithmParameterValue(
                name="min_line_length",
                value=0.3
            ),
            AlgorithmParameterValue(
                name="line_threshold",
                value=30
            ),
            AlgorithmParameterValue(
                name="header_row_count",
                value=1
            ),
            AlgorithmParameterValue(
                name="detect_merged_cells",
                value=True
            )
        ]
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        try:
            import cv2
            import numpy as np
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(f"Required dependency not found: {str(e)}")
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if "document_path" not in input_data:
            raise ValueError("Document path not provided in input data")
        if "tables" not in input_data:
            raise ValueError("Table detection results not provided in input data")
    
    def _detect_lines(
        self,
        image: np.ndarray,
        min_length: float,
        threshold: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect horizontal and vertical lines in the image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Block size
            3    # C constant
        )

        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Calculate kernel sizes based on image dimensions
        h_kernel_length = max(int(image.shape[1] * min_length), 20)
        v_kernel_length = max(int(image.shape[0] * min_length), 20)

        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (h_kernel_length, max(int(image.shape[0] * 0.001), 1))
        )
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        horizontal = cv2.dilate(
            horizontal,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
            iterations=1
        )

        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(int(image.shape[1] * 0.001), 1), v_kernel_length)
        )
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        vertical = cv2.dilate(
            vertical,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
            iterations=1
        )

        return horizontal, vertical

    def _find_intersections(
        self,
        horizontal: np.ndarray,
        vertical: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find intersection points of horizontal and vertical lines"""
        # Get intersection points
        intersections = cv2.bitwise_and(horizontal, vertical)
        
        # Find coordinates of intersection points
        intersection_points = np.column_stack(np.where(intersections > 0))
        
        # Convert to (x,y) format and sort
        points = [(x, y) for y, x in intersection_points]
        points.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
        
        return points

    def _create_grid(
        self,
        intersections: List[Tuple[int, int]],
        tolerance: int = 5
    ) -> Tuple[List[int], List[int]]:
        """Create grid from intersection points"""
        if not intersections:
            return [], []

        # Extract x and y coordinates
        x_coords = sorted(set(x for x, _ in intersections))
        y_coords = sorted(set(y for _, y in intersections))

        # Merge nearby coordinates (within tolerance)
        merged_x = [x_coords[0]]
        merged_y = [y_coords[0]]

        for x in x_coords[1:]:
            if x - merged_x[-1] > tolerance:
                merged_x.append(x)

        for y in y_coords[1:]:
            if y - merged_y[-1] > tolerance:
                merged_y.append(y)

        return merged_x, merged_y

    def _detect_merged_cells(
        self,
        horizontal: np.ndarray,
        vertical: np.ndarray,
        x_coords: List[int],
        y_coords: List[int]
    ) -> List[Dict[str, Any]]:
        """Detect merged cells in the table"""
        merged_cells = []
        
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                # Check for missing internal lines
                cell_region_h = horizontal[
                    y_coords[i]:y_coords[i+1],
                    x_coords[j]:x_coords[j+1]
                ]
                cell_region_v = vertical[
                    y_coords[i]:y_coords[i+1],
                    x_coords[j]:x_coords[j+1]
                ]
                
                # Calculate spans
                row_span = 1
                col_span = 1
                
                # Check horizontal merging
                k = i + 1
                while k < len(y_coords) - 1:
                    if np.sum(horizontal[y_coords[k], x_coords[j]:x_coords[j+1]]) > 0:
                        break
                    row_span += 1
                    k += 1
                
                # Check vertical merging
                k = j + 1
                while k < len(x_coords) - 1:
                    if np.sum(vertical[y_coords[i]:y_coords[i+1], x_coords[k]]) > 0:
                        break
                    col_span += 1
                    k += 1
                
                if row_span > 1 or col_span > 1:
                    merged_cells.append({
                        "row": i,
                        "col": j,
                        "row_span": row_span,
                        "col_span": col_span
                    })
        
        return merged_cells

    def _create_cells(
        self,
        x_coords: List[int],
        y_coords: List[int],
        merged_cells: List[Dict[str, Any]],
        header_rows: int,
        image: np.ndarray
    ) -> List[Cell]:
        """Create cell objects from grid coordinates"""
        cells = []
        processed_cells = set()  # Track cells that are part of merged cells

        # Process merged cells first
        for merged in merged_cells:
            row, col = merged["row"], merged["col"]
            row_span, col_span = merged["row_span"], merged["col_span"]
            
            # Add the merged cell
            cell = Cell(
                bbox=BoundingBox(
                    x1=x_coords[col],
                    y1=y_coords[row],
                    x2=x_coords[col + col_span],
                    y2=y_coords[row + row_span]
                ),
                row_span=row_span,
                col_span=col_span,
                is_header=row < header_rows,
                confidence=Confidence(
                    score=0.9,  # High confidence for detected merged cells
                    method="grid_detection"
                )
            )
            cells.append(cell)
            
            # Mark all constituent cells as processed
            for r in range(row, row + row_span):
                for c in range(col, col + col_span):
                    processed_cells.add((r, c))

        # Process remaining single cells
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                if (i, j) not in processed_cells:
                    cell = Cell(
                        bbox=BoundingBox(
                            x1=x_coords[j],
                            y1=y_coords[i],
                            x2=x_coords[j + 1],
                            y2=y_coords[i + 1]
                        ),
                        row_span=1,
                        col_span=1,
                        is_header=i < header_rows,
                        confidence=Confidence(
                            score=0.95,  # High confidence for regular cells
                            method="grid_detection"
                        )
                    )
                    cells.append(cell)

        return cells

    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table structure detection"""
        try:
            # Get parameters
            min_line_length = parameters.get("min_line_length", 0.3)
            line_threshold = parameters.get("line_threshold", 30)
            header_rows = parameters.get("header_row_count", 1)
            detect_merged = parameters.get("detect_merged_cells", True)
            
            # Extract document pages using document service
            from app.services.documents.document import extract_document_pages
            from app.enums.document import DocumentType

            # Determine document type from file extension
            if document_path.lower().endswith('.pdf'):
                doc_type = DocumentType.PDF
            elif document_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                doc_type = DocumentType.IMAGE
            elif document_path.lower().endswith(('.docx', '.doc')):
                doc_type = DocumentType.DOCX
            else:
                raise ValueError(f"Unsupported document type: {document_path}")

            # Extract user_id and document_id from path
            path_parts = document_path.split('/')
            if len(path_parts) < 3:
                raise ValueError(f"Invalid document path format: {document_path}")
            user_id = path_parts[0]
            document_id = path_parts[1]

            # Get document pages
            document_pages = await extract_document_pages(
                document_path=f"/uploads/{document_path}",
                document_type=doc_type,
                user_id=user_id,
                document_id=document_id
            )
            
            # Get table detection results
            table_results = previous_results.get("table_analysis.table_detection", {})
            if not table_results:
                raise ValueError("No table detection results found in previous steps")
            
            # Process each page
            final_results = []
            total_tables = 0
            
            for page_idx, page_result in enumerate(table_results.get("results", [])):
                # Get corresponding document page
                doc_page = next((p for p in document_pages.pages if p.page_number == page_idx + 1), None)
                if not doc_page:
                    continue

                # Load image from the page's image_url
                image_path = doc_page.image_url.lstrip('/')
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image from {image_path}")

                page_tables = []
                page_info = page_result["page_info"]
                
                for table_loc in page_result["tables"]:
                    # Extract table region
                    bbox = table_loc["bbox"]
                    table_img = image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
                    
                    # Detect lines
                    horizontal, vertical = self._detect_lines(
                        table_img,
                        min_line_length,
                        line_threshold
                    )
                    
                    # Find intersections and create grid
                    intersections = self._find_intersections(horizontal, vertical)
                    x_coords, y_coords = self._create_grid(intersections)
                    
                    if not x_coords or not y_coords:
                        continue
                    
                    # Detect merged cells if enabled
                    merged_cells = []
                    if detect_merged:
                        merged_cells = self._detect_merged_cells(
                            horizontal,
                            vertical,
                            x_coords,
                            y_coords
                        )
                    
                    # Create cells
                    cells = self._create_cells(
                        x_coords,
                        y_coords,
                        merged_cells,
                        header_rows,
                        table_img
                    )
                    
                    # Create table structure
                    table_structure = TableStructure(
                        bbox=BoundingBox(**bbox),
                        cells=cells,
                        num_rows=len(y_coords) - 1,
                        num_cols=len(x_coords) - 1,
                        confidence=Confidence(
                            score=float(table_loc["confidence"]["score"]),
                            method="grid_detection"
                        )
                    )
                    
                    page_tables.append(table_structure)
                    total_tables += 1
                
                # Create page result
                if page_tables:
                    page_result = PageTableStructureResult(
                        page_info=PageInfo(**page_info),
                        tables=page_tables,
                        processing_info={
                            "min_line_length": min_line_length,
                            "line_threshold": line_threshold,
                            "header_rows": header_rows,
                            "merged_cells_detected": detect_merged
                        }
                    )
                    final_results.append(page_result)
            
            # Create final result
            final_result = TableStructureResult(
                results=final_results,
                total_pages_processed=len(final_results),
                total_tables_processed=total_tables,
                metadata={
                    "algorithm": "cv_structure",
                    "version": "1.0.0",
                    "parameters": parameters
                }
            )
            
            return final_result.dict()
            
        except Exception as e:
            raise RuntimeError(f"Table structure detection failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass 