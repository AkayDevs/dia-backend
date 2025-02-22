from typing import Dict, Any
import cv2
import numpy as np
from app.services.analysis.configs.base import BaseAlgorithm
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionInfo, AlgorithmParameter
from app.enums.document import DocumentType
from app.schemas.analysis.results.table_detection import TableDetectionResult

class BasicTableDetectionAlgorithm(BaseAlgorithm):
    """Basic table detection using OpenCV"""
    
    def get_info(self) -> AlgorithmDefinitionInfo:
        return AlgorithmDefinitionInfo(
            code="basic_detection",
            name="Basic Table Detection",
            version="1.0.0",
            description="Basic table detection using OpenCV contour detection",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            parameters=[
                AlgorithmParameter(
                    name="max_tables",
                    description="Maximum number of tables to detect per page",
                    type="integer",
                    required=False,
                    default=10,
                    constraints={
                        "min": 1,
                        "max": 50
                    }
                ),
                AlgorithmParameter(
                    name="min_table_size",
                    description="Minimum table size as percentage of page size",
                    type="float",
                    required=False,
                    default=0.05,
                    constraints={
                        "min": 0.01,
                        "max": 1.0
                    }
                )
            ],
            implementation_path="app.services.analysis.configs.definitions.table_analysis.algorithms.basic_detection.BasicTableDetectionAlgorithm",
            is_active=True
        )
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Required dependency not found: {str(e)}")
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if "document_path" not in input_data:
            raise ValueError("Document path not provided in input data")
    
    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table detection"""
        try:
            # Get parameters with defaults
            max_tables = parameters.get("max_tables", 10)
            min_table_size = parameters.get("min_table_size", 0.05)
            
            # Process document
            image = cv2.imread(document_path)
            if image is None:
                raise ValueError(f"Could not read image from {document_path}")
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect tables
            tables = self._detect_tables(gray, min_table_size)
            tables = tables[:max_tables]  # Limit number of tables
            
            # Format results
            result = {
                "tables": [
                    {
                        "bbox": {
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3])
                        },
                        "confidence": {
                            "score": float(confidence),
                            "method": "contour_detection"
                        },
                        "type": "bordered" if is_bordered else "borderless"
                    }
                    for bbox, confidence, is_bordered in tables
                ],
                "page_info": {
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "dpi": 300  # Default DPI
                },
                "metadata": {
                    "algorithm": "basic_detection",
                    "version": "1.0.0",
                    "parameters": parameters,
                    "processing_info": {
                        "threshold_method": "adaptive",
                        "min_table_size": min_table_size
                    }
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Table detection failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass
    
    def _detect_tables(
        self,
        gray_image: np.ndarray,
        min_table_size: float
    ) -> list[tuple[list[int], float, bool]]:
        """
        Detect tables in grayscale image
        Returns list of (bbox, confidence, is_bordered) tuples
        """
        try:
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process contours
            tables = []
            img_area = gray_image.shape[0] * gray_image.shape[1]
            
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by size
                if area < img_area * min_table_size:
                    continue
                
                # Calculate confidence based on area and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                area_score = min(area / img_area / 0.5, 1.0)  # Normalize area score
                ratio_score = min(abs(aspect_ratio - 1.5) / 1.5, 1.0)  # Prefer tables with 1.5 aspect ratio
                confidence = (area_score + ratio_score) / 2
                
                # Check if table is bordered
                border_region = thresh[y:y+h, x:x+w]
                border_pixels = cv2.countNonZero(border_region)
                is_bordered = border_pixels > (w + h) * 2  # Simple heuristic
                
                tables.append(([x, y, x+w, y+h], confidence, is_bordered))
            
            # Sort by confidence
            tables.sort(key=lambda x: x[1], reverse=True)
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Table detection processing failed: {str(e)}") 