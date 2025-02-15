from typing import Dict, Any
import cv2
import numpy as np
from app.analysis.base.base_algorithm import BaseAlgorithm
from app.analysis.registry.components import AlgorithmInfo, AnalysisIdentifier
from app.schemas.analysis import Parameter
from app.enums.document import DocumentType
from app.schemas.results.table_detection import (
    TableDetectionOutput,
    TableDetectionResult,
    TableLocation,
    BoundingBox,
    Confidence,
    PageInfo
)

class BasicTableDetectionAlgorithm(BaseAlgorithm):
    """Basic table detection using OpenCV"""
    
    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            identifier=AnalysisIdentifier(
                name="Basic Table Detection",
                code="basic_detection",
                version="1.0.0"
            ),
            description="Basic table detection using OpenCV contour detection",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            parameters=[
                Parameter(
                    name="max_tables",
                    description="Maximum number of tables to detect per page",
                    type="integer",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=50
                ),
                Parameter(
                    name="min_table_size",
                    description="Minimum table size as percentage of page size",
                    type="float",
                    required=False,
                    default=0.05,
                    min_value=0.01,
                    max_value=1.0
                )
            ],
            implementation_path="app.analysis.types.table_analysis.algorithms.basic_detection.BasicTableDetectionAlgorithm"
        )
    
    async def validate_requirements(self) -> bool:
        """Validate dependencies"""
        try:
            import cv2
            import numpy as np
            return True
        except ImportError:
            return False
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate algorithm parameters"""
        try:
            max_tables = parameters.get("max_tables", 10)
            min_table_size = parameters.get("min_table_size", 0.05)
            
            if not (1 <= max_tables <= 50):
                return False
            if not (0.01 <= min_table_size <= 1.0):
                return False
            
            return True
        except Exception:
            return False
    
    async def execute(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute table detection"""
        try:
            # Get parameters
            max_tables = parameters.get("max_tables", 10)
            min_table_size = parameters.get("min_table_size", 0.05)
            
            # Process each page
            results = []
            total_tables = 0
            
            for page_data in input_data["pages"]:
                # Convert image to grayscale
                image = cv2.imread(page_data["image_path"])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect tables using contours
                tables = self._detect_tables(gray, min_table_size)
                
                # Limit number of tables
                tables = tables[:max_tables]
                total_tables += len(tables)
                
                # Create result for this page
                page_result = TableDetectionResult(
                    page_info=PageInfo(
                        page_number=page_data["page_number"],
                        width=image.shape[1],
                        height=image.shape[0]
                    ),
                    tables=[
                        TableLocation(
                            bbox=BoundingBox(
                                x1=int(bbox[0]),
                                y1=int(bbox[1]),
                                x2=int(bbox[2]),
                                y2=int(bbox[3])
                            ),
                            confidence=Confidence(
                                score=confidence,
                                method="contour_detection"
                            ),
                            table_type="bordered" if is_bordered else "borderless"
                        )
                        for bbox, confidence, is_bordered in tables
                    ],
                    processing_info={
                        "threshold_method": "adaptive",
                        "min_table_size": min_table_size
                    }
                )
                results.append(page_result)
            
            # Create final output
            return TableDetectionOutput(
                total_pages_processed=len(input_data["pages"]),
                total_tables_found=total_tables,
                results=results,
                metadata={
                    "algorithm": "basic_detection",
                    "version": "1.0.0",
                    "parameters": parameters
                }
            ).dict()
            
        except Exception as e:
            raise Exception(f"Table detection failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup temporary resources"""
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
        # Implementation here - this is just a placeholder
        # In a real implementation, you would:
        # 1. Apply adaptive thresholding
        # 2. Find contours
        # 3. Filter contours by size and shape
        # 4. Detect if table is bordered
        # 5. Calculate confidence scores
        return []  # [(bbox, confidence, is_bordered), ...] 