from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDocumentProcessor(ABC):
    """Base class for all document processing services."""
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is suitable for processing."""
        pass

class BaseTableDetector(BaseDocumentProcessor):
    """Base class for table detection algorithms."""
    
    @abstractmethod
    async def detect_tables(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect tables in a document.
        
        Args:
            file_path: Path to the document
            parameters: Dictionary of parameters for detection
                - confidence_threshold: float, minimum confidence for detection
                - min_row_count: int, minimum number of rows to consider as table
                - use_ml_detection: bool, whether to use ML-based detection
                - extract_data: bool, whether to extract table data
                - enhance_image: bool, whether to enhance images for OCR
                
        Returns:
            List of dictionaries containing table information:
            - page: int, page number (1-based)
            - bbox: List[float], bounding box coordinates [x0, y0, x1, y1]
            - rows: int, number of rows
            - columns: int, number of columns
            - data: List[List[str]], table content as 2D array
        """
        pass

class BaseTextExtractor(BaseDocumentProcessor):
    """Base class for text extraction algorithms."""
    
    @abstractmethod
    async def extract_text(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Extract text from a document."""
        pass