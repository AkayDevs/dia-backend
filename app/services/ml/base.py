from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.schemas.analysis import (
    TableDetectionParameters,
    TextExtractionParameters,
    TextSummarizationParameters,
    TemplateConversionParameters
)

from app.schemas.analysis import (
    TableDetectionResult,
    TextExtractionResult,
    TextSummarizationResult,
    TemplateConversionResult
)

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
        parameters: TableDetectionParameters
    ) -> TableDetectionResult:
        """
        Detect tables in a document.
        
        Args:
            file_path: Path to the document
            parameters: TableDetectionParameters

        Returns:
            TableDetectionResult containing table information:
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
        parameters: TextExtractionParameters
    ) -> TextExtractionResult:
        """Extract text from a document."""
        pass