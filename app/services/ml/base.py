from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseDocumentProcessor(ABC):
    """Base class for all document processing services."""
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is suitable for processing."""
        pass

class BaseTableDetector(BaseDocumentProcessor):
    """Base class for table detection algorithms."""
    
    @abstractmethod
    def detect_tables(
        self,
        file_path: str,
        confidence_threshold: float = 0.5,
        min_row_count: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Detect tables in a document."""
        pass

class BaseTextExtractor(BaseDocumentProcessor):
    """Base class for text extraction algorithms."""
    
    @abstractmethod
    def extract_text(
        self,
        file_path: str,
        extract_layout: bool = True,
        detect_lists: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract text from a document."""
        pass

class BaseTextSummarizer(BaseDocumentProcessor):
    """Base class for text summarization algorithms."""
    
    @abstractmethod
    def summarize_text(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate summary from text."""
        pass

class BaseTemplateConverter(BaseDocumentProcessor):
    """Base class for template conversion algorithms."""
    
    @abstractmethod
    def convert_template(
        self,
        file_path: str,
        target_format: str,
        preserve_styles: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert document to target format."""
        pass

class BaseDocumentClassifier(BaseDocumentProcessor):
    """Base class for document classification algorithms."""
    
    @abstractmethod
    def classify_document(
        self,
        file_path: str,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Classify document type and content."""
        pass

class BaseEntityExtractor(BaseDocumentProcessor):
    """Base class for entity extraction algorithms."""
    
    @abstractmethod
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract entities from text."""
        pass

class BaseDocumentComparer(BaseDocumentProcessor):
    """Base class for document comparison algorithms."""
    
    @abstractmethod
    def compare_documents(
        self,
        file_path_1: str,
        file_path_2: str,
        comparison_type: str = "content",
        **kwargs
    ) -> Dict[str, Any]:
        """Compare two documents."""
        pass 