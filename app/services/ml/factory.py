from typing import Dict, Type, Optional, List
import logging
import mimetypes
from pathlib import Path

from .base import (
    BaseDocumentProcessor,
    BaseTableDetector,
    BaseTextExtractor,
    BaseTextSummarizer,
    BaseTemplateConverter,
    BaseDocumentClassifier,
    BaseEntityExtractor,
    BaseDocumentComparer
)

from .table_detection import ImageTableDetector, PDFTableDetector

logger = logging.getLogger(__name__)

class BaseProcessorFactory:
    """Base factory for creating document processor instances."""
    
    _processors: Dict[str, Type[BaseDocumentProcessor]] = {}
    
    @classmethod
    def get_supported_formats(cls) -> Dict[str, list]:
        """Get all supported formats for each processor type."""
        formats = {}
        for name, processor_cls in cls._processors.items():
            processor = processor_cls()
            formats[name] = processor.supported_formats
        return formats

class TableDetectionFactory(BaseProcessorFactory):
    """Factory for creating table detection instances."""
    
    _processors = {
        "image": ImageTableDetector,
        "pdf": PDFTableDetector,
    }
    
    @classmethod
    def get_detector(cls, file_path: str) -> Optional[BaseTableDetector]:
        """Get appropriate detector for the file type."""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                logger.error(f"Could not determine mime type for file: {file_path}")
                return None
            
            if mime_type.startswith('image/'):
                return cls._processors["image"]()
            elif mime_type == 'application/pdf':
                return cls._processors["pdf"]()
            else:
                logger.error(f"Unsupported file type: {mime_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating detector for file {file_path}: {str(e)}")
            return None

# Placeholder factories for unimplemented services
class TextExtractionFactory(BaseProcessorFactory):
    """Factory for creating text extraction instances."""
    
    @classmethod
    def get_extractor(cls, file_path: str) -> Optional[BaseTextExtractor]:
        """Get appropriate extractor for the file type."""
        logger.warning("Text extraction not implemented yet")
        return None

class TextSummarizationFactory(BaseProcessorFactory):
    """Factory for creating text summarization instances."""
    
    @classmethod
    def get_summarizer(cls, model_type: str = "transformer") -> Optional[BaseTextSummarizer]:
        """Get text summarizer instance."""
        logger.warning("Text summarization not implemented yet")
        return None

class TemplateConversionFactory(BaseProcessorFactory):
    """Factory for creating template conversion instances."""
    
    @classmethod
    def get_converter(cls, source_format: str) -> Optional[BaseTemplateConverter]:
        """Get appropriate converter for the source format."""
        logger.warning("Template conversion not implemented yet")
        return None

class DocumentClassificationFactory(BaseProcessorFactory):
    """Factory for creating document classification instances."""
    
    @classmethod
    def get_classifier(cls) -> Optional[BaseDocumentClassifier]:
        """Get document classifier instance."""
        logger.warning("Document classification not implemented yet")
        return None

class EntityExtractionFactory(BaseProcessorFactory):
    """Factory for creating entity extraction instances."""
    
    @classmethod
    def get_extractor(cls, model_type: str = "spacy") -> Optional[BaseEntityExtractor]:
        """Get entity extractor instance."""
        logger.warning("Entity extraction not implemented yet")
        return None

class DocumentComparisonFactory(BaseProcessorFactory):
    """Factory for creating document comparison instances."""
    
    @classmethod
    def get_comparer(cls, comparison_type: str = "content") -> Optional[BaseDocumentComparer]:
        """Get document comparer instance."""
        logger.warning("Document comparison not implemented yet")
        return None 