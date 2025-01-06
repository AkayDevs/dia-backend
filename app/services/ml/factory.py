from typing import Dict, Type, Optional, List, Any, Callable
import logging
import mimetypes
from pathlib import Path
import torch
from abc import ABC, abstractmethod

from .base import (
    BaseDocumentProcessor,
    BaseTableDetector,
    BaseTextExtractor,
    BaseTextSummarizer,
    BaseTemplateConverter
)

from .table_detection import ImageTableDetector, PDFTableDetector
from app.schemas.analysis import AnalysisType, AnalysisStatus

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class UnsupportedFormatError(ProcessingError):
    """Exception raised when file format is not supported."""
    pass


class ModelLoadError(ProcessingError):
    """Exception raised when model fails to load."""
    pass


class BaseMLFactory(ABC):
    """Base class for ML model factories."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the PyTorch model."""
        pass
        
    def ensure_model_loaded(self) -> None:
        """Ensure model is loaded before processing."""
        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise ModelLoadError(f"Failed to load model: {str(e)}")
            
    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to the appropriate device."""
        return data.to(self.device)
        
    @abstractmethod
    def get_supported_parameters(self, document_type: str) -> Dict[str, Any]:
        """Get supported parameters for document type."""
        pass
        
    @abstractmethod
    def validate_parameters(self, document_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for document type."""
        pass
        
    @abstractmethod
    async def process(
        self,
        file_path: str,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Process the document with progress tracking."""
        pass


class ProcessorFactory:
    """Factory for creating document processor instances."""
    
    _processors: Dict[AnalysisType, Type[BaseDocumentProcessor]] = {
        AnalysisType.TABLE_DETECTION: PDFTableDetector,
        AnalysisType.TEXT_EXTRACTION: BaseTextExtractor,
        AnalysisType.TEXT_SUMMARIZATION: BaseTextSummarizer,
        AnalysisType.TEMPLATE_CONVERSION: BaseTemplateConverter
    }
    
    @classmethod
    def get_processor(cls, analysis_type: AnalysisType) -> Optional[BaseDocumentProcessor]:
        """Get processor instance for the specified analysis type."""
        processor_cls = cls._processors.get(analysis_type)
        if processor_cls:
            try:
                return processor_cls()
            except Exception as e:
                logger.error(f"Failed to create processor for {analysis_type}: {str(e)}")
                return None
        return None
    
    @classmethod
    def get_supported_formats(cls, analysis_type: AnalysisType) -> List[str]:
        """Get supported formats for the specified analysis type."""
        processor = cls.get_processor(analysis_type)
        if processor:
            return processor.supported_formats
        return []
    
    @classmethod
    def validate_format(cls, analysis_type: AnalysisType, file_path: str) -> bool:
        """Validate if file format is supported for the analysis type."""
        processor = cls.get_processor(analysis_type)
        if not processor:
            return False
        
        try:
            return processor.validate_file(file_path)
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            return False


class TableDetectionFactory(BaseMLFactory):
    """Factory for table detection models."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {
            "pdf": {
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for table detection"
                    },
                    "min_row_count": {
                        "type": "int",
                        "default": 2,
                        "min": 1,
                        "description": "Minimum number of rows to consider as table"
                    }
                }
            },
            "image": {
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for table detection"
                    }
                }
            }
        }
    
    def load_model(self) -> None:
        """Load table detection PyTorch model."""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                     path='path/to/table_detection_weights.pt')
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load table detection model: {str(e)}")
            raise ModelLoadError(f"Failed to load table detection model: {str(e)}")
            
    def get_supported_parameters(self, document_type: str) -> Dict[str, Any]:
        """Get supported parameters for document type."""
        if document_type not in self.supported_formats:
            raise UnsupportedFormatError(f"Unsupported document type: {document_type}")
        return self.supported_formats[document_type]["parameters"]
        
    def validate_parameters(self, document_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for document type."""
        if document_type not in self.supported_formats:
            return False
            
        supported_params = self.supported_formats[document_type]["parameters"]
        
        try:
            for param_name, param_spec in supported_params.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    
                    # Type checking
                    if param_spec["type"] == "float":
                        value = float(value)
                    elif param_spec["type"] == "int":
                        value = int(value)
                        
                    # Range checking
                    if "min" in param_spec and value < param_spec["min"]:
                        return False
                    if "max" in param_spec and value > param_spec["max"]:
                        return False
                        
            return True
        except (ValueError, TypeError):
            return False
        
    async def process(
        self,
        file_path: str,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Process document for table detection with progress tracking."""
        self.ensure_model_loaded()
        
        try:
            if progress_callback:
                await progress_callback(10, "Loading document")
                
            # Get document type
            document_type = "pdf" if file_path.lower().endswith(".pdf") else "image"
            
            # Validate parameters
            if not self.validate_parameters(document_type, parameters):
                raise ValueError("Invalid parameters for table detection")
                
            if progress_callback:
                await progress_callback(20, "Detecting tables")
                
            # Process based on document type
            if document_type == "pdf":
                detector = PDFTableDetector()
            else:
                detector = ImageTableDetector()
                
            tables = detector.detect_tables(file_path, **parameters)
            
            if progress_callback:
                await progress_callback(90, "Finalizing results")
                
            return {
                "tables": tables,
                "page_numbers": [table.get("page_number", 1) for table in tables],
                "confidence_scores": [table["confidence"] for table in tables]
            }
                
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            raise ProcessingError(f"Table detection failed: {str(e)}")


class TextExtractionFactory(BaseMLFactory):
    """Factory for text extraction models."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {
            "pdf": {
                "parameters": {
                    "extract_layout": {
                        "type": "bool",
                        "default": True,
                        "description": "Whether to preserve document layout"
                    },
                    "detect_lists": {
                        "type": "bool",
                        "default": True,
                        "description": "Whether to detect and preserve lists"
                    }
                }
            },
            "image": {
                "parameters": {
                    "language": {
                        "type": "str",
                        "default": "eng",
                        "description": "Language of the text in image"
                    },
                    "enhance_image": {
                        "type": "bool",
                        "default": True,
                        "description": "Whether to enhance image before OCR"
                    }
                }
            }
        }
    
    def load_model(self) -> None:
        """Load OCR and NLP models."""
        try:
            self.ocr_model = torch.hub.load('your/ocr/model', 'custom', 
                                         path='path/to/ocr_weights.pt')
            self.nlp_model = torch.hub.load('your/nlp/model', 'custom',
                                         path='path/to/nlp_weights.pt')
            self.ocr_model.to(self.device)
            self.nlp_model.to(self.device)
            self.ocr_model.eval()
            self.nlp_model.eval()
        except Exception as e:
            logger.error(f"Failed to load text extraction models: {str(e)}")
            raise ModelLoadError(f"Failed to load text extraction models: {str(e)}")
            
    def get_supported_parameters(self, document_type: str) -> Dict[str, Any]:
        """Get supported parameters for document type."""
        if document_type not in self.supported_formats:
            raise UnsupportedFormatError(f"Unsupported document type: {document_type}")
        return self.supported_formats[document_type]["parameters"]
        
    def validate_parameters(self, document_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for document type."""
        if document_type not in self.supported_formats:
            return False
            
        supported_params = self.supported_formats[document_type]["parameters"]
        
        try:
            for param_name, param_spec in supported_params.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    
                    # Type checking
                    if param_spec["type"] == "bool":
                        value = bool(value)
                    elif param_spec["type"] == "str":
                        value = str(value)
                        
            return True
        except (ValueError, TypeError):
            return False
        
    async def process(
        self,
        file_path: str,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Process document for text extraction with progress tracking."""
        self.ensure_model_loaded()
        
        try:
            if progress_callback:
                await progress_callback(10, "Loading document")
                
            # Get document type
            document_type = "pdf" if file_path.lower().endswith(".pdf") else "image"
            
            # Validate parameters
            if not self.validate_parameters(document_type, parameters):
                raise ValueError("Invalid parameters for text extraction")
                
            if progress_callback:
                await progress_callback(20, "Extracting text")
                
            # TODO: Implement actual text extraction logic
            result = {
                "text": "",
                "pages": [],
                "metadata": {}
            }
            
            if progress_callback:
                await progress_callback(90, "Finalizing results")
                
            return result
                
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise ProcessingError(f"Text extraction failed: {str(e)}")


class TextSummarizationFactory(BaseMLFactory):
    """Factory for text summarization models."""
    
    def load_model(self) -> None:
        """Load summarization PyTorch model."""
        try:
            self.model = torch.hub.load('facebook/bart-large-cnn', 'model')
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load summarization model: {str(e)}")
            raise ModelLoadError(f"Failed to load summarization model: {str(e)}")
        
    async def process(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process text for summarization."""
        self.ensure_model_loaded()
        
        try:
            with torch.no_grad():
                # TODO: Implement actual summarization logic
                results = {
                    "summary": "",
                    "original_length": 0,
                    "summary_length": 0,
                    "key_points": []
                }
                return results
                
        except Exception as e:
            logger.error(f"Text summarization failed: {str(e)}")
            raise ProcessingError(f"Text summarization failed: {str(e)}")


class TemplateConversionFactory(BaseMLFactory):
    """Factory for template conversion models."""
    
    def load_model(self) -> None:
        """Load template conversion PyTorch model."""
        try:
            self.model = torch.hub.load('your/template/model', 'custom',
                                     path='path/to/template_weights.pt')
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load template conversion model: {str(e)}")
            raise ModelLoadError(f"Failed to load template conversion model: {str(e)}")
        
    async def process(self, document_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process document for template conversion."""
        self.ensure_model_loaded()
        
        try:
            with torch.no_grad():
                # TODO: Implement actual template conversion logic
                results = {
                    "converted_file_url": "",
                    "original_format": "",
                    "target_format": parameters.get("target_format", "docx"),
                    "conversion_metadata": {}
                }
                return results
                
        except Exception as e:
            logger.error(f"Template conversion failed: {str(e)}")
            raise ProcessingError(f"Template conversion failed: {str(e)}")



# Factory mapping
FACTORY_MAP = {
    AnalysisType.TABLE_DETECTION: TableDetectionFactory,
    AnalysisType.TEXT_EXTRACTION: TextExtractionFactory,
    AnalysisType.TEXT_SUMMARIZATION: TextSummarizationFactory,
    AnalysisType.TEMPLATE_CONVERSION: TemplateConversionFactory
} 