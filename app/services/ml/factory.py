from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging
import torch
from pathlib import Path
from app.core.config import settings
from app.exceptions import ModelLoadError, ProcessingError, UnsupportedFormatError, ValidationError
from app.schemas.analysis import (
    AnalysisType,
    TableDetectionParameters,
    TextExtractionParameters,
    TextSummarizationParameters,
    TemplateConversionParameters,
    AnalysisParameters,
    TableDetectionResult,
    TextExtractionResult,
    TextSummarizationResult,
    TemplateConversionResult
)


logger = logging.getLogger(__name__)


class BaseMLFactory(ABC):
    """Base class for ML model factories."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.supported_formats: set[str] = set()
        self.parameter_schema: type[AnalysisParameters] = AnalysisParameters  # Base parameter schema
        logger.info(f"Initialized {self.__class__.__name__} with device: {self.device}")
        
    @abstractmethod
    def get_description(self) -> str:
        """Get description of what this factory's models do."""
        pass

    def load_model(self) -> None:
        """Load the PyTorch model."""
        try:
            # Placeholder for model loading - will be implemented by subclasses
            self.model = None
            logger.info(f"{self.__class__.__name__} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__} model: {str(e)}")
            raise ModelLoadError(f"Failed to load {self.__class__.__name__} model: {str(e)}")
        
    def ensure_model_loaded(self) -> None:
        """Ensure model is loaded before processing."""
        if self.model is None:
            logger.debug(f"Loading model for {self.__class__.__name__}")
            self.load_model()
            
    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to the appropriate device."""
        return data.to(self.device)
        
    def get_supported_parameters(self) -> Dict[str, Any]:
        """Get supported parameters for this factory.
        
        Returns:
            Dict[str, Any]: Dictionary containing parameter definitions with their types,
            default values, constraints, and descriptions.
        """
        if not self.parameter_schema:
            logger.error("No parameter schema defined")
            raise ValueError("No parameter schema defined for this factory")
            
        # Get schema using Pydantic's model_json_schema
        schema = self.parameter_schema.model_json_schema()
        
        # Extract and format parameter information
        return {
            name: {
                "type": prop.get("type"),
                "default": prop.get("default"),
                "description": prop.get("description"),
                **({"min": prop.get("minimum")} if "minimum" in prop else {}),
                **({"max": prop.get("maximum")} if "maximum" in prop else {})
            }
            for name, prop in schema.get("properties", {}).items()
        }
        
    def validate_parameters(self, document_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for document type."""
        try:
            # Handle image types using the generic image format
            if document_type.lower() in ["jpg", "jpeg", "png"]:
                document_type = "image"
            
            # Check if document type is not supported
            if document_type not in self.supported_formats:
                raise UnsupportedFormatError(f"Document type '{document_type}' is not supported")
            
            # Validate using pydantic model
            validated_params = self.parameter_schema(**parameters)
            
            # Update parameters with validated values
            parameters.update(validated_params.model_dump())
            
            return True
            
        except ValidationError as e:
            logger.error(f"Parameter validation error: {str(e)}")
            raise ValidationError(f"Parameter validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during parameter validation: {str(e)}")
            raise

    async def process(
        self,
        file_path: str,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> TableDetectionResult | TextExtractionResult | TextSummarizationResult | TemplateConversionResult:
        """Process the document with progress tracking."""
        try:
            # Ensure model is loaded
            self.ensure_model_loaded()
            
            # Convert to absolute path
            abs_path = Path(settings.UPLOAD_DIR) / file_path.replace("/uploads/", "")
            if not abs_path.exists():
                raise FileNotFoundError(f"Document not found at path: {abs_path}")
            
            # Get document type and handle image mime types
            document_type = Path(file_path).suffix.lower()[1:]
            
            # Handle image types
            if document_type in ["jpg", "jpeg", "png"]:
                # Verify it's actually an image file
                import imghdr
                actual_type = imghdr.what(str(abs_path))
                if actual_type not in ["jpeg", "png"]:
                    raise UnsupportedFormatError(f"Unsupported or invalid image type: {actual_type}")
                # Use generic 'image' type for parameter validation
                document_type = "image"
            
            # Validate parameters for document type
            self.validate_parameters(document_type, parameters)
            
            # Start processing
            if progress_callback:
                await progress_callback(0.0, f"Starting {self.__class__.__name__.lower().replace('factory', '')}")
            
            # Process document
            result = await self._process_document(str(abs_path), parameters)
            
            # Complete processing
            if progress_callback:
                await progress_callback(1.0, f"{self.__class__.__name__.replace('Factory', '')} completed")
            
            return result
            
        except FileNotFoundError as e:
            logger.error(f"Document not found: {str(e)}")
            raise
        except UnsupportedFormatError as e:
            logger.error(f"Unsupported format: {str(e)}")
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__.lower()}: {str(e)}")
            raise ProcessingError(f"{self.__class__.__name__.replace('Factory', '')} failed: {str(e)}")
    
    @abstractmethod
    async def _process_document(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document implementation to be overridden by subclasses."""
        pass


class TableDetectionFactory(BaseMLFactory):
    """Factory for table detection models."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {"pdf", "docx", "image"}  # image type for png, jpg, jpeg
        self.parameter_schema = TableDetectionParameters
        
    def get_description(self) -> str:
        return "Detect and extract tables from documents using advanced computer vision models and document parsing. Supports PDF documents, Word documents, and images (PNG, JPG, JPEG)."

    async def _process_document(
        self,
        file_path: str,
        parameters: TableDetectionParameters
    ) -> TableDetectionResult:
        try:
            # Get file extension without the dot and convert to lowercase
            document_type = Path(file_path).suffix.lower()[1:]
            
            # Log the document type for debugging
            logger.info(f"Processing document type: {document_type}")
            
            # Get appropriate detector based on document type
            if document_type == "docx":
                from .table_detection.word_detector import WordTableDetector
                detector = WordTableDetector()
            elif document_type == "pdf":
                from .table_detection.pdf_detector import PDFTableDetector
                detector = PDFTableDetector()
            elif document_type in ["jpg", "jpeg", "png"]:
                from .table_detection.image_detector import ImageTableDetector
                detector = ImageTableDetector()
            else:
                raise UnsupportedFormatError(f"Unsupported document type: {document_type}")
            
            # Validate file
            if not detector.validate_file(file_path):
                raise ValidationError(f"Invalid or corrupted {document_type.upper()} file")
            
            # Process using the detector with validated parameters
            tables = await detector.detect_tables(file_path, parameters)
            
            # Return the tables directly since filtering is done in detector
            return tables
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except UnsupportedFormatError as e:
            logger.error(f"Format error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            raise ProcessingError(f"Table detection failed: {str(e)}")


class TextExtractionFactory(BaseMLFactory):
    """Factory for text extraction models."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {"pdf", "docx", "png", "jpg", "jpeg"}
        self.parameter_schema = TextExtractionParameters

    def get_description(self) -> str:
        return "Extract text content from documents with layout preservation and structure detection. Supports PDF documents, images, and various text formats."

    async def _process_document(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            document_type = Path(file_path).suffix.lower()[1:]
            
            # Get appropriate detector based on document type
            if document_type == "docx":
                from .table_detection.word_detector import WordTextExtractor
                detector = WordTextExtractor()
            elif document_type == "pdf":
                from .table_detection.pdf_detector import PDFTextExtractor
                detector = PDFTextExtractor()
            elif document_type in ["png", "jpg", "jpeg"]:
                from .table_detection.image_detector import ImageTextExtractor
                detector = ImageTextExtractor()
            else:
                raise UnsupportedFormatError(f"Unsupported document type: {document_type}")
            
            # Process using the detector
            text = await detector.extract_text(file_path, parameters)
            return {"text": text}
            
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            raise ProcessingError(f"Text extraction failed: {str(e)}")


class TextSummarizationFactory(BaseMLFactory):
    """Factory for text summarization models."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {"pdf", "docx", "txt"}
        self.parameter_schema = TextSummarizationParameters

    def get_description(self) -> str:
        return "Generate concise and coherent summaries of text content using advanced language models. Supports PDF documents and text files."

    async def _process_document(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"summary": "Summarized text will appear here"}


class TemplateConversionFactory(BaseMLFactory):
    """Factory for template conversion models."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {"pdf", "docx"}
        self.parameter_schema = TemplateConversionParameters

    def get_description(self) -> str:
        return "Convert documents between different formats while preserving layout and styling. Supports PDF and DOCX formats."

    async def _process_document(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "converted_file": "path/to/converted/file",
            "format": parameters.get("target_format", "pdf")
        }


# Update the factory map with all factories
FACTORY_MAP = {
    AnalysisType.TABLE_DETECTION: TableDetectionFactory,
    AnalysisType.TEXT_EXTRACTION: TextExtractionFactory,
    AnalysisType.TEXT_SUMMARIZATION: TextSummarizationFactory,
    AnalysisType.TEMPLATE_CONVERSION: TemplateConversionFactory
} 