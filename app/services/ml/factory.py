from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging
import torch
from app.schemas.analysis import AnalysisType
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass

class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass

class UnsupportedFormatError(Exception):
    """Raised when document format is not supported."""
    pass

class ValidationError(Exception):
    """Raised when parameter validation fails."""
    pass

class BaseMLFactory(ABC):
    """Base class for ML model factories."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.supported_formats: Dict[str, Dict[str, Any]] = {}
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
        
    def get_supported_parameters(self, document_type: str) -> Dict[str, Any]:
        """Get supported parameters for document type."""
        if document_type not in self.supported_formats:
            logger.error(f"Unsupported document type: {document_type}")
            raise UnsupportedFormatError(f"Document type '{document_type}' is not supported")
        return self.supported_formats[document_type]["parameters"]
        
    def validate_parameters(self, document_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for document type."""
        try:
            # Handle image types (jpg, jpeg, png) using the generic image format
            if document_type.lower() in ["jpg", "jpeg", "png"]:
                document_type = "image"
            
            if document_type not in self.supported_formats:
                raise UnsupportedFormatError(f"Document type '{document_type}' is not supported")
                
            supported_params = self.supported_formats[document_type]["parameters"]
            
            # Add default values for missing parameters
            for param_name, param_spec in supported_params.items():
                if param_name not in parameters and "default" in param_spec:
                    parameters[param_name] = param_spec["default"]
            
            # Validate parameters
            for param_name, param_spec in supported_params.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    
                    # Type checking and conversion
                    if param_spec["type"] == "float":
                        value = float(value)
                        if "min" in param_spec and value < param_spec["min"]:
                            raise ValidationError(f"Parameter '{param_name}' must be >= {param_spec['min']}")
                        if "max" in param_spec and value > param_spec["max"]:
                            raise ValidationError(f"Parameter '{param_name}' must be <= {param_spec['max']}")
                    elif param_spec["type"] == "integer":
                        value = int(value)
                        if "min" in param_spec and value < param_spec["min"]:
                            raise ValidationError(f"Parameter '{param_name}' must be >= {param_spec['min']}")
                        if "max" in param_spec and value > param_spec["max"]:
                            raise ValidationError(f"Parameter '{param_name}' must be <= {param_spec['max']}")
                    elif param_spec["type"] == "boolean":
                        if not isinstance(value, bool):
                            raise ValidationError(f"Parameter '{param_name}' must be a boolean")
                    elif param_spec["type"] == "string":
                        if not isinstance(value, str):
                            raise ValidationError(f"Parameter '{param_name}' must be a string")
                        if "enum" in param_spec and value not in param_spec["enum"]:
                            raise ValidationError(f"Parameter '{param_name}' must be one of: {param_spec['enum']}")
                    
                    parameters[param_name] = value
                    
            return True
            
        except (ValueError, TypeError) as e:
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
    ) -> Dict[str, Any]:
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
                        "type": "integer",
                        "default": 2,
                        "min": 1,
                        "description": "Minimum number of rows to consider as table"
                    },
                    "use_ml_detection": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to use ML-based detection"
                    },
                    "extract_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to extract table data"
                    },
                    "enhance_image": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to enhance images for OCR"
                    }
                }
            },
            "docx": {
                "parameters": {
                    "min_row_count": {
                        "type": "integer",
                        "default": 2,
                        "min": 1,
                        "description": "Minimum number of rows to consider as table"
                    },
                    "extract_structure": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to analyze table structure"
                    },
                    "extract_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to extract table data"
                    }
                }
            },
            "image": {  # Generic image type for all image formats (png, jpg, jpeg)
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for table detection"
                    },
                    "min_row_count": {
                        "type": "integer",
                        "default": 2,
                        "min": 1,
                        "description": "Minimum number of rows to consider as table"
                    },
                    "enhance_image": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to enhance image for better detection"
                    },
                    "extract_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to extract table data"
                    }
                }
            }
        }
    
    def get_description(self) -> str:
        return "Detect and extract tables from documents using advanced computer vision models and document parsing. Supports PDF documents, Word documents, and images (PNG, JPG, JPEG)."

    async def _process_document(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            
            # Process using the detector
            tables = await detector.detect_tables(file_path, parameters)
            return {"tables": tables}
            
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
        self.supported_formats = {
            "pdf": {
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for text extraction"
                    },
                    "extract_layout": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to preserve document layout"
                    },
                    "detect_lists": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to detect and format lists"
                    }
                }
            },
            "png": {
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for text extraction"
                    }
                }
            },
            "jpg": {
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for text extraction"
                    }
                }
            },
            "jpeg": {
                "parameters": {
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Minimum confidence score for text extraction"
                    }
                }
            },
            "docx": {
                "parameters": {
                    "extract_layout": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to preserve document layout"
                    },
                    "detect_lists": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to detect and format lists"
                    }
                }
            }
        }

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
        self.supported_formats = {
            "pdf": {
                "parameters": {
                    "max_length": {
                        "type": "integer",
                        "default": 150,
                        "min": 50,
                        "max": 500,
                        "description": "Maximum length of the summary in words"
                    },
                    "min_length": {
                        "type": "integer",
                        "default": 50,
                        "min": 20,
                        "max": 200,
                        "description": "Minimum length of the summary in words"
                    }
                }
            },
            "docx": {
                "parameters": {
                    "max_length": {
                        "type": "integer",
                        "default": 150,
                        "min": 50,
                        "max": 500,
                        "description": "Maximum length of the summary in words"
                    },
                    "min_length": {
                        "type": "integer",
                        "default": 50,
                        "min": 20,
                        "max": 200,
                        "description": "Minimum length of the summary in words"
                    }
                }
            },
            "txt": {
                "parameters": {
                    "max_length": {
                        "type": "integer",
                        "default": 150,
                        "min": 50,
                        "max": 500,
                        "description": "Maximum length of the summary in words"
                    },
                    "min_length": {
                        "type": "integer",
                        "default": 50,
                        "min": 20,
                        "max": 200,
                        "description": "Minimum length of the summary in words"
                    }
                }
            }
        }

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
        self.supported_formats = {
            "pdf": {
                "parameters": {
                    "target_format": {
                        "type": "string",
                        "default": "docx",
                        "enum": ["pdf", "docx"],
                        "description": "Target format for conversion"
                    },
                    "preserve_styles": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to preserve document styles"
                    }
                }
            },
            "docx": {
                "parameters": {
                    "target_format": {
                        "type": "string",
                        "default": "pdf",
                        "enum": ["pdf", "docx"],
                        "description": "Target format for conversion"
                    },
                    "preserve_styles": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to preserve document styles"
                    }
                }
            }
        }

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