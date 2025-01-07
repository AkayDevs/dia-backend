from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging
import torch
from app.schemas.analysis import AnalysisType
from pathlib import Path

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

class BaseMLFactory(ABC):
    """Base class for ML model factories."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.supported_formats: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    def get_description(self) -> str:
        """Get description of what this factory's models do."""
        pass

    def load_model(self) -> None:
        """Load the PyTorch model."""
        try:
            # Placeholder for model loading - will be implemented by subclasses
            self.model = None
            logger.info(f"{self.__class__.__name__} model loaded")
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__} model: {str(e)}")
            raise ModelLoadError(f"Failed to load {self.__class__.__name__} model: {str(e)}")
        
    def ensure_model_loaded(self) -> None:
        """Ensure model is loaded before processing."""
        if self.model is None:
            self.load_model()
            
    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to the appropriate device."""
        return data.to(self.device)
        
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
                    elif param_spec["type"] == "integer":
                        value = int(value)
                    elif param_spec["type"] == "boolean":
                        if not isinstance(value, bool):
                            return False
                    elif param_spec["type"] == "string":
                        if not isinstance(value, str):
                            return False
                        if "enum" in param_spec and value not in param_spec["enum"]:
                            return False
                        
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
        """Process the document with progress tracking."""
        try:
            self.ensure_model_loaded()
            if progress_callback:
                await progress_callback(0.0, f"Starting {self.__class__.__name__.lower().replace('factory', '')}")
            
            result = await self._process_document(file_path, parameters)
            
            if progress_callback:
                await progress_callback(1.0, f"{self.__class__.__name__.replace('Factory', '')} completed")
            
            return result
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
                    "detect_headers": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to detect and mark table headers"
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
                        "description": "Minimum confidence score for table detection"
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
                        "description": "Minimum confidence score for table detection"
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
                        "description": "Minimum confidence score for table detection"
                    }
                }
            }
        }
    
    def get_description(self) -> str:
        return "Detect and extract tables from documents using advanced computer vision models and document parsing. Supports PDF documents, Word documents, and images."

    async def _process_document(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            document_type = Path(file_path).suffix.lower()[1:]
            
            # Get appropriate detector based on document type
            if document_type == "docx":
                from .detectors.word_detector import WordTableDetector
                detector = WordTableDetector()
            elif document_type == "pdf":
                from .detectors.pdf_detector import PDFTableDetector
                detector = PDFTableDetector()
            else:  # Image formats
                from .detectors.image_detector import ImageTableDetector
                detector = ImageTableDetector()
            
            # Process using the detector
            return await detector.detect(file_path, parameters)
            
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
                from .detectors.word_detector import WordTextExtractor
                detector = WordTextExtractor()
            elif document_type == "pdf":
                from .detectors.pdf_detector import PDFTextExtractor
                detector = PDFTextExtractor()
            else:  # Image formats
                from .detectors.image_detector import ImageTextExtractor
                detector = ImageTextExtractor()
            
            # Process using the detector
            return await detector.detect(file_path, parameters)
            
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