

class DocumentNotFoundError(Exception):
    """Exception raised when a document is not found."""
    pass

class AnalysisError(Exception):
    """Base exception for analysis-related errors."""
    pass

class ParameterValidationError(AnalysisError):
    """Exception raised when analysis parameters are invalid."""
    pass

class UnsupportedAnalysisError(AnalysisError):
    """Exception raised when analysis type is not supported for document type."""
    pass

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
