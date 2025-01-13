from typing import Optional

class AnalysisError(Exception):
    """Base exception for all analysis-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AnalysisInitializationError(AnalysisError):
    """Raised when there is an error initializing the analysis components."""
    pass

class AnalysisParameterError(AnalysisError):
    """Raised when there is an error with analysis parameters."""
    pass

class AnalysisExecutionError(AnalysisError):
    """Raised when there is an error during analysis execution."""
    pass

class TableDetectionError(AnalysisError):
    """Raised when there is an error during table detection."""
    pass

class TableStructureRecognitionError(AnalysisError):
    """Raised when there is an error during table structure recognition."""
    pass

class TableDataExtractionError(AnalysisError):
    """Raised when there is an error during table data extraction."""
    pass

class ModelLoadingError(AnalysisError):
    """Raised when there is an error loading an ML model."""
    pass

class InvalidAnalysisStateError(AnalysisError):
    """Raised when attempting an operation that is invalid for the current analysis state."""
    pass

class DocumentProcessingError(AnalysisError):
    """Raised when there is an error processing the input document."""
    pass

class UnsupportedAnalysisTypeError(AnalysisError):
    """Raised when an unsupported analysis type is requested."""
    pass

class AnalysisTimeoutError(AnalysisError):
    """Raised when an analysis operation times out."""
    pass

class StepValidationError(AnalysisError):
    """Raised when there is an error validating a step's results or parameters."""
    pass 