from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
import enum
from datetime import datetime


class AnalysisStatus(str, enum.Enum):
    """Status of document analysis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisType(str, enum.Enum):
    """Types of analysis supported by the system."""
    TABLE_DETECTION = "table_detection"
    TEXT_EXTRACTION = "text_extraction"
    TEXT_SUMMARIZATION = "text_summarization"
    TEMPLATE_CONVERSION = "template_conversion"


class AnalysisParameters(BaseModel):
    """Base parameters for all analysis types."""
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to consider a detection valid"
    )
    max_results: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of results to return"
    )

    model_config = ConfigDict(extra="allow")


class TableDetectionParameters(AnalysisParameters):
    """Parameters specific to table detection."""
    min_row_count: Optional[int] = Field(
        default=2,
        ge=1,
        description="Minimum number of rows to consider a valid table"
    )
    detect_headers: bool = Field(
        default=True,
        description="Whether to detect table headers"
    )


class TextExtractionParameters(AnalysisParameters):
    """Parameters specific to text extraction."""
    extract_layout: bool = Field(
        default=True,
        description="Whether to preserve document layout"
    )
    detect_lists: bool = Field(
        default=True,
        description="Whether to detect and format lists"
    )


class TextSummarizationParameters(AnalysisParameters):
    """Parameters specific to text summarization."""
    max_length: int = Field(
        default=150,
        ge=50,
        le=500,
        description="Maximum length of the summary in words"
    )
    min_length: int = Field(
        default=50,
        ge=20,
        le=200,
        description="Minimum length of the summary in words"
    )

    @validator("min_length")
    def validate_min_length(cls, v, values):
        if "max_length" in values and v >= values["max_length"]:
            raise ValueError("min_length must be less than max_length")
        return v


class TemplateConversionParameters(AnalysisParameters):
    """Parameters specific to template conversion."""
    target_format: str = Field(
        default="docx",
        pattern="^(docx|pdf)$",
        description="Target format for conversion"
    )
    preserve_styles: bool = Field(
        default=True,
        description="Whether to preserve document styles"
    )



class AnalysisResultBase(BaseModel):
    """Base schema for analysis results."""
    type: AnalysisType = Field(..., description="Type of analysis performed")
    result: Optional[Dict[str, Any]] = Field(None, description="Analysis results")


class AnalysisResult(AnalysisResultBase):
    """Schema for complete analysis result."""
    id: str = Field(..., description="Result unique identifier")
    document_id: str = Field(..., description="ID of the analyzed document")
    status: AnalysisStatus = Field(..., description="Analysis status")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for analysis")
    error: Optional[str] = Field(None, description="Error message if analysis failed")
    created_at: datetime = Field(..., description="When analysis was started")
    completed_at: Optional[datetime] = Field(None, description="When analysis completed")
    progress: float = Field(default=0.0, description="Analysis progress (0.0 to 1.0)")

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


def validate_analysis_parameters(parameters: Dict[str, Any], analysis_type: Optional[AnalysisType]) -> Dict[str, Any]:
    """Validate parameters based on analysis type.
    
    Args:
        parameters: Parameters to validate
        analysis_type: Type of analysis these parameters are for
        
    Returns:
        Dict[str, Any]: Validated and normalized parameters
    """
    if not analysis_type:
        return parameters

    parameter_models = {
        AnalysisType.TABLE_DETECTION: TableDetectionParameters,
        AnalysisType.TEXT_EXTRACTION: TextExtractionParameters,
        AnalysisType.TEXT_SUMMARIZATION: TextSummarizationParameters,
        AnalysisType.TEMPLATE_CONVERSION: TemplateConversionParameters
    }

    model = parameter_models.get(analysis_type)
    if model:
        return model(**parameters).model_dump()
    return parameters


class AnalysisRequest(BaseModel):
    """Request to perform analysis on a document."""
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis-specific parameters"
    )

    @validator("parameters")
    def validate_parameters(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters based on analysis type."""
        return validate_analysis_parameters(v, values.get("analysis_type"))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_type": "text_extraction",
                "parameters": {
                    "confidence_threshold": 0.7,
                    "extract_layout": True,
                    "detect_lists": True
                }
            }
        }
    )


class BatchAnalysisDocument(BaseModel):
    """Single document analysis request in a batch."""
    document_id: str = Field(..., description="ID of the document to analyze")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis-specific parameters"
    )

    @validator("parameters")
    def validate_parameters(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters based on analysis type."""
        return validate_analysis_parameters(v, values.get("analysis_type"))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "analysis_type": "text_extraction",
                "parameters": {
                    "confidence_threshold": 0.7,
                    "extract_layout": True,
                    "detect_lists": True
                }
            }
        }
    )


class BatchAnalysisRequest(BaseModel):
    """Request to perform analysis on multiple documents."""
    documents: List[BatchAnalysisDocument] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of documents to analyze"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "analysis_type": "text_extraction",
                        "parameters": {
                            "confidence_threshold": 0.7,
                            "extract_layout": True
                        }
                    },
                    {
                        "document_id": "660e8400-e29b-41d4-a716-446655440000",
                        "analysis_type": "table_detection",
                        "parameters": {
                            "confidence_threshold": 0.8,
                            "min_row_count": 3
                        }
                    }
                ]
            }
        }
    )


class BatchAnalysisError(BaseModel):
    """Error details for a failed analysis in a batch."""
    document_id: str = Field(..., description="ID of the document that failed")
    error: str = Field(..., description="Error message")


class BatchAnalysisResponse(BaseModel):
    """Response for a batch analysis request."""
    results: List[AnalysisResult] = Field(..., description="Successfully created analysis tasks")
    errors: Optional[List[BatchAnalysisError]] = Field(None, description="Failed analysis requests")
    total_submitted: int = Field(..., description="Total number of successful submissions")
    total_failed: int = Field(..., description="Total number of failed submissions")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_id": "123e4567-e89b-12d3-a456-426614174000",
                        "type": "text_extraction",
                        "status": "pending",
                        "parameters": {
                            "confidence_threshold": 0.7,
                            "extract_layout": True
                        },
                        "created_at": "2024-01-06T12:00:00Z"
                    }
                ],
                "errors": [
                    {
                        "document_id": "660e8400-e29b-41d4-a716-446655440000",
                        "error": "Document not found"
                    }
                ],
                "total_submitted": 1,
                "total_failed": 1
            }
        }
    )


# Type-specific result schemas
class TableDetectionResult(BaseModel):
    """Result schema for table detection."""
    tables: List[Dict[str, Any]] = Field(..., description="Detected tables")
    page_numbers: List[int] = Field(..., description="Pages containing tables")
    confidence_scores: List[float] = Field(..., description="Detection confidence scores")

    model_config = ConfigDict(extra="allow")


class TextExtractionResult(BaseModel):
    """Result schema for text extraction."""
    text: str = Field(..., description="Extracted text content")
    pages: List[Dict[str, Any]] = Field(..., description="Page-wise content")
    metadata: Dict[str, Any] = Field(..., description="Extraction metadata")

    model_config = ConfigDict(extra="allow")


class TextSummarizationResult(BaseModel):
    """Result schema for text summarization."""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text length in words")
    summary_length: int = Field(..., description="Summary length in words")
    key_points: List[str] = Field(..., description="Extracted key points")

    model_config = ConfigDict(extra="allow")


class TemplateConversionResult(BaseModel):
    """Result schema for template conversion."""
    converted_file_url: str = Field(..., description="URL to converted file")
    original_format: str = Field(..., description="Original file format")
    target_format: str = Field(..., description="Target file format")
    conversion_metadata: Dict[str, Any] = Field(..., description="Conversion metadata")

    model_config = ConfigDict(extra="allow")


class AnalysisResultCreate(BaseModel):
    """Schema for creating a new analysis result."""
    id: str
    document_id: str
    type: AnalysisType
    status: AnalysisStatus
    parameters: Dict[str, Any] = Field(default_factory=dict)
    progress: float = Field(default=0.0)
    created_at: datetime
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class AnalysisResultUpdate(BaseModel):
    """Schema for updating an existing analysis result."""
    result: Optional[Dict[str, Any]] = Field(None, description="Updated analysis results")
    status: Optional[AnalysisStatus] = Field(None, description="Updated analysis status")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": {
                    "text": "Updated content",
                    "pages": 6
                },
                "status": "completed"
            }
        }
    ) 