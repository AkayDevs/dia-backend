from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    TABLE_DETECTION = "table_detection"
    TEXT_EXTRACTION = "text_extraction"
    TEXT_SUMMARIZATION = "text_summarization"
    TEMPLATE_CONVERSION = "template_conversion"
    DOCUMENT_CLASSIFICATION = "document_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    DOCUMENT_COMPARISON = "document_comparison"


class AnalysisStatus(str, Enum):
    """Status of an analysis task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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


class TemplateConversionParameters(AnalysisParameters):
    """Parameters specific to template conversion."""
    target_format: str = Field(
        default="docx",
        description="Target format for conversion"
    )
    preserve_styles: bool = Field(
        default=True,
        description="Whether to preserve document styles"
    )


class DocumentClassificationParameters(AnalysisParameters):
    """Parameters specific to document classification."""
    model_type: Optional[str] = Field(
        default=None,
        description="Specific classification model to use"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include document metadata in classification"
    )


class EntityExtractionParameters(AnalysisParameters):
    """Parameters specific to entity extraction."""
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Types of entities to extract"
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include surrounding context for entities"
    )


class DocumentComparisonParameters(AnalysisParameters):
    """Parameters specific to document comparison."""
    comparison_document_id: str = Field(
        ...,
        description="ID of the document to compare against"
    )
    comparison_type: str = Field(
        default="content",
        description="Type of comparison to perform"
    )
    include_visual_diff: bool = Field(
        default=True,
        description="Whether to include visual difference markers"
    )


class AnalysisRequest(BaseModel):
    """Request to perform analysis on a document."""
    document_id: str = Field(..., description="ID of the document to analyze")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis-specific parameters"
    )

    @validator("parameters")
    def validate_parameters(cls, v, values):
        """Validate parameters based on analysis type."""
        analysis_type = values.get("analysis_type")
        if not analysis_type:
            return v

        # Map analysis types to their parameter validators
        parameter_models = {
            AnalysisType.TABLE_DETECTION: TableDetectionParameters,
            AnalysisType.TEXT_EXTRACTION: TextExtractionParameters,
            AnalysisType.TEXT_SUMMARIZATION: TextSummarizationParameters,
            AnalysisType.TEMPLATE_CONVERSION: TemplateConversionParameters,
            AnalysisType.DOCUMENT_CLASSIFICATION: DocumentClassificationParameters,
            AnalysisType.ENTITY_EXTRACTION: EntityExtractionParameters,
            AnalysisType.DOCUMENT_COMPARISON: DocumentComparisonParameters,
        }

        # Validate parameters using the appropriate model
        model = parameter_models.get(analysis_type)
        if model:
            return model(**v).model_dump()
        return v


class AnalysisResult(BaseModel):
    """Base schema for analysis results."""
    id: str
    document_id: str
    analysis_type: AnalysisType
    status: AnalysisStatus
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TableDetectionResult(BaseModel):
    """Result schema for table detection."""
    tables: List[Dict[str, Any]]
    page_numbers: List[int]
    confidence_scores: List[float]


class TextExtractionResult(BaseModel):
    """Result schema for text extraction."""
    text: str
    pages: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class TextSummarizationResult(BaseModel):
    """Result schema for text summarization."""
    summary: str
    original_length: int
    summary_length: int
    key_points: List[str]


class TemplateConversionResult(BaseModel):
    """Result schema for template conversion."""
    converted_file_url: str
    original_format: str
    target_format: str
    conversion_metadata: Dict[str, Any] 