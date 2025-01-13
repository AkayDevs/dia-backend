from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator, ConfigDict
import enum
from datetime import datetime
from app.enums.analysis import AnalysisMode, AnalysisStatus, AnalysisType


# Step Approval Request ------------------------------------------------------------

class StepApprovalRequest(BaseModel):
    """Request for approving/rejecting a step result."""
    step: enum.Enum
    action: Literal["approve", "reject"]
    feedback: Optional[Dict[str, Any]] = None
    modifications: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": "detection",
                "action": "approve",
                "feedback": {
                    "comment": "Tables detected correctly"
                },
                "modifications": {
                    "remove_tables": [0, 2],
                    "adjust_coordinates": {
                        "1": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
                    }
                }
            }
        }
    )

# Analysis parameters ------------------------------------------------------------

class AnalysisParameters(BaseModel):
    """Base parameters for all analysis types."""
    mode: Optional[AnalysisMode] = Field(
        default=AnalysisMode.AUTOMATIC,
        description="Analysis execution mode"
    )
    step_parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Step-specific parameters for granular analysis"
    )
    confidence_threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results"
    )
    max_results: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of results to return"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "automatic",
                "confidence_threshold": 0.7,
                "max_results": 10,
                "step_parameters": {
                    "detection": {
                        "min_table_size": 100
                    }
                }
            }
        }
    )


# Analysis Result Schema ------------------------------------------------------------

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
    mode: Optional[AnalysisMode] = Field(None, description="Analysis execution mode")
    current_step: Optional[str] = Field(None, description="Current step in granular analysis")
    step_results: Optional[Dict[str, Any]] = Field(None, description="Results for each step in granular analysis")

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# Analysis Request Schema ------------------------------------------------------------

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

    # parameter_model = PARAMETER_MAPPINGS.get(analysis_type)
    # if parameter_model:
    #     return parameter_model(**parameters).model_dump()
    return parameters


class AnalysisRequest(BaseModel):
    """Request to perform analysis on a document."""
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    mode: Optional[AnalysisMode] = Field(default=AnalysisMode.AUTOMATIC, description="Analysis execution mode")
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



# Analysis Result Create Schema ------------------------------------------------------------

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


# Analysis Result Update Schema ------------------------------------------------------------

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