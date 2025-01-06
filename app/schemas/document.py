from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from app.schemas.analysis import AnalysisStatus, AnalysisType
import enum


class DocumentType(str, enum.Enum):
    """Document types supported by the system."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"


class DocumentBase(BaseModel):
    """Base document schema with common attributes."""
    name: str = Field(..., min_length=1, max_length=255, description="Document name")
    type: DocumentType = Field(..., description="Document type")
    size: int = Field(..., gt=0, description="Document size in bytes")
    url: str = Field(..., min_length=1, description="Document storage URL")


class DocumentCreate(DocumentBase):
    """Schema for document creation."""
    class Config:
        @staticmethod
        def schema_extra(schema: dict) -> None:
            schema["example"] = {
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,  # 1MB
                "url": "uploads/example.pdf"
            }


class DocumentUpdate(BaseModel):
    """Schema for document updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[AnalysisStatus] = None
    error_message: Optional[str] = None

    class Config:
        @staticmethod
        def schema_extra(schema: dict) -> None:
            schema["example"] = {
                "name": "updated_name.pdf",
                "status": "completed"
            }


class Document(DocumentBase):
    """Schema for document response."""
    id: str = Field(..., description="Document unique identifier")
    status: AnalysisStatus = Field(..., description="Document analysis status")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    user_id: str = Field(..., description="Owner user ID")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")

    model_config = ConfigDict(from_attributes=True)

    class Config:
        @staticmethod
        def schema_extra(schema: dict) -> None:
            schema["example"] = {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,
                "url": "uploads/example.pdf",
                "status": "completed",
                "uploaded_at": "2024-01-06T12:00:00Z",
                "updated_at": "2024-01-06T12:30:00Z",
                "user_id": "123e4567-e89b-12d3-a456-426614174000"
            }


class AnalysisResultBase(BaseModel):
    """Base schema for analysis results."""
    type: AnalysisType = Field(..., description="Type of analysis performed")
    result: Optional[Dict[str, Any]] = Field(None, description="Analysis results")


class AnalysisResultCreate(AnalysisResultBase):
    """Schema for creating analysis results."""
    document_id: str = Field(..., description="Associated document ID")

    class Config:
        @staticmethod
        def schema_extra(schema: dict) -> None:
            schema["example"] = {
                "type": "text_extraction",
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "result": {
                    "text": "Extracted text content",
                    "pages": 5
                }
            }


class AnalysisResultUpdate(BaseModel):
    """Schema for updating analysis results."""
    result: Dict[str, Any] = Field(..., description="Updated analysis results")


class AnalysisResult(AnalysisResultBase):
    """Schema for analysis result response."""
    id: str = Field(..., description="Result unique identifier")
    document_id: str = Field(..., description="Associated document ID")
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)

    class Config:
        @staticmethod
        def schema_extra(schema: dict) -> None:
            schema["example"] = {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "type": "text_extraction",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "result": {
                    "text": "Extracted text content",
                    "pages": 5
                },
                "created_at": "2024-01-06T12:00:00Z"
            }


class DocumentWithAnalysis(Document):
    """Schema for document with its analysis results."""
    analysis_results: List[AnalysisResult] = Field(
        default_factory=list,
        description="List of analysis results"
    )

    model_config = ConfigDict(from_attributes=True)

    class Config:
        @staticmethod
        def schema_extra(schema: dict) -> None:
            schema["example"] = {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,
                "url": "uploads/example.pdf",
                "status": "completed",
                "uploaded_at": "2024-01-06T12:00:00Z",
                "updated_at": "2024-01-06T12:30:00Z",
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "analysis_results": [
                    {
                        "id": "998e8400-e29b-41d4-a716-446655440000",
                        "type": "text_extraction",
                        "result": {
                            "text": "Extracted text content",
                            "pages": 5
                        },
                        "created_at": "2024-01-06T12:15:00Z"
                    }
                ]
            }


# Analysis Parameters Schemas
class TableDetectionParams(BaseModel):
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    table_type: str = Field("bordered", pattern="^(bordered|borderless|both)$")


class TextExtractionParams(BaseModel):
    include_layout: bool = False
    extract_tables: bool = True


class TextSummarizationParams(BaseModel):
    max_length: int = Field(150, ge=50, le=500)
    min_length: int = Field(50, ge=30, le=100)


class TemplateConversionParams(BaseModel):
    output_format: str = Field("docx", pattern="^(docx|pdf)$")
    preserve_layout: bool = True


class AnalysisParameters(BaseModel):
    type: AnalysisType
    params: Optional[
        TableDetectionParams |
        TextExtractionParams |
        TextSummarizationParams |
        TemplateConversionParams
    ] = None 