from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from app.schemas.analysis import AnalysisResult
import enum


class DocumentType(str, enum.Enum):
    """Document types supported by the system."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"


class TagBase(BaseModel):
    """Base schema for document tags."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Tag name"
    )


class TagCreate(TagBase):
    """Schema for tag creation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "important"
            }
        }
    )


class Tag(TagBase):
    """Schema for tag response."""
    id: int = Field(..., description="Tag unique identifier")
    created_at: datetime = Field(..., description="Tag creation timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "important",
                "created_at": "2024-01-06T12:00:00Z"
            }
        }
    )


class DocumentBase(BaseModel):
    """Base document schema with common attributes."""
    name: str = Field(..., min_length=1, max_length=255, description="Document name")
    type: DocumentType = Field(..., description="Document type")
    size: int = Field(..., gt=0, description="Document size in bytes")
    url: str = Field(..., min_length=1, description="Document storage URL")


class DocumentCreate(DocumentBase):
    """Schema for document creation."""
    tag_ids: Optional[List[int]] = Field(
        default=None,
        description="List of tag IDs to associate with the document"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,  # 1MB
                "url": "uploads/example.pdf",
                "tag_ids": [1, 2]
            }
        }
    )


class DocumentUpdate(BaseModel):
    """Schema for document updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    tag_ids: Optional[List[int]] = Field(None, description="List of tag IDs to update")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "name": "updated_name.pdf",
                "tag_ids": [1, 3, 4]
            }
        }
    )


class Document(DocumentBase):
    """Schema for document response."""
    id: str = Field(..., description="Document unique identifier")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    user_id: str = Field(..., description="Owner user ID")
    tags: List[Tag] = Field(default_factory=list, description="Document tags")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,
                "url": "uploads/example.pdf",
                "uploaded_at": "2024-01-06T12:00:00Z",
                "updated_at": "2024-01-06T12:30:00Z",
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "tags": [
                    {
                        "id": 1,
                        "name": "important",
                        "created_at": "2024-01-06T12:00:00Z"
                    }
                ]
            }
        }
    )


class DocumentWithAnalysis(Document):
    """Schema for document with its analysis results."""
    analysis_results: List[AnalysisResult] = Field(
        default_factory=list,
        description="List of analysis results"
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,
                "url": "uploads/example.pdf",
                "uploaded_at": "2024-01-06T12:00:00Z",
                "updated_at": "2024-01-06T12:30:00Z",
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "tags": [
                    {
                        "id": 1,
                        "name": "important",
                        "created_at": "2024-01-06T12:00:00Z"
                    }
                ],
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
        }
    )