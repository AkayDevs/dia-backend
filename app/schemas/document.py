from typing import Optional, Any, List, Dict, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

from app.enums.document import DocumentType
from app.schemas.analysis import Analysis as AnalysisResponseSchema


# Tag schema ---------------------------------------------------------------

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


# Document schema ---------------------------------------------------------------

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
    previous_version_id: Optional[str] = Field(
        default=None,
        description="ID of the previous version of this document"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "example.pdf",
                "type": "pdf",
                "size": 1048576,  # 1MB
                "url": "uploads/example.pdf",
                "tag_ids": [1, 2],
                "previous_version_id": None
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
    previous_version_id: Optional[str] = Field(None, description="ID of the previous version")
    is_archived: bool = Field(default=False, description="Whether this document is archived")
    archived_at: Optional[datetime] = Field(None, description="When the document was archived")
    retention_until: Optional[datetime] = Field(None, description="Date until which the archived document will be retained")

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
                "previous_version_id": None,
                "is_archived": False,
                "archived_at": None,
                "retention_until": None
            }
        }
    )


class DocumentWithAnalysis(Document):
    """Schema for document with its analysis results."""
    analyses: List[AnalysisResponseSchema] = Field(
        default_factory=list,
        description="List of analyses performed on this document"
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
                "analyses": [
                    {
                        "id": "998e8400-e29b-41d4-a716-446655440000",
                        "type": "table_analysis",
                        "mode": "automatic",
                        "status": "completed",
                        "created_at": "2024-01-06T12:15:00Z",
                        "completed_at": "2024-01-06T12:20:00Z",
                        "step_results": {
                            "table_detection": {
                                "status": "completed",
                                "result": {
                                    "tables_found": 2,
                                    "locations": [
                                        {"page": 1, "bbox": [100, 100, 500, 300]},
                                        {"page": 2, "bbox": [150, 200, 550, 400]}
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        }
    )


class DocumentPage(BaseModel):
    """Schema for a document page."""
    page_number: int = Field(..., ge=1, description="Page number")
    width: int = Field(..., gt=0, description="Page width in pixels")
    height: int = Field(..., gt=0, description="Page height in pixels")
    image_url: str = Field(..., description="URL to the page image")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page_number": 1,
                "width": 800,
                "height": 1200,
                "image_url": "/uploads/123/page_1.png"
            }
        }
    )


class DocumentPages(BaseModel):
    """Schema for document pages response."""
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    pages: List[DocumentPage] = Field(..., description="List of document pages")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_pages": 2,
                "pages": [
                    {
                        "page_number": 1,
                        "width": 800,
                        "height": 1200,
                        "image_url": "/uploads/123/page_1.png"
                    },
                    {
                        "page_number": 2,
                        "width": 800,
                        "height": 1200,
                        "image_url": "/uploads/123/page_2.png"
                    }
                ]
            }
        }
    )