from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class TagBase(BaseModel):
    """Base tag schema with common attributes."""
    name: str = Field(..., min_length=1, max_length=50, description="Tag name")


class TagCreate(TagBase):
    """Schema for tag creation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "important"
            }
        }
    )


class TagUpdate(BaseModel):
    """Schema for tag updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=50)

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "name": "very-important"
            }
        }
    )


class Tag(TagBase):
    """Schema for tag response."""
    id: str = Field(..., description="Tag unique identifier")
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "important",
                "created_at": "2024-01-06T12:00:00Z"
            }
        }
    )


class DocumentTags(BaseModel):
    """Schema for updating document tags."""
    tag_ids: List[str] = Field(..., description="List of tag IDs to assign to document")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tag_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "660e8400-e29b-41d4-a716-446655440000"
                ]
            }
        }
    ) 