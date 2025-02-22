from typing import List, Optional, Dict, Any
from pydantic import Field
from .base import BaseResultSchema

class TextBlock(BaseResultSchema):
    """Schema for a block of text with position"""
    text: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., description="Confidence score of extraction")
    page: int = Field(..., description="Page number")
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class TextExtractionResult(BaseResultSchema):
    """Schema for text extraction step results"""
    
    schema_info = {
        "name": "text_extraction_result",
        "description": "Standard schema for text extraction results",
        "version": "1.0.0"
    }
    
    text_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="List of extracted text blocks"
    )
    language: Optional[str] = Field(None, description="Detected document language")
    total_pages: int = Field(..., description="Total number of pages processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the extraction"
    ) 