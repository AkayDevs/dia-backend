from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel
from app.schemas.analysis.results.base import BaseResultSchema
from app.schemas.analysis.results.table_shared import Confidence, PageInfo, BoundingBox

class TableLocation(BaseModel):
    """Standard table location information."""
    bbox: BoundingBox
    confidence: Confidence
    table_type: Optional[str] = Field(None, description="Type of table if detected (e.g., 'bordered', 'borderless')")

class PageTableDetectionResult(BaseModel):
    """Results for a single page of table detection"""
    page_info: PageInfo = Field(
        ...,
        description="Information about the processed page"
    )
    tables: List[TableLocation] = Field(
        default_factory=list,
        description="List of detected tables with their locations"
    )
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information for this page"
    )

class TableDetectionResult(BaseResultSchema):
    """Schema for table detection results across all pages"""
    
    schema_info = {
        "name": "table_detection_result",
        "description": "Results from table detection step",
        "version": "1.0.0"
    }
    
    results: List[PageTableDetectionResult] = Field(
        default_factory=list,
        description="List of table detection results for each page"
    )
    total_pages_processed: int = Field(
        default=1,
        description="Total number of pages processed"
    )
    total_tables_found: int = Field(
        default=0,
        description="Total number of tables found across all pages"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the detection process"
    ) 