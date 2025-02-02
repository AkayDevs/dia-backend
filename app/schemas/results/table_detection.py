from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from app.schemas.results.shared import Confidence, PageInfo, BoundingBox

class TableLocation(BaseModel):
    """Standard table location information."""
    bbox: BoundingBox
    confidence: Confidence
    table_type: Optional[str] = Field(None, description="Type of table if detected (e.g., 'bordered', 'borderless')")

class TableDetectionResult(BaseModel):
    """Standard output for table detection step."""
    page_info: PageInfo
    tables: List[TableLocation]
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information (e.g., parameters used)"
    )

class TableDetectionOutput(BaseModel):
    """Complete output for table detection."""
    total_pages_processed: int
    total_tables_found: int
    results: List[TableDetectionResult]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis"
    )