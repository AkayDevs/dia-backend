from typing import List, Dict, Any
from pydantic import Field
from app.schemas.analysis.results.base import BaseResultSchema

class TableDetectionResult(BaseResultSchema):
    """Schema for table detection results"""
    
    schema_info = {
        "name": "table_detection_result",
        "description": "Results from table detection step",
        "version": "1.0.0"
    }
    
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected tables with their locations"
    )
    page_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the processed page"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the detection process"
    ) 