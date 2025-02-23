from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel
from app.schemas.analysis.results.base import BaseResultSchema
from app.schemas.analysis.results.table_shared import Confidence, PageInfo, BoundingBox

class CellContent(BaseModel):
    """Standard cell content information."""
    text: str
    confidence: Confidence
    data_type: Optional[str] = Field(None, description="Detected data type (e.g., 'text', 'number', 'date')")
    normalized_value: Optional[Any] = Field(None, description="Normalized value if applicable")

class TableData(BaseModel):
    """Standard table data information."""
    bbox: BoundingBox
    cells: List[List[CellContent]]  # 2D array representing the table
    confidence: Confidence

class PageTableDataResult(BaseModel):
    """Results for a single page of table data extraction"""
    page_info: PageInfo = Field(
        ...,
        description="Information about the processed page"
    )
    tables: List[TableData] = Field(
        default_factory=list,
        description="List of table data on this page"
    )
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information for this page"
    )

class TableDataResult(BaseResultSchema):
    """Schema for table data results across all pages"""
    
    schema_info = {
        "name": "table_data_result",
        "description": "Results from table data extraction step",
        "version": "1.0.0"
    }
    
    results: List[PageTableDataResult] = Field(
        default_factory=list,
        description="List of table data results for each page"
    )
    total_pages_processed: int = Field(
        default=1,
        description="Total number of pages processed"
    )
    total_tables_processed: int = Field(
        default=0,
        description="Total number of tables processed across all pages"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the data extraction process"
    )
