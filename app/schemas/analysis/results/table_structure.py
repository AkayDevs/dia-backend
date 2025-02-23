from typing import List, Dict, Any
from pydantic import Field, BaseModel
from app.schemas.analysis.results.base import BaseResultSchema
from app.schemas.analysis.results.table_shared import Confidence, PageInfo, BoundingBox

class Cell(BaseModel):
    """Standard cell information."""
    bbox: BoundingBox
    row_span: int = Field(1, ge=1, description="Number of rows this cell spans")
    col_span: int = Field(1, ge=1, description="Number of columns this cell spans")
    is_header: bool = Field(False, description="Whether this cell is a header")
    confidence: Confidence

class TableStructure(BaseModel):
    """Standard table structure information."""
    bbox: BoundingBox
    cells: List[Cell]
    num_rows: int = Field(..., ge=1, description="Number of rows in the table")
    num_cols: int = Field(..., ge=1, description="Number of columns in the table")
    confidence: Confidence

class PageTableStructureResult(BaseModel):
    """Results for a single page of table structure recognition"""
    page_info: PageInfo = Field(
        ...,
        description="Information about the processed page"
    )
    tables: List[TableStructure] = Field(
        default_factory=list,
        description="List of table structures on this page"
    )
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information for this page"
    )

class TableStructureResult(BaseResultSchema):
    """Schema for table structure results across all pages"""
    
    schema_info = {
        "name": "table_structure_result",
        "description": "Results from table structure recognition step",
        "version": "1.0.0"
    }
    
    results: List[PageTableStructureResult] = Field(
        default_factory=list,
        description="List of table structure results for each page"
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
        description="Additional metadata about the structure recognition process"
    )
