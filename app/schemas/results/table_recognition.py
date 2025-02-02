from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.results.shared import Confidence, PageInfo, BoundingBox

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

class TableStructureResult(BaseModel):
    """Standard output for table structure recognition step."""
    page_info: PageInfo
    tables: List[TableStructure]
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information"
    )

class TableStructureOutput(BaseModel):
    """Complete output for table structure recognition."""
    total_pages_processed: int
    total_tables_processed: int
    results: List[TableStructureResult]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis"
    )