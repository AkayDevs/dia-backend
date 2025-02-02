from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.schemas.results.shared import Confidence, PageInfo, BoundingBox

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

class TableDataResult(BaseModel):
    """Standard output for table data extraction step."""
    page_info: PageInfo
    tables: List[TableData]
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information"
    )

class TableDataOutput(BaseModel):
    """Complete output for table data extraction."""
    total_pages_processed: int
    total_tables_processed: int
    results: List[TableDataResult]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis"
    )