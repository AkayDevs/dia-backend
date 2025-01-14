from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

class BoundingBox(BaseModel):
    """Standard bounding box representation."""
    x1: float = Field(..., description="Left coordinate (normalized 0-1)")
    y1: float = Field(..., description="Top coordinate (normalized 0-1)")
    x2: float = Field(..., description="Right coordinate (normalized 0-1)")
    y2: float = Field(..., description="Bottom coordinate (normalized 0-1)")

    @validator('x1', 'x2', 'y1', 'y2')
    def validate_coordinates(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Coordinates must be normalized between 0 and 1")
        return v

class Confidence(BaseModel):
    """Standard confidence score representation."""
    score: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    method: str = Field(..., description="Method used to calculate confidence")

class PageInfo(BaseModel):
    """Standard page information."""
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    width: int = Field(..., gt=0, description="Page width in pixels")
    height: int = Field(..., gt=0, description="Page height in pixels")

# Table Detection Results
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

# Table Structure Recognition Results
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

# Table Data Extraction Results
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

# Mapping between step types and their output schemas
STEP_OUTPUT_SCHEMAS = {
    "table_detection": TableDetectionOutput,
    "table_structure_recognition": TableStructureOutput,
    "table_data_extraction": TableDataOutput
} 