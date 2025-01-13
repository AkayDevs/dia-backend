from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field, validator, ConfigDict
from app.enums.table_analysis import TableDetectionAlgorithm, TableStructureRecognitionAlgorithm, TableDataExtractionAlgorithm
from app.schemas.analysis import AnalysisParameters
from app.enums.analysis import AnalysisStatus
from datetime import datetime
import json

# Table Detection Parameters -----------------------------------------------------

class BaseTableDetectionParameters(BaseModel):
    """
    Base parameters for table detection.
    These parameters are common across all table detection algorithms.
    """
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to consider a table detection valid."
    )
    max_tables: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of tables to detect per document."
    )
    min_table_size: Optional[float] = Field(
        default=0.1,
        gt=0.0,
        description="Minimum table size as a fraction of the page size."
    )
    page_range: Optional[str] = Field(
        default="all",
        description="Pages to include in the analysis (e.g., '1-3,5,7-10')."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confidence_threshold": 0.6,
                "max_tables": 5,
                "min_table_size": 0.15,
                "page_range": "1-2,4"
            }
        }
    )

class MSATableDetectionParameters(BaseTableDetectionParameters):
    """
    Parameters specific to the Microsoft Table Transformer (MSA) algorithm.
    """
    model_version: Optional[str] = Field(
        default="latest",
        description="Version of the Microsoft Table Transformer model to use."
    )
    refine_edges: bool = Field(
        default=True,
        description="Whether to refine table edges for more precise detection."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confidence_threshold": 0.7,
                "max_tables": 3,
                "min_table_size": 0.2,
                "page_range": "1-5",
                "model_version": "v1.2",
                "refine_edges": False
            }
        }
    )

class CustomTableDetectionParameters(BaseTableDetectionParameters):
    """
    Parameters specific to a custom table detection algorithm.
    """
    resize_factor: float = Field(
        default=1.0,
        gt=0.0,
        description="Factor by which to resize the document images before detection."
    )
    use_grayscale: bool = Field(
        default=True,
        description="Whether to convert images to grayscale before processing."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confidence_threshold": 0.55,
                "max_tables": 4,
                "min_table_size": 0.1,
                "page_range": "all",
                "resize_factor": 0.8,
                "use_grayscale": True
            }
        }
    )

class YOLOTableDetectionParameters(BaseTableDetectionParameters):
    """
    Parameters specific to the YOLO-based table detection algorithm.
    """
    model_path: str = Field(
        ...,
        description="Path to the pre-trained YOLO model weights."
    )
    iou_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Intersection over Union (IoU) threshold for non-max suppression."
    )
    class_labels: Optional[list] = Field(
        default=["table"],
        description="List of class labels to detect."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confidence_threshold": 0.65,
                "max_tables": 6,
                "min_table_size": 0.12,
                "page_range": "2-6",
                "model_path": "/models/yolo_table_detector.pt",
                "iou_threshold": 0.5,
                "class_labels": ["table", "chart"]
            }
        }
    )

class TableAnalysisDetectionParameters(BaseModel):
    """
    Parameters for the table detection step.
    Allows users to select the detection algorithm and provide corresponding parameters.
    """
    algorithm: TableDetectionAlgorithm = Field(
        ...,
        description="Algorithm to use for table detection."
    )
    parameters: Union[
        MSATableDetectionParameters,
        CustomTableDetectionParameters,
        YOLOTableDetectionParameters
    ] = Field(
        ...,
        description="Parameters specific to the selected table detection algorithm."
    )

    @validator("parameters")
    def validate_parameters(cls, v, values):
        algorithm = values.get("algorithm")
        if algorithm == TableDetectionAlgorithm.MSA and not isinstance(v, MSATableDetectionParameters):
            raise ValueError("Parameters must be of type MSATableDetectionParameters for MSA algorithm.")
        elif algorithm == TableDetectionAlgorithm.CUSTOM and not isinstance(v, CustomTableDetectionParameters):
            raise ValueError("Parameters must be of type CustomTableDetectionParameters for Custom algorithm.")
        elif algorithm == TableDetectionAlgorithm.YOLO and not isinstance(v, YOLOTableDetectionParameters):
            raise ValueError("Parameters must be of type YOLOTableDetectionParameters for YOLO algorithm.")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "algorithm": "msa",
                "parameters": {
                    "confidence_threshold": 0.7,
                    "max_tables": 3,
                    "min_table_size": 0.2,
                    "page_range": "1-5",
                    "model_version": "v1.2",
                    "refine_edges": False
                }
            }
        }
    )

# Table Structure Recognition Parameters -----------------------------------------

class BaseTableStructureRecognitionParameters(BaseModel):
    """Base parameters for table structure recognition."""
    max_iterations: Optional[int] = Field(
        default=10,
        ge=1,
        description="Maximum number of iterations to perform for structure recognition."
    )

class MSATableStructureRecognitionParameters(BaseTableStructureRecognitionParameters):
    """Parameters for table structure recognition using Microsoft Table Transformer (MSA)."""
    model_version: Optional[str] = Field(
        default="latest",
        description="Version of the Microsoft Table Transformer model to use."
    )
    refine_boundaries: bool = Field(
        default=True,
        description="Whether to refine table boundaries for more precise detection. May be slower."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_iterations": 15,
                "refine_boundaries": True
            }
        }
    )

class CustomTableStructureRecognitionParameters(BaseTableStructureRecognitionParameters):
    """Parameters for table structure recognition using a custom algorithm."""
    grid_size: float = Field(
        default=0.1,
        gt=0.0,
        description="Size of the grid cells to use for structure recognition."
    )
    use_smoothing: bool = Field(
        default=True,
        description="Whether to apply smoothing to the structure recognition results. May be slower."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_iterations": 15,
                "grid_size": 0.05,
                "use_smoothing": True
            }
        }
    )

class YOLOTableStructureRecognitionParameters(BaseTableStructureRecognitionParameters):
    """Parameters for table structure recognition using YOLO."""
    iou_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Intersection over Union (IoU) threshold for non-max suppression."
    )
    class_labels: Optional[list] = Field(
        default=["header", "row", "column", "cell"],
        description="List of class labels to detect."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_iterations": 15,
                "iou_threshold": 0.5,
                "class_labels": ["header", "row", "column", "cell"]
            }
        }
    )

class TableAnalysisStructureRecognitionParameters(BaseModel):
    """Parameters for the table structure recognition step."""
    algorithm: TableStructureRecognitionAlgorithm = Field(
        ...,
        description="Algorithm to use for table structure recognition."
    )
    parameters: Union[
        MSATableStructureRecognitionParameters,
        CustomTableStructureRecognitionParameters,
        YOLOTableStructureRecognitionParameters
    ] = Field(
        ...,
        description="Parameters specific to the selected structure recognition algorithm."
    )

    @validator("parameters")
    def validate_parameters(cls, v, values):
        algorithm = values.get("algorithm")
        if algorithm == TableStructureRecognitionAlgorithm.MSA and not isinstance(v, MSATableStructureRecognitionParameters):
            raise ValueError("Parameters must be of type MSATableStructureRecognitionParameters for MSA algorithm.")
        elif algorithm == TableStructureRecognitionAlgorithm.CUSTOM and not isinstance(v, CustomTableStructureRecognitionParameters):
            raise ValueError("Parameters must be of type CustomTableStructureRecognitionParameters for Custom algorithm.")
        elif algorithm == TableStructureRecognitionAlgorithm.YOLO and not isinstance(v, YOLOTableStructureRecognitionParameters):
            raise ValueError("Parameters must be of type YOLOTableStructureRecognitionParameters for YOLO algorithm.")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "algorithm": "msa",
                "parameters": {
                    "max_iterations": 15,
                    "refine_boundaries": True
                }
            }
        }
    )


# Table Data Extraction Parameters ----------------------------------------------

class BaseTableDataExtractionParameters(BaseModel):
    """Base parameters for table data extraction."""
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to consider a table cell valid."
    )
    max_attempts: Optional[int] = Field(
        default=10,
        ge=1,
        description="Maximum number of attempts to extract data from a table cell."
    )

    class Config:
        schema_extra = {
            "example": {
                "confidence_threshold": 0.7,
                "max_attempts": 15
            }
        }

class MSATableDataExtractionParameters(BaseTableDataExtractionParameters):
    """Parameters for table data extraction using Microsoft Table Transformer (MSA)."""
    model_version: Optional[str] = Field(
        default="latest",
        description="Version of the Microsoft Table Transformer model to use."
    )
    preprocess_image: bool = Field(
        default=True,
        description="Whether to preprocess the image before data extraction."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "v1.2",
                "preprocess_image": True,
                "max_attempts": 2
            }
        }
    )

class CustomTableDataExtractionParameters(BaseTableDataExtractionParameters):
    """Parameters for table data extraction using a custom algorithm."""
    use_spell_check: bool = Field(
        default=True,
        description="Whether to apply spell checking to extracted text."
    )
    language: str = Field(
        default="en",
        description="Language code for OCR and spell checking."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "use_spell_check": True,
                "language": "en"
            }
        }
    )

class YOLOTableDataExtractionParameters(BaseTableDataExtractionParameters):
    """Parameters for table data extraction using YOLO."""
    iou_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Intersection over Union (IoU) threshold for non-max suppression."
    )
    class_labels: Optional[list] = Field(
        default=["text"],
        description="List of class labels to detect."
    )

    class Config:
        schema_extra = {
            "example": {
                "confidence_threshold": 0.7,
                "max_attempts": 15,
                "iou_threshold": 0.5,
                "class_labels": ["text"]
            }
        }

class TableAnalysisDataExtractionParameters(BaseModel):
    """Parameters for the table data extraction step."""
    algorithm: TableDataExtractionAlgorithm = Field(
        ...,
        description="Algorithm to use for table data extraction."
    )
    parameters: Union[
        MSATableDataExtractionParameters,
        CustomTableDataExtractionParameters
    ] = Field(
        ...,
        description="Parameters specific to the selected data extraction algorithm."
    )

    @validator("parameters")
    def validate_parameters(cls, v, values):
        algorithm = values.get("algorithm")
        if algorithm == TableDataExtractionAlgorithm.MSA and not isinstance(v, MSATableDataExtractionParameters):
            raise ValueError("Parameters must be of type MSATableDataExtractionParameters for MSA algorithm.")
        elif algorithm == TableDataExtractionAlgorithm.CUSTOM and not isinstance(v, CustomTableDataExtractionParameters):
            raise ValueError("Parameters must be of type CustomTableDataExtractionParameters for Custom algorithm.")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "algorithm": "msa",
                "parameters": {
                    "model_version": "v1.2",
                    "preprocess_image": True,
                    "max_attempts": 2
                }
            }
        }
    )

# Table Analysis Parameters -----------------------------------------------------

class TableAnalysisParameters(AnalysisParameters):
    """Parameters for table analysis."""
    detection: Optional[TableAnalysisDetectionParameters] = Field(
        None,
        description="Parameters for table detection step."
    )
    structure_recognition: Optional[TableAnalysisStructureRecognitionParameters] = Field(
        None,
        description="Parameters for table structure recognition step."
    )
    data_extraction: Optional[TableAnalysisDataExtractionParameters] = Field(
        None,
        description="Parameters for table data extraction step."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "step_by_step",
                "confidence_threshold": 0.7,
                "detection": {
                    "algorithm": "msa",
                    "parameters": {
                        "confidence_threshold": 0.7,
                        "max_tables": 3,
                        "min_table_size": 0.2,
                        "page_range": "1-5",
                        "model_version": "v1.2",
                        "refine_edges": False
                    }
                }
            }
        }
    )


# Table Analysis Result Helper Classes -----------------------------------------

class BoundingBox(BaseModel):
    """
    Represents the coordinates of a bounding box around a detected table.
    """
    x: float = Field(..., description="The x-coordinate of the top-left corner.")
    y: float = Field(..., description="The y-coordinate of the top-left corner.")
    width: float = Field(..., gt=0, description="The width of the bounding box.")
    height: float = Field(..., gt=0, description="The height of the bounding box.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x": 100.0,
                "y": 200.0,
                "width": 300.0,
                "height": 150.0
            }
        }
    )

class DetectedTable(BaseModel):
    """Detected table information."""
    table_id: str = Field(..., description="Unique identifier for the table")
    bounding_box: BoundingBox = Field(..., description="Table bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "table_id": "table_1_1",
                "bounding_box": {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 300.0,
                    "height": 150.0
                },
                "confidence": 0.95
            }
        }
    )

class Cell(BaseModel):
    """Cell information in a table."""
    row: int = Field(..., ge=1, description="Row number (1-based)")
    column: int = Field(..., ge=1, description="Column number (1-based)")
    rowspan: int = Field(default=1, ge=1, description="Number of rows this cell spans")
    colspan: int = Field(default=1, ge=1, description="Number of columns this cell spans")
    bounding_box: BoundingBox = Field(..., description="Cell bounding box")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "row": 1,
                "column": 1,
                "rowspan": 1,
                "colspan": 1,
                "bounding_box": {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 100.0,
                    "height": 50.0
                }
            }
        }
    )

class RecognizedTableStructure(BaseModel):
    """Recognized table structure."""
    table_id: str = Field(..., description="Unique identifier for the table")
    rows: int = Field(..., gt=0, description="Number of rows")
    columns: int = Field(..., gt=0, description="Number of columns")
    cells: List[Cell] = Field(..., description="List of cells in the table")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Structure recognition confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "table_id": "table_1_1",
                "rows": 3,
                "columns": 4,
                "cells": [
                    {
                        "row": 1,
                        "column": 1,
                        "rowspan": 1,
                        "colspan": 1,
                        "bounding_box": {
                            "x": 100.0,
                            "y": 200.0,
                            "width": 100.0,
                            "height": 50.0
                        }
                    }
                ],
                "confidence": 0.92
            }
        }
    )

class ExtractedCell(BaseModel):
    """Extracted cell data."""
    row: int = Field(..., ge=1, description="Row number (1-based)")
    column: int = Field(..., ge=1, description="Column number (1-based)")
    data: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "row": 1,
                "column": 1,
                "data": "Sample text",
                "confidence": 0.95
            }
        }
    )

class ExtractedTableData(BaseModel):
    """Extracted table data."""
    table_id: str = Field(..., description="Unique identifier for the table")
    cells: List[ExtractedCell] = Field(..., description="List of extracted cells")
    confidence: float = Field(..., description="Overall confidence score for the extraction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "table_id": "table_1_1",
                "cells": [
                    {
                        "row": 1,
                        "column": 1,
                        "data": "Sample text",
                        "confidence": 0.95
                    }
                ],
                "confidence": 0.92
            }
        }
    )


# Table Analysis Result Schema --------------------------------------------------


class BaseTableAnalysisStepResult(BaseModel):
    """Result schema for table analysis."""
    status: AnalysisStatus
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    accuracy: Optional[float] = None

    # model_config = ConfigDict(
    #     from_attributes=True,
    #     json_encoders={
    #         datetime: lambda v: v.isoformat()
    #     }
    # )
    
class TableAnalysisDetectionResult(BaseTableAnalysisStepResult):
    """Result schema for table analysis."""
    detected_tables: List[DetectedTable] = Field(..., description="List of detected tables.")
    total_tables: int = Field(..., description="Total number of tables detected.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": AnalysisStatus.COMPLETED,
                "error": None,
                "created_at": datetime.now(),
                "completed_at": datetime.now(),
                "accuracy": 0.95,
                # "detected_tables": [DetectedTable.model_validate_json(json.dumps(DetectedTable.model_config.schema_extra["example"]))],
                "total_tables": 1
            }
        }
    )

class TableAnalysisStructureRecognitionResult(BaseTableAnalysisStepResult):
    """Result schema for table analysis."""
    recognized_structure: List[RecognizedTableStructure] = Field(..., description="List of recognized structure for each table.")
    total_structures: int = Field(..., description="Total number of tables structure recognized.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": AnalysisStatus.COMPLETED,
                "error": None,
                "created_at": datetime.now(),
                "completed_at": datetime.now(),
                "accuracy": 0.95,
                # "recognized_structure": [RecognizedTableStructure.model_validate_json(json.dumps(RecognizedTableStructure.model_config.schema_extra["example"]))],
                "total_structures": 1
            }
        }
    )

class TableAnalysisDataExtractionResult(BaseTableAnalysisStepResult):
    """Result schema for table analysis."""
    extracted_table: ExtractedTableData = Field(..., description="List of extracted tables.")
    average_confidence: float = Field(..., description="Average confidence score of the table data extraction.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": AnalysisStatus.COMPLETED,
                "error": None,
                "created_at": datetime.now(),
                "completed_at": datetime.now(),
                "accuracy": 0.95,
                # "extracted_table": ExtractedTableData.model_validate_json(json.dumps(ExtractedTableData.model_config.schema_extra["example"])),
                "average_confidence": 0.85
            }
        }
    )