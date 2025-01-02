from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"

class DocumentBase(BaseModel):
    name: str
    type: DocumentType

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: int
    size: int
    status: DocumentStatus
    url: str
    file_path: str
    metadata: Optional[Dict[str, Any]] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    has_tables: Optional[bool] = None
    has_images: Optional[bool] = None
    created_at: datetime
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    owner_id: int

    class Config:
        from_attributes = True

class AnalysisConfig(BaseModel):
    analysis_type: str  # table_detection, text_extraction, etc.
    options: Dict[str, Any] = {}

class BatchAnalysisRequest(BaseModel):
    document_ids: List[int]
    analysis_config: AnalysisConfig

class AnalysisResult(BaseModel):
    id: int
    document_id: int
    analysis_type: str
    result_data: str  # JSON string of results
    created_at: datetime

    class Config:
        from_attributes = True

class ParameterDefinition(BaseModel):
    type: str
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    options: Optional[List[str]] = None
    default: Any

class AnalysisTypeParameters(BaseModel):
    parameters: Dict[str, ParameterDefinition]

class AnalysisParameters(BaseModel):
    available_types: List[str]
    parameters: Dict[str, Dict[str, ParameterDefinition]]

class ExportRequest(BaseModel):
    document_ids: List[int]
    format: str = "pdf"  # pdf, csv, json, etc.
    include_visualizations: bool = True

class ExportResponse(BaseModel):
    export_url: str
    format: str
    expires_at: datetime 