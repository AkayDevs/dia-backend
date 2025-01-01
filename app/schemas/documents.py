from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

class DocumentBase(BaseModel):
    title: str
    file_type: str

class Document(DocumentBase):
    id: int
    status: str
    created_at: datetime
    file_path: str
    owner_id: int

    class Config:
        from_attributes = True

class AnalysisConfig(BaseModel):
    analysis_type: str  # table_detection, text_extraction, etc.
    options: dict = {}  # Additional analysis options

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