from typing import Optional, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from app.db.models.document import DocumentType, AnalysisStatus, AnalysisType


class DocumentBase(BaseModel):
    name: str
    type: DocumentType
    size: int
    url: str


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[AnalysisStatus] = None


class Document(DocumentBase):
    id: str
    status: AnalysisStatus
    uploaded_at: datetime
    user_id: str

    class Config:
        from_attributes = True


class AnalysisResultBase(BaseModel):
    type: AnalysisType
    result: Optional[Any] = None


class AnalysisResultCreate(AnalysisResultBase):
    document_id: str


class AnalysisResultUpdate(BaseModel):
    result: Any


class AnalysisResult(AnalysisResultBase):
    id: str
    document_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentWithAnalysis(Document):
    analysis_results: List[AnalysisResult]

    class Config:
        from_attributes = True


# Analysis Parameters Schemas
class TableDetectionParams(BaseModel):
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    table_type: str = Field("bordered", pattern="^(bordered|borderless|both)$")


class TextExtractionParams(BaseModel):
    include_layout: bool = False
    extract_tables: bool = True


class TextSummarizationParams(BaseModel):
    max_length: int = Field(150, ge=50, le=500)
    min_length: int = Field(50, ge=30, le=100)


class TemplateConversionParams(BaseModel):
    output_format: str = Field("docx", pattern="^(docx|pdf)$")
    preserve_layout: bool = True


class AnalysisParameters(BaseModel):
    type: AnalysisType
    params: Optional[
        TableDetectionParams |
        TextExtractionParams |
        TextSummarizationParams |
        TemplateConversionParams
    ] = None 