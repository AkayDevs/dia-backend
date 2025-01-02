from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
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

class DocumentMetadata(BaseModel):
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    has_tables: Optional[bool] = None
    has_images: Optional[bool] = None

class DocumentBase(BaseModel):
    name: str
    type: DocumentType
    size: int

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    status: Optional[DocumentStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    has_tables: Optional[bool] = None
    has_images: Optional[bool] = None
    processed_at: Optional[datetime] = None

class Document(DocumentBase):
    id: int
    status: DocumentStatus
    url: str
    metadata: Optional[DocumentMetadata] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    user_id: int

    class Config:
        from_attributes = True 