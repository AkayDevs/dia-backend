from typing import List, Optional, Any
from pydantic import BaseModel
from app.enums.document import DocumentType
from uuid import UUID
from datetime import datetime


class Parameter(BaseModel):
    name: str
    description: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    required: bool = False
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None

class AlgorithmBase(BaseModel):
    name: str
    description: Optional[str] = None
    version: str
    supported_document_types: List[DocumentType]
    parameters: List[Parameter] = []
    is_active: bool = True

class AlgorithmCreate(AlgorithmBase):
    step_id: UUID

class AlgorithmUpdate(AlgorithmBase):
    pass

class Algorithm(AlgorithmBase):
    id: UUID
    step_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True