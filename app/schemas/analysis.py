from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

from app.db.models.analysis import AnalysisTypeEnum, AnalysisStepEnum
from app.schemas.document import DocumentType

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

class AnalysisStepBase(BaseModel):
    name: AnalysisStepEnum
    description: Optional[str] = None
    order: int
    base_parameters: List[Parameter] = []

class AnalysisStepCreate(AnalysisStepBase):
    analysis_type_id: UUID

class AnalysisStepUpdate(AnalysisStepBase):
    pass

class AnalysisStep(AnalysisStepBase):
    id: UUID
    analysis_type_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    algorithms: List[Algorithm] = []

    class Config:
        from_attributes = True

class AnalysisTypeBase(BaseModel):
    name: AnalysisTypeEnum
    description: Optional[str] = None
    supported_document_types: List[DocumentType]

class AnalysisTypeCreate(AnalysisTypeBase):
    pass

class AnalysisTypeUpdate(AnalysisTypeBase):
    pass

class AnalysisType(AnalysisTypeBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    steps: List[AnalysisStep] = []

    class Config:
        from_attributes = True

class AnalysisStepResultBase(BaseModel):
    parameters: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    user_corrections: Optional[Dict[str, Any]] = None
    status: str = "pending"
    error_message: Optional[str] = None

class AnalysisStepResultCreate(AnalysisStepResultBase):
    analysis_id: UUID
    step_id: UUID
    algorithm_id: UUID

class AnalysisStepResultUpdate(AnalysisStepResultBase):
    pass

class AnalysisStepResult(AnalysisStepResultBase):
    id: UUID
    analysis_id: UUID
    step_id: UUID
    algorithm_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnalysisBase(BaseModel):
    mode: str = Field(..., pattern="^(automatic|step_by_step)$")
    status: str = "pending"
    error_message: Optional[str] = None

class AnalysisCreate(AnalysisBase):
    document_id: UUID
    analysis_type_id: UUID

class AnalysisUpdate(AnalysisBase):
    pass

class Analysis(AnalysisBase):
    id: UUID
    document_id: UUID
    analysis_type_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    step_results: List[AnalysisStepResult] = []

    class Config:
        from_attributes = True

# Request/Response Models
class AnalysisRequest(BaseModel):
    analysis_type_id: UUID
    mode: str = Field(..., pattern="^(automatic|step_by_step)$")
    algorithm_configs: Dict[UUID, Dict[str, Any]] = {}  # step_id -> {algorithm_id, parameters}

class StepExecutionRequest(BaseModel):
    algorithm_id: UUID
    parameters: Dict[str, Any] = {} 