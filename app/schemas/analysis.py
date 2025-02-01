from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

from app.schemas.algorithm import Parameter, Algorithm
from app.enums.document import DocumentType
from app.enums.analysis import AnalysisTypeEnum, AnalysisStepEnum



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
    mode: str = Field(..., pattern="^(automatic|step_by_step)$")

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