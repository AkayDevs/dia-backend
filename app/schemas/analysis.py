from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

from app.schemas.algorithm import Parameter, Algorithm
from app.enums.document import DocumentType
from app.enums.analysis import AnalysisTypeEnum, AnalysisStepEnum



class AnalysisStepBase(BaseModel):
    """
    Standard analysis step representation.
    """
    name: AnalysisStepEnum
    description: Optional[str] = None
    order: int
    base_parameters: List[Parameter] = []

class AnalysisStepCreate(AnalysisStepBase):
    """
    Analysis step creation schema.
    """
    analysis_type_id: UUID

class AnalysisStepUpdate(AnalysisStepBase):
    """
    Analysis step update schema.
    """
    pass

class AnalysisStep(AnalysisStepBase):
    """
    Analysis step representation.
    """
    id: UUID
    analysis_type_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    algorithms: List[Algorithm] = []

    class Config:
        from_attributes = True


# Analysis Type Schemas ------------------------------------------------------------

class AnalysisTypeBase(BaseModel):
    """
    Standard analysis type representation.
    """
    name: AnalysisTypeEnum
    description: Optional[str] = None
    supported_document_types: List[DocumentType]

class AnalysisTypeCreate(AnalysisTypeBase):
    """
    Analysis type creation schema.
    """
    pass

class AnalysisTypeUpdate(AnalysisTypeBase):
    """
    Analysis type update schema.
    """
    pass

class AnalysisType(AnalysisTypeBase):
    """
    Analysis type representation.
    """
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    steps: List[AnalysisStep] = []

    class Config:
        from_attributes = True


# Analysis Step Result Schemas ------------------------------------------------------------

class AnalysisStepResultBase(BaseModel):
    """
    Standard analysis step result representation.
    """
    parameters: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    user_corrections: Optional[Dict[str, Any]] = None
    status: str = "pending"
    error_message: Optional[str] = None

class AnalysisStepResultCreate(AnalysisStepResultBase):
    """
    Analysis step result creation schema.
    """
    analysis_id: UUID
    step_id: UUID
    algorithm_id: UUID

class AnalysisStepResultUpdate(AnalysisStepResultBase):
    """
    Analysis step result update schema.
    """
    pass

class AnalysisStepResult(AnalysisStepResultBase):
    """
    Analysis step result representation.
    """
    id: UUID
    analysis_id: UUID
    step_id: UUID
    algorithm_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Analysis Schemas ------------------------------------------------------------

class AnalysisBase(BaseModel):
    """
    Standard analysis representation.
    """
    mode: str = Field(..., pattern="^(automatic|step_by_step)$")
    status: str = "pending"
    error_message: Optional[str] = None

class AnalysisCreate(AnalysisBase):
    """
    Analysis creation schema.
    """
    document_id: UUID
    analysis_type_id: UUID

class AnalysisUpdate(AnalysisBase):
    """
    Analysis update schema.
    """
    pass

class Analysis(AnalysisBase):
    """
    Analysis representation.
    """
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
    """
    Analysis request schema.
    """
    analysis_type_id: UUID
    mode: str = Field(..., pattern="^(automatic|step_by_step)$")
    algorithm_configs: Dict[UUID, Dict[str, Any]] = {}  # step_id -> {algorithm_id, parameters}

class StepExecutionRequest(BaseModel):
    """
    Step execution request schema.
    """
    algorithm_id: UUID
    parameters: Dict[str, Any] = {} 