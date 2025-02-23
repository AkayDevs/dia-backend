from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.enums.analysis import AnalysisStatus
from app.schemas.analysis.results.base import BaseResultSchema

class StepExecutionResultBase(BaseModel):
    """Base schema for step execution result data"""
    analysis_run_id: str = Field(..., description="ID of the parent analysis run")
    step_definition_id: str = Field(..., description="ID of the step definition")
    algorithm_definition_id: str = Field(..., description="ID of the algorithm definition")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for this execution")

class StepExecutionResultCreate(StepExecutionResultBase):
    """Schema for creating a new step execution result"""
    status: AnalysisStatus = Field(AnalysisStatus.PENDING, description="Initial execution status")

class StepExecutionResultUpdate(BaseModel):
    """Schema for updating a step execution result"""
    status: Optional[AnalysisStatus] = None
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[BaseResultSchema] = None
    user_corrections: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class StepExecutionResultInDB(StepExecutionResultBase):
    """Schema for step execution result as stored in database"""
    id: str
    status: AnalysisStatus
    result: Optional[BaseResultSchema] = Field(
        default=None,
        description="Validated analysis result following the step's result schema"
    )
    user_corrections: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User-provided corrections to the result"
    )
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class StepExecutionResultInfo(StepExecutionResultInDB):
    """Schema for step execution result with basic information"""
    pass