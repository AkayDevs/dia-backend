from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from app.enums.analysis import AnalysisStatus, AnalysisMode
from .step_result import StepExecutionResultInfo

class AnalysisRunBase(BaseModel):
    """Base schema for analysis run data"""
    document_id: str = Field(..., description="ID of the document being analyzed")
    analysis_definition_id: str = Field(..., description="ID of the analysis definition")
    mode: AnalysisMode = Field(..., description="Analysis execution mode")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the analysis run")

class AnalysisRunCreate(AnalysisRunBase):
    """Schema for creating a new analysis run"""
    pass

class AnalysisRunUpdate(BaseModel):
    """Schema for updating an analysis run"""
    status: Optional[AnalysisStatus] = None
    config: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class AnalysisRunInDB(AnalysisRunBase):
    """Schema for analysis run as stored in database"""
    id: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class AnalysisRunInfo(AnalysisRunInDB):
    """Schema for analysis run with basic information"""
    pass

class AnalysisRunWithResults(AnalysisRunInfo):
    """Schema for analysis run with step results"""
    step_results: List[StepExecutionResultInfo] = Field(
        default_factory=list,
        description="Results of individual analysis steps"
    )

    class Config:
        from_attributes = True 