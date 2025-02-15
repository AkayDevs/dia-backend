from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, conint
from datetime import datetime

from app.analysis.schemas.algorithms import AlgorithmInfo

class AnalysisStepBase(BaseModel):
    """Base schema for analysis step data"""
    code: str = Field(..., description="Unique identifier code from registry")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    description: Optional[str] = Field(None, description="Detailed description")
    order: int = Field(..., description="Execution order in analysis type", ge=0)
    base_parameters: List[Dict[str, Any]] = Field(
        default=[], description="Base parameter definitions for the step"
    )
    result_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for step results"
    )
    implementation_path: str = Field(
        ..., description="Python path to implementation class"
    )
    is_active: bool = Field(True, description="Whether this step is active")

class AnalysisStepCreate(AnalysisStepBase):
    """Schema for creating a new analysis step"""
    analysis_type_id: str = Field(..., description="ID of the parent analysis type")

class AnalysisStepUpdate(BaseModel):
    """Schema for updating an analysis step"""
    name: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = Field(None, ge=0)
    base_parameters: Optional[List[Dict[str, Any]]] = None
    result_schema: Optional[Dict[str, Any]] = None
    implementation_path: Optional[str] = None
    is_active: Optional[bool] = None

class AnalysisStepInDB(AnalysisStepBase):
    """Schema for analysis step as stored in database"""
    id: str
    analysis_type_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnalysisStepWithAlgorithms(AnalysisStepInDB):
    """Schema for analysis step with its algorithms"""
    algorithms: List[AlgorithmInfo] = Field(
        default_factory=list, description="Available algorithms for this step"
    )

class AnalysisStepInfo(BaseModel):
    """Schema for basic analysis step information"""
    id: str
    code: str
    name: str
    version: str
    description: Optional[str] = None
    order: int
    is_active: bool

    class Config:
        from_attributes = True

class AnalysisStepParameter(BaseModel):
    """Schema for step parameter definition"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (e.g., 'string', 'number')")
    description: Optional[str] = Field(None, description="Parameter description")
    required: bool = Field(True, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value if any")
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Parameter constraints (e.g., min, max, pattern)"
    )

class AnalysisStepResult(BaseModel):
    """Schema for step execution result"""
    step_id: str = Field(..., description="ID of the analysis step")
    analysis_id: str = Field(..., description="ID of the parent analysis")
    algorithm_id: str = Field(..., description="ID of the algorithm used")
    status: str = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Step results")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True 