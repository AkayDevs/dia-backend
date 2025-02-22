from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .algorithms import AlgorithmDefinitionInfo, AlgorithmParameter

class StepDefinitionBase(BaseModel):
    """Base schema for step definition data"""
    code: str = Field(..., description="Unique identifier code")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    description: Optional[str] = Field(None, description="Detailed description")
    order: int = Field(..., description="Execution order in analysis", ge=0)
    base_parameters: List[AlgorithmParameter] = Field(
        default=[], description="Base parameter definitions for the step"
    )
    result_schema_path: str = Field(
        ..., description="Python path to result schema class"
    )
    implementation_path: str = Field(
        ..., description="Python path to implementation class"
    )
    is_active: bool = Field(True, description="Whether this step is active")

class StepDefinitionCreate(StepDefinitionBase):
    """Schema for creating a new step definition"""
    analysis_definition_id: str = Field(..., description="ID of the parent analysis definition")

class StepDefinitionUpdate(BaseModel):
    """Schema for updating a step definition"""
    name: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = Field(None, ge=0)
    base_parameters: Optional[List[AlgorithmParameter]] = None
    result_schema_path: Optional[str] = None
    implementation_path: Optional[str] = None
    is_active: Optional[bool] = None

class StepDefinitionInDB(StepDefinitionBase):
    """Schema for step definition as stored in database"""
    id: str
    analysis_definition_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class StepDefinitionWithAlgorithms(StepDefinitionInDB):
    """Schema for step definition with its algorithms"""
    algorithms: List[AlgorithmDefinitionInfo] = Field(
        default_factory=list, description="Available algorithms for this step"
    )

class StepDefinitionInfo(BaseModel):
    """Schema for basic step definition information"""
    id: str
    code: str
    name: str
    version: str
    description: Optional[str] = None
    order: int
    is_active: bool

    class Config:
        from_attributes = True

class StepParameter(BaseModel):
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