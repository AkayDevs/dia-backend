from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.enums.document import DocumentType
from app.analysis.schemas.steps import AnalysisStepInfo, AnalysisStepWithAlgorithms

class AnalysisTypeBase(BaseModel):
    """Base schema for analysis type data"""
    code: str = Field(..., description="Unique identifier code from registry")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    description: Optional[str] = Field(None, description="Detailed description")
    supported_document_types: List[DocumentType] = Field(
        ..., description="List of supported document types"
    )
    implementation_path: str = Field(
        ..., description="Python path to implementation class"
    )
    is_active: bool = Field(True, description="Whether this analysis type is active")

class AnalysisTypeCreate(AnalysisTypeBase):
    """Schema for creating a new analysis type"""
    pass

class AnalysisTypeUpdate(BaseModel):
    """Schema for updating an analysis type"""
    name: Optional[str] = None
    description: Optional[str] = None
    supported_document_types: Optional[List[DocumentType]] = None
    implementation_path: Optional[str] = None
    is_active: Optional[bool] = None

class AnalysisTypeInDB(AnalysisTypeBase):
    """Schema for analysis type as stored in database"""
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnalysisTypeWithSteps(AnalysisTypeInDB):
    """Schema for analysis type with its steps"""
    steps: List[AnalysisStepWithAlgorithms] = Field(
        default_factory=list, description="Analysis steps in order"
    )

class AnalysisTypeInfo(BaseModel):
    """Schema for basic analysis type information"""
    id: str
    code: str
    name: str
    version: str
    description: Optional[str] = None
    supported_document_types: List[DocumentType]
    is_active: bool

    class Config:
        from_attributes = True

class AnalysisConfig(BaseModel):
    """Schema for analysis configuration"""
    analysis_type_id: str = Field(..., description="ID of the analysis type to use")
    document_id: str = Field(..., description="ID of the document to analyze")
    step_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Step-specific configurations keyed by step ID"
    )

class Analysis(BaseModel):
    """Schema for analysis execution"""
    id: str
    analysis_type_id: str
    document_id: str
    status: str = Field(..., description="Overall analysis status")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis configuration"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnalysisWithResults(Analysis):
    """Schema for analysis with step results"""
    analysis_type: AnalysisTypeInfo
    steps: List[AnalysisStepInfo]
    results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Step results keyed by step ID"
    )
