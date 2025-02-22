from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.enums.document import DocumentType
from .steps import StepDefinitionInfo, StepDefinitionWithAlgorithms

class AnalysisDefinitionBase(BaseModel):
    """Base schema for analysis definition data"""
    code: str = Field(..., description="Unique identifier code")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    description: Optional[str] = Field(None, description="Detailed description")
    supported_document_types: List[DocumentType] = Field(
        ..., description="List of supported document types"
    )
    implementation_path: str = Field(
        ..., description="Python path to implementation class"
    )
    is_active: bool = Field(True, description="Whether this analysis definition is active")

class AnalysisDefinitionCreate(AnalysisDefinitionBase):
    """Schema for creating a new analysis definition"""
    pass

class AnalysisDefinitionUpdate(BaseModel):
    """Schema for updating an analysis definition"""
    name: Optional[str] = None
    description: Optional[str] = None
    supported_document_types: Optional[List[DocumentType]] = None
    implementation_path: Optional[str] = None
    is_active: Optional[bool] = None

class AnalysisDefinitionInDB(AnalysisDefinitionBase):
    """Schema for analysis definition as stored in database"""
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnalysisDefinitionWithSteps(AnalysisDefinitionInDB):
    """Schema for analysis definition with its steps"""
    steps: List[StepDefinitionInfo] = Field(
        default_factory=list, description="Analysis steps in order"
    )

class AnalysisDefinitionWithStepsAndAlgorithms(AnalysisDefinitionWithSteps):
    """Schema for analysis definition with its steps and algorithms"""
    steps: List[StepDefinitionWithAlgorithms] = Field(
        default_factory=list, description="Analysis steps with algorithms"
    )

class AnalysisDefinitionInfo(BaseModel):
    """Schema for basic analysis definition information"""
    id: str
    code: str
    name: str
    version: str
    description: Optional[str] = None
    supported_document_types: List[DocumentType]
    is_active: bool

    class Config:
        from_attributes = True 