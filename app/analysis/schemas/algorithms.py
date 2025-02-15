from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.enums.document import DocumentType

class AlgorithmBase(BaseModel):
    """Base schema for algorithm data"""
    code: str = Field(..., description="Unique identifier code from registry")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    description: Optional[str] = Field(None, description="Detailed description")
    supported_document_types: List[DocumentType] = Field(
        ..., description="List of supported document types"
    )
    parameters: List[Dict[str, Any]] = Field(
        default=[], description="List of parameter definitions"
    )
    implementation_path: str = Field(
        ..., description="Python path to implementation class"
    )
    is_active: bool = Field(True, description="Whether this algorithm is active")

class AlgorithmCreate(AlgorithmBase):
    """Schema for creating a new algorithm"""
    step_id: str = Field(..., description="ID of the parent analysis step")

class AlgorithmUpdate(BaseModel):
    """Schema for updating an algorithm"""
    name: Optional[str] = None
    description: Optional[str] = None
    supported_document_types: Optional[List[DocumentType]] = None
    parameters: Optional[List[Dict[str, Any]]] = None
    implementation_path: Optional[str] = None
    is_active: Optional[bool] = None

class AlgorithmInDB(AlgorithmBase):
    """Schema for algorithm as stored in database"""
    id: str
    step_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AlgorithmInfo(BaseModel):
    """Schema for basic algorithm information"""
    id: str
    code: str
    name: str
    version: str
    description: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True

class AlgorithmParameter(BaseModel):
    """Schema for algorithm parameter definition"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (e.g., 'string', 'number')")
    description: Optional[str] = Field(None, description="Parameter description")
    required: bool = Field(True, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value if any")
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Parameter constraints (e.g., min, max, pattern)"
    )

class AlgorithmConfig(BaseModel):
    """Schema for algorithm configuration in analysis"""
    algorithm_id: str = Field(..., description="ID of the algorithm to use")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
    )

class AlgorithmSelection(BaseModel):
    """Schema for selecting an algorithm for a step"""
    code: str = Field(..., description="Algorithm code")
    version: str = Field(..., description="Algorithm version")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Algorithm parameters"
    )
