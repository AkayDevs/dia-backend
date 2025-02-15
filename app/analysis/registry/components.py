from typing import List, Dict, Type, Any, Optional
from pydantic import BaseModel
from .identifiers import AnalysisIdentifier
from app.schemas.analysis import Parameter
from app.enums.document import DocumentType

class AnalysisStepInfo(BaseModel):
    """Information about an analysis step"""
    identifier: AnalysisIdentifier
    description: str
    order: int
    base_parameters: List[Parameter] = []
    result_schema: str  # Python path to the result schema class
    algorithms: List["AlgorithmInfo"] = []

class AlgorithmInfo(BaseModel):
    """Information about an algorithm"""
    identifier: AnalysisIdentifier
    description: str
    supported_document_types: List[DocumentType]
    parameters: List[Parameter]
    implementation_path: str  # Python path to the implementation class

class AnalysisTypeInfo(BaseModel):
    """Information about an analysis type"""
    identifier: AnalysisIdentifier
    description: str
    supported_document_types: List[DocumentType]
    steps: List[AnalysisStepInfo]
    implementation_path: str  # Python path to the implementation class 