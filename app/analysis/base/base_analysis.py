from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.analysis.registry.components import AnalysisTypeInfo
from app.schemas.document import Document

class BaseAnalysis(ABC):
    """Base class for all analysis implementations"""
    
    @abstractmethod
    def get_info(self) -> AnalysisTypeInfo:
        """Get analysis type information"""
        pass
    
    @abstractmethod
    async def validate_input(self, document: Document) -> bool:
        """Validate if the document can be processed by this analysis"""
        pass
    
    @abstractmethod
    async def prepare_document(self, document: Document) -> Dict[str, Any]:
        """Prepare document for analysis and return initial data"""
        pass
    
    @abstractmethod
    async def cleanup(self, document: Document) -> None:
        """Cleanup any temporary resources"""
        pass 