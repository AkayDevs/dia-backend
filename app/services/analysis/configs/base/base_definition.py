from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from app.schemas.analysis.configs.definitions import AnalysisDefinitionInfo
from app.schemas.document import Document as DocumentInfo
from app.enums.document import DocumentType

class BaseDefinition(ABC):
    """Base class for all analysis definition implementations"""
    
    @abstractmethod
    def get_info(self) -> AnalysisDefinitionInfo:
        """Get analysis definition information"""
        pass
    
    @abstractmethod
    async def validate_requirements(self) -> None:
        """
        Validate that all required dependencies are available.
        Raises exception if requirements are not met.
        """
        pass
    
    @abstractmethod
    async def validate_document(self, document: DocumentInfo) -> None:
        """
        Validate if document can be processed by this analysis.
        Raises exception if validation fails.
        """
        pass
    
    @abstractmethod
    async def prepare_document(
        self,
        document: DocumentInfo,
        document_path: str
    ) -> Dict[str, Any]:
        """
        Prepare document for analysis.
        
        Args:
            document: Document information
            document_path: Path to the document file
            
        Returns:
            Dictionary containing prepared document data
            
        Raises:
            Exception: If preparation fails
        """
        pass
    
    @abstractmethod
    async def validate_results(
        self,
        results: List[Dict[str, Any]],
        document: DocumentInfo
    ) -> None:
        """
        Validate final analysis results.
        
        Args:
            results: List of results from all steps
            document: Document information
            
        Raises:
            Exception: If validation fails
        """
        pass
    
    @abstractmethod
    async def post_process_results(
        self,
        results: List[Dict[str, Any]],
        document: DocumentInfo
    ) -> Dict[str, Any]:
        """
        Post-process all analysis results.
        
        Args:
            results: List of results from all steps
            document: Document information
            
        Returns:
            Dictionary containing final processed results
            
        Raises:
            Exception: If post-processing fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self, document: DocumentInfo) -> None:
        """
        Cleanup any temporary resources.
        Should be called even if analysis fails.
        """
        pass 