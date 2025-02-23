from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase

class BaseAlgorithm(ABC):
    """Base class for all algorithm implementations"""
    
    @abstractmethod
    def get_info(self) -> AlgorithmDefinitionBase:
        """Get algorithm definition information"""
        pass
    
    @abstractmethod
    async def validate_requirements(self) -> None:
        """
        Validate that all required dependencies are available.
        Raises exception if requirements are not met.
        """
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data format.
        Raises exception if validation fails.
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """
        Execute the algorithm implementation.
        
        Args:
            document_path: Path to the document being analyzed
            parameters: Algorithm parameters from configuration
            previous_results: Results from previous steps, keyed by step name
            
        Returns:
            Dictionary containing the algorithm results
            
        Raises:
            Exception: If execution fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup any temporary resources.
        Should be called even if execution fails.
        """
        pass 