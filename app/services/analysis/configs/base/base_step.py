from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from app.schemas.analysis.configs.steps import StepDefinitionInfo
from app.schemas.analysis.executions import StepExecutionResultInfo

class BaseStep(ABC):
    """Base class for all step implementations"""
    
    @abstractmethod
    def get_info(self) -> StepDefinitionInfo:
        """Get step definition information"""
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
    async def validate_result(self, result: Dict[str, Any]) -> None:
        """
        Validate step execution result.
        Raises exception if validation fails.
        """
        pass
    
    @abstractmethod
    async def prepare_execution(
        self,
        document_path: str,
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """
        Prepare data for step execution.
        
        Args:
            document_path: Path to the document being analyzed
            previous_results: Results from previous steps, keyed by step name
            
        Returns:
            Dictionary containing prepared data for algorithm execution
            
        Raises:
            Exception: If preparation fails
        """
        pass
    
    @abstractmethod
    async def post_process_result(
        self,
        result: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """
        Post-process algorithm execution result.
        
        Args:
            result: Raw result from algorithm execution
            previous_results: Results from previous steps, keyed by step name
            
        Returns:
            Dictionary containing processed result
            
        Raises:
            Exception: If post-processing fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup any temporary resources.
        Should be called even if execution fails.
        """
        pass 