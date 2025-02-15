from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.analysis.registry.components import AnalysisStepInfo
from app.schemas.analysis import AnalysisStepResult

class BaseStep(ABC):
    """Base class for all analysis step implementations"""
    
    @abstractmethod
    def get_info(self) -> AnalysisStepInfo:
        """Get step information"""
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for this step"""
        pass
    
    @abstractmethod
    async def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate step result"""
        pass
    
    @abstractmethod
    async def prepare_step(self, previous_result: Optional[AnalysisStepResult] = None) -> Dict[str, Any]:
        """Prepare data for step execution"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass 