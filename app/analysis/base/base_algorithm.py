from abc import ABC, abstractmethod
from typing import Dict, Any
from app.analysis.registry.components import AlgorithmInfo

class BaseAlgorithm(ABC):
    """Base class for all algorithm implementations"""
    
    @abstractmethod
    def get_info(self) -> AlgorithmInfo:
        """Get algorithm information"""
        pass
    
    @abstractmethod
    async def validate_requirements(self) -> bool:
        """Validate that all required dependencies are available"""
        pass
    
    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate algorithm parameters"""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the algorithm"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass 