from typing import Dict, Optional, List
import logging
from .components import AnalysisTypeInfo, AnalysisStepInfo, AlgorithmInfo

logger = logging.getLogger(__name__)

class AnalysisRegistry:
    """Central registry for all analysis types, steps, and algorithms"""
    
    _analysis_types: Dict[str, AnalysisTypeInfo] = {}
    _steps: Dict[str, AnalysisStepInfo] = {}
    _algorithms: Dict[str, AlgorithmInfo] = {}
    
    @classmethod
    def register_analysis_type(cls, analysis_type: AnalysisTypeInfo) -> None:
        """Register a new analysis type"""
        code = analysis_type.identifier.code
        if code in cls._analysis_types:
            logger.warning(f"Analysis type {code} already registered. Updating...")
        
        cls._analysis_types[code] = analysis_type
        
        # Register associated steps
        for step in analysis_type.steps:
            cls.register_step(step, analysis_type.identifier.code)
    
    @classmethod
    def register_step(cls, step: AnalysisStepInfo, analysis_type_code: str) -> None:
        """Register a new analysis step"""
        code = f"{analysis_type_code}.{step.identifier.code}"
        if code in cls._steps:
            logger.warning(f"Step {code} already registered. Updating...")
        
        cls._steps[code] = step
        
        # Register associated algorithms
        for algo in step.algorithms:
            cls.register_algorithm(algo, code)
    
    @classmethod
    def register_algorithm(cls, algorithm: AlgorithmInfo, step_code: str) -> None:
        """Register a new algorithm"""
        code = f"{step_code}.{algorithm.identifier.code}"
        if code in cls._algorithms:
            logger.warning(f"Algorithm {code} already registered. Updating...")
        
        cls._algorithms[code] = algorithm
    
    @classmethod
    def get_analysis_type(cls, code: str) -> Optional[AnalysisTypeInfo]:
        """Get analysis type by code"""
        return cls._analysis_types.get(code)
    
    @classmethod
    def get_step(cls, code: str) -> Optional[AnalysisStepInfo]:
        """Get step by code"""
        return cls._steps.get(code)
    
    @classmethod
    def get_algorithm(cls, code: str) -> Optional[AlgorithmInfo]:
        """Get algorithm by code"""
        return cls._algorithms.get(code)
    
    @classmethod
    def list_analysis_types(cls) -> List[AnalysisTypeInfo]:
        """List all registered analysis types"""
        return list(cls._analysis_types.values())
    
    @classmethod
    def list_steps(cls, analysis_type_code: str) -> List[AnalysisStepInfo]:
        """List all steps for an analysis type"""
        return [
            step for step_code, step in cls._steps.items()
            if step_code.startswith(f"{analysis_type_code}.")
        ]
    
    @classmethod
    def list_algorithms(cls, step_code: str) -> List[AlgorithmInfo]:
        """List all algorithms for a step"""
        return [
            algo for algo_code, algo in cls._algorithms.items()
            if algo_code.startswith(f"{step_code}.")
        ]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)"""
        cls._analysis_types.clear()
        cls._steps.clear()
        cls._algorithms.clear() 