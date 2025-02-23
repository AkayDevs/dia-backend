from typing import Dict, Optional, List, Set
import logging
from app.schemas.analysis.configs.definitions import AnalysisDefinitionBase, AnalysisDefinitionInfo
from app.schemas.analysis.configs.steps import StepDefinitionBase
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase

logger = logging.getLogger(__name__)

class AnalysisRegistry:
    """Central registry for all analysis definitions, steps, and algorithms"""
    
    _analysis_definitions: Dict[str, AnalysisDefinitionBase] = {}  # code -> definition
    _steps: Dict[str, StepDefinitionBase] = {}  # analysis_code.step_code -> step
    _algorithms: Dict[str, AlgorithmDefinitionBase] = {}  # analysis_code.step_code.algo_code -> algorithm
    
    @classmethod
    def register_analysis_definition(cls, definition: AnalysisDefinitionBase) -> None:
        """Register a new analysis definition"""
        if definition.code in cls._analysis_definitions:
            logger.warning(f"Analysis definition {definition.code} already registered. Updating...")
        cls._analysis_definitions[definition.code] = definition
        logger.info(f"Registered analysis definition: {definition.code}")
    
    @classmethod
    def register_step(cls, step: StepDefinitionBase, analysis_code: str) -> None:
        """Register a new analysis step"""
        if analysis_code not in cls._analysis_definitions:
            raise ValueError(f"Analysis definition {analysis_code} not found")
        
        step_code = f"{analysis_code}.{step.code}"
        if step_code in cls._steps:
            logger.warning(f"Step {step_code} already registered. Updating...")
        cls._steps[step_code] = step
        logger.info(f"Registered step: {step_code}")
    
    @classmethod
    def register_algorithm(cls, algorithm: AlgorithmDefinitionBase, step_code: str) -> None:
        """Register a new algorithm"""
        if step_code not in cls._steps:
            raise ValueError(f"Step {step_code} not found")
            
        algo_code = f"{step_code}.{algorithm.code}"
        if algo_code in cls._algorithms:
            logger.warning(f"Algorithm {algo_code} already registered. Updating...")
        cls._algorithms[algo_code] = algorithm
        logger.info(f"Registered algorithm: {algo_code}")
    
    @classmethod
    def get_analysis_definition(cls, code: str) -> Optional[AnalysisDefinitionBase]:
        """Get analysis definition by code"""
        return cls._analysis_definitions.get(code)
    
    @classmethod
    def get_step(cls, code: str) -> Optional[StepDefinitionBase]:
        """Get step by code"""
        return cls._steps.get(code)
    
    @classmethod
    def get_algorithm(cls, code: str) -> Optional[AlgorithmDefinitionBase]:
        """Get algorithm by code"""
        return cls._algorithms.get(code)
    
    @classmethod
    def list_analysis_definitions(cls) -> List[AnalysisDefinitionBase]:
        """List all registered analysis definitions"""
        return list(cls._analysis_definitions.values())
    
    @classmethod
    def list_steps(cls, analysis_code: str) -> List[StepDefinitionBase]:
        """List all steps for an analysis definition"""
        return [
            step for code, step in cls._steps.items()
            if code.startswith(f"{analysis_code}.")
        ]
    
    @classmethod
    def list_algorithms(cls, step_code: str) -> List[AlgorithmDefinitionBase]:
        """List all algorithms for a step"""
        return [
            algo for code, algo in cls._algorithms.items()
            if code.startswith(f"{step_code}.")
        ]
    
    @classmethod
    def deregister_analysis_definition(cls, code: str) -> None:
        """Remove an analysis definition and all its associated steps and algorithms"""
        if code not in cls._analysis_definitions:
            return
            
        # Remove steps and algorithms with matching prefix
        cls._steps = {
            k: v for k, v in cls._steps.items()
            if not k.startswith(f"{code}.")
        }
        cls._algorithms = {
            k: v for k, v in cls._algorithms.items()
            if not k.startswith(f"{code}.")
        }
        cls._analysis_definitions.pop(code, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations"""
        cls._analysis_definitions.clear()
        cls._steps.clear()
        cls._algorithms.clear()
        logger.info("Analysis registry cleared") 