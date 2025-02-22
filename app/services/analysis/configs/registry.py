from typing import Dict, Optional, List, Set
import logging
from app.schemas.analysis.configs.definitions import AnalysisDefinitionInfo
from app.schemas.analysis.configs.steps import StepDefinitionInfo
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionInfo

logger = logging.getLogger(__name__)

class AnalysisRegistry:
    """Central registry for all analysis definitions, steps, and algorithms"""
    
    _analysis_definitions: Dict[str, AnalysisDefinitionInfo] = {}
    _steps: Dict[str, StepDefinitionInfo] = {}
    _algorithms: Dict[str, AlgorithmDefinitionInfo] = {}
    
    # Track relationships
    _analysis_steps: Dict[str, Set[str]] = {}  # analysis_code -> set of step codes
    _step_algorithms: Dict[str, Set[str]] = {}  # step_code -> set of algorithm codes
    
    @classmethod
    def register_analysis_definition(cls, definition: AnalysisDefinitionInfo) -> None:
        """Register a new analysis definition"""
        if definition.code in cls._analysis_definitions:
            logger.warning(f"Analysis definition {definition.code} already registered. Updating...")
        
        cls._analysis_definitions[definition.code] = definition
        cls._analysis_steps[definition.code] = set()
        logger.info(f"Registered analysis definition: {definition.code}")
    
    @classmethod
    def register_step(cls, step: StepDefinitionInfo, analysis_code: str) -> None:
        """Register a new analysis step"""
        if analysis_code not in cls._analysis_definitions:
            raise ValueError(f"Analysis definition {analysis_code} not found")
        
        step_code = f"{analysis_code}.{step.code}"
        if step_code in cls._steps:
            logger.warning(f"Step {step_code} already registered. Updating...")
        
        cls._steps[step_code] = step
        cls._analysis_steps[analysis_code].add(step_code)
        cls._step_algorithms[step_code] = set()
        logger.info(f"Registered step: {step_code}")
    
    @classmethod
    def register_algorithm(cls, algorithm: AlgorithmDefinitionInfo, step_code: str) -> None:
        """Register a new algorithm"""
        if step_code not in cls._steps:
            raise ValueError(f"Step {step_code} not found")
            
        algo_code = f"{step_code}.{algorithm.code}"
        if algo_code in cls._algorithms:
            logger.warning(f"Algorithm {algo_code} already registered. Updating...")
        
        cls._algorithms[algo_code] = algorithm
        cls._step_algorithms[step_code].add(algo_code)
        logger.info(f"Registered algorithm: {algo_code}")
    
    @classmethod
    def get_analysis_definition(cls, code: str) -> Optional[AnalysisDefinitionInfo]:
        """Get analysis definition by code"""
        return cls._analysis_definitions.get(code)
    
    @classmethod
    def get_step(cls, code: str) -> Optional[StepDefinitionInfo]:
        """Get step by code"""
        return cls._steps.get(code)
    
    @classmethod
    def get_algorithm(cls, code: str) -> Optional[AlgorithmDefinitionInfo]:
        """Get algorithm by code"""
        return cls._algorithms.get(code)
    
    @classmethod
    def list_analysis_definitions(cls) -> List[AnalysisDefinitionInfo]:
        """List all registered analysis definitions"""
        return list(cls._analysis_definitions.values())
    
    @classmethod
    def list_steps(cls, analysis_code: str) -> List[StepDefinitionInfo]:
        """List all steps for an analysis definition"""
        if analysis_code not in cls._analysis_steps:
            return []
        return [cls._steps[code] for code in cls._analysis_steps[analysis_code]]
    
    @classmethod
    def list_algorithms(cls, step_code: str) -> List[AlgorithmDefinitionInfo]:
        """List all algorithms for a step"""
        if step_code not in cls._step_algorithms:
            return []
        return [cls._algorithms[code] for code in cls._step_algorithms[step_code]]
    
    @classmethod
    def deregister_analysis_definition(cls, code: str) -> None:
        """Remove an analysis definition and all its associated steps and algorithms"""
        if code not in cls._analysis_definitions:
            return
            
        # Remove associated steps and their algorithms
        for step_code in cls._analysis_steps.get(code, set()):
            # Remove associated algorithms
            for algo_code in cls._step_algorithms.get(step_code, set()):
                cls._algorithms.pop(algo_code, None)
            cls._step_algorithms.pop(step_code, None)
            cls._steps.pop(step_code, None)
            
        cls._analysis_steps.pop(code, None)
        cls._analysis_definitions.pop(code, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations"""
        cls._analysis_definitions.clear()
        cls._steps.clear()
        cls._algorithms.clear()
        cls._analysis_steps.clear()
        cls._step_algorithms.clear()
        logger.info("Analysis registry cleared") 