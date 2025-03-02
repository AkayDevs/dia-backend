from typing import Dict, Any, Optional, Type, List
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import importlib
import inspect
import re

from app.crud import crud_analysis_config, crud_document, crud_analysis_execution
from app.schemas.analysis.executions import (
    AnalysisRunInfo,
    StepExecutionResultInfo,
    StepExecutionResultUpdate,
    AnalysisRunUpdate,
    AnalysisRunWithResults
)
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase, AlgorithmParameterValue, AlgorithmParameter
from app.schemas.document import Document
from app.core.config import settings
from app.enums.analysis import AnalysisStatus
from app.services.analysis.results.schema_loader import ResultSchemaLoader
from app.services.analysis.configs.base.base_algorithm import BaseAlgorithm
from app.services.analysis.configs.registry import AnalysisRegistry

logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """Orchestrates the execution of analysis steps."""
    
    def _get_implementation(self, algorithm: AlgorithmDefinitionBase) -> Optional[Type[BaseAlgorithm]]:
        """Get implementation class for an algorithm."""
        try:
            # Import the implementation class
            module_path, class_name = algorithm.implementation_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            implementation_class = getattr(module, class_name)
            
            # Validate it's a proper algorithm implementation
            if not issubclass(implementation_class, BaseAlgorithm):
                raise TypeError(f"Implementation {algorithm.implementation_path} must inherit from BaseAlgorithm")
            
            return implementation_class
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading implementation for algorithm {algorithm.code}: {str(e)}")
            return None
    
    def _validate_parameters(self, provided_params: Dict[str, Any], param_definitions: List[AlgorithmParameter]) -> None:
        """Validate parameters against their definitions."""
        for param_def in param_definitions:
            if param_def.required and param_def.name not in provided_params:
                raise ValueError(f"Required parameter '{param_def.name}' not provided")
            
            if param_def.name in provided_params:
                value = provided_params[param_def.name]
                
                # Type validation based on string type
                type_mapping = {
                    "string": str,
                    "integer": int,
                    "float": float,
                    "boolean": bool
                }
                expected_type = type_mapping.get(param_def.type)
                if not expected_type:
                    logger.warning(f"Unknown parameter type: {param_def.type}")
                    continue
                
                if not isinstance(value, expected_type):
                    raise ValueError(f"Parameter '{param_def.name}' must be of type {param_def.type}")
                
                # Validate constraints if they exist
                if param_def.constraints:
                    if param_def.type in ["integer", "float"]:
                        # Numeric constraints
                        if "min" in param_def.constraints and value < param_def.constraints["min"]:
                            raise ValueError(f"Parameter '{param_def.name}' must be >= {param_def.constraints['min']}")
                        if "max" in param_def.constraints and value > param_def.constraints["max"]:
                            raise ValueError(f"Parameter '{param_def.name}' must be <= {param_def.constraints['max']}")
                    
                    elif param_def.type == "string":
                        # String constraints
                        if "min_length" in param_def.constraints and len(value) < param_def.constraints["min_length"]:
                            raise ValueError(f"Parameter '{param_def.name}' must be at least {param_def.constraints['min_length']} characters")
                        if "max_length" in param_def.constraints and len(value) > param_def.constraints["max_length"]:
                            raise ValueError(f"Parameter '{param_def.name}' must be at most {param_def.constraints['max_length']} characters")
                        if "pattern" in param_def.constraints and not re.match(param_def.constraints["pattern"], value):
                            raise ValueError(f"Parameter '{param_def.name}' must match pattern: {param_def.constraints['pattern']}")
                        if "allowed_values" in param_def.constraints and value not in param_def.constraints["allowed_values"]:
                            raise ValueError(f"Parameter '{param_def.name}' must be one of: {param_def.constraints['allowed_values']}")

    async def run_step(self, db: Session, step_result_id: str) -> None:
        """Execute a single analysis step."""
        step_result: Optional[StepExecutionResultInfo] = None
        analysis_run: Optional[AnalysisRunInfo] = None
        implementation: Optional[BaseAlgorithm] = None
        
        try:
            # Get step result and related info
            step_result = crud_analysis_execution.step_execution_result.get(db, id=step_result_id)
            if not step_result:
                logger.error(f"Step result not found: {step_result_id}")
                return
            
            # Update status to in_progress
            step_result_update = StepExecutionResultUpdate(
                status=AnalysisStatus.IN_PROGRESS,
                started_at=datetime.utcnow()
            )
            step_result = crud_analysis_execution.step_execution_result.update(
                db,
                db_obj=step_result,
                obj_in=step_result_update
            )
            
            # Get document path
            analysis_run = crud_analysis_execution.analysis_run.get(db, id=step_result.analysis_run_id)
            document: Document = crud_document.document.get(db, id=analysis_run.document_id)
            
            try:
                # Get algorithm implementation
                algorithm_code = f"{step_result.step_code}.{step_result.algorithm_code}"
                algorithm = AnalysisRegistry.get_algorithm(algorithm_code)
                if not algorithm:
                    raise Exception(f"Algorithm not found: {algorithm_code}")
                
                implementation_class = self._get_implementation(algorithm)
                if not implementation_class:
                    raise Exception(f"Implementation not found for algorithm: {algorithm.implementation_path}")
                
                implementation = implementation_class()
                
                # Validate requirements and parameters
                # await implementation.validate_requirements()
                # self._validate_parameters(step_result.parameters, algorithm.parameters)
                
                # Get previous results
                previous_results = self._get_previous_results(analysis_run, step_result)
                
                # Execute implementation
                result = await implementation.execute(
                    document_path=document.url.replace("/uploads/", ""),
                    parameters=step_result.parameters,
                    previous_results=previous_results
                )
                
                # Validate result
                step_definition = AnalysisRegistry.get_step(step_result.step_code)
                result_schema_class = ResultSchemaLoader.load_schema(step_definition.result_schema_path)
                validated_result = result_schema_class(**result)
                
                # Update step result
                step_result_update = StepExecutionResultUpdate(
                    status=AnalysisStatus.COMPLETED,
                    result=validated_result,
                    completed_at=datetime.utcnow()
                )
                crud_analysis_execution.step_execution_result.update_result(
                    db,
                    db_obj=step_result,
                    result=validated_result,
                    status=AnalysisStatus.COMPLETED
                )
                
            finally:
                if implementation:
                    await implementation.cleanup()

            # Update analysis status
            self._update_analysis_status(db, analysis_run)
            
        except Exception as e:
            logger.error(f"Error executing step {step_result_id}: {str(e)}")
            if step_result:
                step_result_update = StepExecutionResultUpdate(
                    status=AnalysisStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )
                crud_analysis_execution.step_execution_result.update(
                    db,
                    db_obj=step_result,
                    obj_in=step_result_update
                )
                if analysis_run:
                    self._update_analysis_status(db, analysis_run)
    
    def _get_previous_results(self, analysis_run: AnalysisRunWithResults, current_step: StepExecutionResultInfo) -> Dict[str, Dict[str, Any]]:
        """Get results from previous steps."""
        previous_results = {}
        if analysis_run.step_results:
            # Get registered steps and their order
            registered_steps = AnalysisRegistry.list_steps(analysis_run.analysis_code)
            step_order = {
                f"{analysis_run.analysis_code}.{step.code}": step.order 
                for step in registered_steps
            }
            sorted_results = sorted(
                analysis_run.step_results,
                key=lambda x: step_order.get(x.step_code, float('inf'))
            )
            current_idx = next(
                (i for i, r in enumerate(sorted_results) if r.id == current_step.id),
                -1
            )
            for prev_result in sorted_results[:current_idx]:
                if prev_result.status == AnalysisStatus.COMPLETED:
                    previous_results[prev_result.step_code] = prev_result.result
                else:
                    previous_results[prev_result.step_code] = None
                    logger.warning(f"Cannot use previous results from step {prev_result.step_code} because it failed: {prev_result.error_message}")
        return previous_results
    
    async def run_analysis(self, db: Session, analysis_run_id: str) -> None:
        """Execute all steps in an analysis."""
        try:
            # Get analysis run
            analysis_run = crud_analysis_execution.analysis_run.get(db, id=analysis_run_id)
            if not analysis_run:
                logger.error(f"Analysis run not found: {analysis_run_id}")
                return
            
            # Update status to in_progress
            analysis_update = AnalysisRunUpdate(
                status=AnalysisStatus.IN_PROGRESS,
                started_at=datetime.utcnow()
            )
            analysis_run = crud_analysis_execution.analysis_run.update(
                db,
                db_obj=analysis_run,
                obj_in=analysis_update
            )
            
            # Get registered steps and their order
            registered_steps = AnalysisRegistry.list_steps(analysis_run.analysis_code)
            step_order = {
                f"{analysis_run.analysis_code}.{step.code}": step.order 
                for step in registered_steps
            }
            
            # Sort step results based on registered step order
            step_results = sorted(
                analysis_run.step_results,
                key=lambda x: step_order.get(x.step_code, float('inf'))
            )
            
            for step_result in step_results:
                await self.run_step(db, str(step_result.id))
                
                # If step failed, stop processing
                if step_result.status == AnalysisStatus.FAILED:
                    analysis_update = AnalysisRunUpdate(
                        status=AnalysisStatus.FAILED,
                        error_message=f"Step {step_result.step_code} failed: {step_result.error_message}",
                        completed_at=datetime.utcnow()
                    )
                    crud_analysis_execution.analysis_run.update(
                        db,
                        db_obj=analysis_run,
                        obj_in=analysis_update
                    )
                    return
            
            # Update analysis status
            self._update_analysis_status(db, analysis_run)
            
        except Exception as e:
            logger.error(f"Error running analysis {analysis_run_id}: {str(e)}")
            if analysis_run:
                analysis_update = AnalysisRunUpdate(
                    status=AnalysisStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )
                crud_analysis_execution.analysis_run.update(
                    db,
                    db_obj=analysis_run,
                    obj_in=analysis_update
                )
    
    def _update_analysis_status(self, db: Session, analysis_run: AnalysisRunInfo) -> None:
        """Update analysis status based on step results."""
        all_completed = all(r.status == AnalysisStatus.COMPLETED for r in analysis_run.step_results)
        any_failed = any(r.status == AnalysisStatus.FAILED for r in analysis_run.step_results)
        
        if any_failed:
            status = AnalysisStatus.FAILED
            error_message = next(
                (r.error_message for r in analysis_run.step_results if r.status == AnalysisStatus.FAILED),
                None
            )
        elif all_completed:
            status = AnalysisStatus.COMPLETED
            error_message = None
        else:
            status = AnalysisStatus.IN_PROGRESS
            error_message = None
        
        analysis_update = AnalysisRunUpdate(
            status=status,
            error_message=error_message,
            completed_at=datetime.utcnow() if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED] else None
        )
        crud_analysis_execution.analysis_run.update(
            db,
            db_obj=analysis_run,
            obj_in=analysis_update
        ) 