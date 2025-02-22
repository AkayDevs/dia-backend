from typing import Dict, Any, Optional, Type, List
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import importlib
import inspect
import re

from app.crud import crud_analysis_config, crud_document
from app.schemas.analysis.executions import (
    AnalysisRunInfo,
    StepExecutionResultInfo,
    StepExecutionResultUpdate,
    AnalysisRunUpdate
)
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionInfo
from app.schemas.document import DocumentInfo
from app.core.config import settings
from app.enums.analysis import AnalysisStatus

logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """Orchestrates the execution of analysis steps."""
    
    def __init__(self):
        self.algorithm_implementations: Dict[str, Type] = {}
        self._load_implementations()
    
    def _load_implementations(self) -> None:
        """Load all available algorithm implementations."""
        try:
            logger.info("Starting to load algorithm implementations...")
            implementations_module = importlib.import_module("app.services.analysis.implementations")
            
            # Find all implementation classes
            for module_name in dir(implementations_module):
                if module_name.startswith("_"):
                    continue
                
                module = getattr(implementations_module, module_name)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        # The implementation key matches the implementation_path in AlgorithmDefinition
                        impl_key = f"{module_name}.{name}"
                        self.algorithm_implementations[impl_key] = obj
                        logger.info(f"Successfully loaded implementation: {impl_key}")
                        
            logger.info(f"Loaded {len(self.algorithm_implementations)} algorithm implementations")
            
        except Exception as e:
            logger.error(f"Error loading implementations: {str(e)}")
            raise
    
    def _get_implementation(self, algorithm: AlgorithmDefinitionInfo) -> Optional[Type]:
        """Get implementation class for an algorithm."""
        return self.algorithm_implementations.get(algorithm.implementation_path)
    
    async def run_step(self, db: Session, step_result_id: str) -> None:
        """Execute a single analysis step."""
        step_result: Optional[StepExecutionResultInfo] = None
        analysis_run: Optional[AnalysisRunInfo] = None
        
        try:
            # Get step result
            step_result = crud_analysis_config.step_execution_result.get(db, id=step_result_id)
            if not step_result:
                logger.error(f"Step result not found: {step_result_id}")
                return
            
            # Update status to in_progress
            step_result_update = StepExecutionResultUpdate(
                status=AnalysisStatus.IN_PROGRESS,
                started_at=datetime.utcnow()
            )
            step_result = crud_analysis_config.step_execution_result.update(
                db,
                db_obj=step_result,
                obj_in=step_result_update
            )
            
            # Get document path
            analysis_run = crud_analysis_config.analysis_run.get(db, id=step_result.analysis_run_id)
            document: DocumentInfo = crud_document.document.get(db, id=analysis_run.document_id)
            document_path = document.url.replace("/uploads/", "")
            
            # Get previous step results if they exist
            previous_results = {}
            if analysis_run.step_results:
                # Sort step results by step order
                sorted_results = sorted(
                    analysis_run.step_results,
                    key=lambda x: x.step_definition.order
                )
                
                # Find current step's position
                current_step_idx = next(
                    (i for i, r in enumerate(sorted_results) if r.id == step_result.id),
                    -1
                )
                
                # Get all completed previous steps' results
                for prev_result in sorted_results[:current_step_idx]:
                    if prev_result.status == AnalysisStatus.COMPLETED:
                        previous_results[prev_result.step_definition.name] = prev_result.result
            
            # Get algorithm definition and implementation
            algorithm = crud_analysis_config.algorithm_definition.get(
                db, id=step_result.algorithm_definition_id
            )
            if not algorithm:
                raise Exception(f"Algorithm not found: {step_result.algorithm_definition_id}")
            
            implementation_class = self._get_implementation(algorithm)
            if not implementation_class:
                raise Exception(f"Implementation not found for algorithm: {algorithm.implementation_path}")
            
            # Initialize implementation
            implementation = implementation_class()
            
            # Validate parameters against algorithm definition
            self._validate_parameters(step_result.parameters, algorithm.parameters)
            
            # Execute implementation
            result = await implementation.execute(
                document_path=document_path,
                parameters=step_result.parameters,
                previous_results=previous_results
            )
            
            # Update step result
            step_result_update = StepExecutionResultUpdate(
                status=AnalysisStatus.COMPLETED,
                result=result,
                completed_at=datetime.utcnow()
            )
            crud_analysis_config.step_execution_result.update(
                db,
                db_obj=step_result,
                obj_in=step_result_update
            )

            # Update analysis status if all steps are complete
            self._update_analysis_status(db, analysis_run)
            
        except Exception as e:
            logger.error(f"Error executing step {step_result_id}: {str(e)}")
            if step_result:
                step_result_update = StepExecutionResultUpdate(
                    status=AnalysisStatus.FAILED,
                    result={},
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )
                crud_analysis_config.step_execution_result.update(
                    db,
                    db_obj=step_result,
                    obj_in=step_result_update
                )
                if analysis_run:
                    self._update_analysis_status(db, analysis_run)
    
    def _validate_parameters(self, provided_params: Dict[str, Any], param_definitions: List[Dict[str, Any]]) -> None:
        """Validate parameters against their definitions."""
        for param_def in param_definitions:
            name = param_def["name"]
            required = param_def.get("required", True)
            
            if required and name not in provided_params:
                raise ValueError(f"Required parameter '{name}' not provided")
            
            if name in provided_params:
                value = provided_params[name]
                param_type = param_def.get("type", "string")
                
                # Type validation
                if param_type == "number" and not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{name}' must be a number")
                elif param_type == "string" and not isinstance(value, str):
                    raise ValueError(f"Parameter '{name}' must be a string")
                elif param_type == "boolean" and not isinstance(value, bool):
                    raise ValueError(f"Parameter '{name}' must be a boolean")
                
                # Constraints validation
                constraints = param_def.get("constraints", {})
                if constraints:
                    if param_type == "number":
                        if "min" in constraints and value < constraints["min"]:
                            raise ValueError(f"Parameter '{name}' must be >= {constraints['min']}")
                        if "max" in constraints and value > constraints["max"]:
                            raise ValueError(f"Parameter '{name}' must be <= {constraints['max']}")
                    elif param_type == "string":
                        if "pattern" in constraints and not re.match(constraints["pattern"], value):
                            raise ValueError(f"Parameter '{name}' must match pattern: {constraints['pattern']}")
                        if "min_length" in constraints and len(value) < constraints["min_length"]:
                            raise ValueError(f"Parameter '{name}' must be at least {constraints['min_length']} characters")
                        if "max_length" in constraints and len(value) > constraints["max_length"]:
                            raise ValueError(f"Parameter '{name}' must be at most {constraints['max_length']} characters")

    async def run_analysis(self, db: Session, analysis_run_id: str) -> None:
        """Execute all steps in an analysis."""
        try:
            # Get analysis run
            analysis_run = crud_analysis_config.analysis_run.get(db, id=analysis_run_id)
            if not analysis_run:
                logger.error(f"Analysis run not found: {analysis_run_id}")
                return
            
            # Update status to in_progress
            analysis_update = AnalysisRunUpdate(
                status=AnalysisStatus.IN_PROGRESS,
                started_at=datetime.utcnow()
            )
            analysis_run = crud_analysis_config.analysis_run.update(
                db,
                db_obj=analysis_run,
                obj_in=analysis_update
            )
            
            # Execute steps in order
            step_results = sorted(
                analysis_run.step_results,
                key=lambda x: x.step_definition.order
            )
            
            for step_result in step_results:
                await self.run_step(db, str(step_result.id))
                
                # If step failed, stop processing
                if step_result.status == AnalysisStatus.FAILED:
                    analysis_update = AnalysisRunUpdate(
                        status=AnalysisStatus.FAILED,
                        error_message=f"Step {step_result.step_definition.name} failed: {step_result.error_message}",
                        completed_at=datetime.utcnow()
                    )
                    crud_analysis_config.analysis_run.update(
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
                crud_analysis_config.analysis_run.update(
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
        crud_analysis_config.analysis_run.update(
            db,
            db_obj=analysis_run,
            obj_in=analysis_update
        ) 