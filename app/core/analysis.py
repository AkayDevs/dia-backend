from typing import Dict, Any, Optional, Type
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import importlib
import inspect

from app.crud import crud_analysis, crud_document
from app.db.models.analysis import Analysis, AnalysisStepResult
from app.core.config import settings

logger = logging.getLogger(__name__)

class AnalysisPlugin:
    """Base class for analysis plugins."""
    
    @classmethod
    def get_name(cls) -> str:
        return cls.__name__
    
    @classmethod
    def get_version(cls) -> str:
        return getattr(cls, "VERSION", "1.0.0")
    
    @classmethod
    def get_supported_document_types(cls) -> list:
        return getattr(cls, "SUPPORTED_DOCUMENT_TYPES", [])
    
    @classmethod
    def get_parameters(cls) -> list:
        return getattr(cls, "PARAMETERS", [])
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters against plugin's parameter definitions."""
        pass
    
    async def execute(self, document_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis step on the document."""
        raise NotImplementedError

class AnalysisOrchestrator:
    """Orchestrates the execution of analysis steps."""
    
    def __init__(self):
        self.plugins: Dict[str, Type[AnalysisPlugin]] = {}
        self._load_plugins()
    
    def _load_plugins(self) -> None:
        """Load all available analysis plugins."""
        try:
            # Import all modules in the plugins directory
            plugins_module = importlib.import_module("app.plugins")
            
            # Find all plugin classes
            for module_name in dir(plugins_module):
                if module_name.startswith("_"):
                    continue
                
                module = getattr(plugins_module, module_name)
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, AnalysisPlugin) and 
                        obj != AnalysisPlugin):
                        plugin_key = f"{obj.get_name()}_{obj.get_version()}"
                        self.plugins[plugin_key] = obj
                        
            logger.info(f"Loaded {len(self.plugins)} analysis plugins")
            
        except Exception as e:
            logger.error(f"Error loading plugins: {str(e)}")
            raise
    
    def _get_plugin(self, algorithm_id: str, db: Session) -> Optional[Type[AnalysisPlugin]]:
        """Get plugin class for an algorithm."""
        algorithm = crud_analysis.algorithm.get(db, id=algorithm_id)
        if not algorithm:
            return None
        
        plugin_key = f"{algorithm.name}_{algorithm.version}"
        return self.plugins.get(plugin_key)
    
    async def run_step(self, db: Session, step_result_id: str) -> None:
        """Execute a single analysis step."""
        try:
            # Get step result
            step_result = crud_analysis.analysis_step_result.get(db, id=step_result_id)
            if not step_result:
                logger.error(f"Step result not found: {step_result_id}")
                return
            
            # Update status to in_progress
            step_result.status = "in_progress"
            step_result.updated_at = datetime.utcnow()
            db.add(step_result)
            db.commit()
            
            # Get document path
            analysis = crud_analysis.analysis.get(db, id=step_result.analysis_id)
            document = crud_document.document.get(db, id=analysis.document_id)
            document_path = document.url.replace("/uploads/", "")
            
            # Get and initialize plugin
            plugin_class = self._get_plugin(step_result.algorithm_id, db)
            if not plugin_class:
                raise Exception(f"Plugin not found for algorithm: {step_result.algorithm_id}")
            
            plugin = plugin_class()
            
            # Validate parameters
            plugin.validate_parameters(step_result.parameters)
            
            # Execute plugin
            result = await plugin.execute(document_path, step_result.parameters)
            
            # Update step result
            crud_analysis.analysis_step_result.update_result(
                db,
                db_obj=step_result,
                result=result,
                status="completed"
            )
            
            # Update analysis status if all steps are complete
            self._update_analysis_status(db, analysis)
            
        except Exception as e:
            logger.error(f"Error executing step {step_result_id}: {str(e)}")
            if step_result:
                crud_analysis.analysis_step_result.update_result(
                    db,
                    db_obj=step_result,
                    result={},
                    status="failed",
                    error_message=str(e)
                )
                self._update_analysis_status(db, analysis)
    
    async def run_analysis(self, db: Session, analysis_id: str) -> None:
        """Execute all steps in an analysis."""
        try:
            # Get analysis
            analysis = crud_analysis.analysis.get(db, id=analysis_id)
            if not analysis:
                logger.error(f"Analysis not found: {analysis_id}")
                return
            
            # Update status to in_progress
            crud_analysis.analysis.update_status(
                db,
                db_obj=analysis,
                status="in_progress"
            )
            
            # Execute steps in order
            step_results = sorted(
                analysis.step_results,
                key=lambda x: x.step.order
            )
            
            for step_result in step_results:
                await self.run_step(db, str(step_result.id))
                
                # If step failed, stop processing
                if step_result.status == "failed":
                    crud_analysis.analysis.update_status(
                        db,
                        db_obj=analysis,
                        status="failed",
                        error_message=f"Step {step_result.step.name} failed: {step_result.error_message}"
                    )
                    return
            
            # Update analysis status
            self._update_analysis_status(db, analysis)
            
        except Exception as e:
            logger.error(f"Error running analysis {analysis_id}: {str(e)}")
            if analysis:
                crud_analysis.analysis.update_status(
                    db,
                    db_obj=analysis,
                    status="failed",
                    error_message=str(e)
                )
    
    def _update_analysis_status(self, db: Session, analysis: Analysis) -> None:
        """Update analysis status based on step results."""
        all_completed = all(r.status == "completed" for r in analysis.step_results)
        any_failed = any(r.status == "failed" for r in analysis.step_results)
        
        if any_failed:
            status = "failed"
            error_message = next(
                (r.error_message for r in analysis.step_results if r.status == "failed"),
                None
            )
        elif all_completed:
            status = "completed"
            error_message = None
        else:
            status = "in_progress"
            error_message = None
        
        crud_analysis.analysis.update_status(
            db,
            db_obj=analysis,
            status=status,
            error_message=error_message
        ) 