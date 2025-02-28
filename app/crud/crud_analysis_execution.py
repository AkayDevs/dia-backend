from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import logging
from uuid import uuid4

from app.crud.base import CRUDBase
from app.db.models.analysis_execution import AnalysisRun, StepExecutionResult
from app.db.models.analysis_config import StepDefinition, AlgorithmDefinition
from app.schemas.analysis.executions.analysis_run import (
    AnalysisRunCreate,
    AnalysisRunUpdate,
    AnalysisRunInDB,
    AnalysisRunConfig,
    StepConfig,
)
from app.schemas.analysis.executions.step_result import (
    StepExecutionResultCreate,
    StepExecutionResultUpdate,
    StepExecutionResultInDB
)
from app.enums.analysis import AnalysisStatus, AnalysisMode
from app.schemas.analysis.results.base import BaseResultSchema
from app.services.analysis.configs.registry import AnalysisRegistry

logger = logging.getLogger(__name__)

class CRUDAnalysisRun(CRUDBase[AnalysisRun, AnalysisRunCreate, AnalysisRunUpdate]):
    """CRUD operations for analysis runs."""
    
    def create_with_steps(
        self,
        db: Session,
        *,
        obj_in: AnalysisRunCreate
    ) -> AnalysisRun:
        """
        Create an analysis run with its associated step results.
        The configuration in obj_in should be complete, with all defaults already applied.
        
        Args:
            db: Database session
            obj_in: Analysis run creation data with complete configuration
            
        Returns:
            Created analysis run with step results
        """
        # Create the analysis run
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(
            id=str(uuid4()),
            **obj_in_data,
            status=AnalysisStatus.PENDING
        )
        db.add(db_obj)
        db.flush()  # Get the ID without committing

        # Get steps from registry
        steps = AnalysisRegistry.list_steps(obj_in.analysis_code)
        
        # Create step results for each enabled step
        for step in steps:
            try:
                step_code = f"{obj_in.analysis_code}.{step.code}"
                step_config = obj_in.config.steps.get(step_code)
                
                # Skip if step is disabled or no config found
                if not step_config or not step_config.enabled:
                    logger.info(f"Skipping disabled or unconfigured step {step_code}")
                    continue

                # Create step result
                step_result = StepExecutionResult(
                    id=str(uuid4()),
                    analysis_run_id=db_obj.id,
                    step_code=step_code,
                    algorithm_code=step_config.algorithm.code if step_config.algorithm else None,
                    parameters=step_config.algorithm.parameters if step_config.algorithm else {},
                    status=AnalysisStatus.PENDING,
                    timeout=step_config.timeout,
                    retry_count=step_config.retry
                )
                db.add(step_result)

            except Exception as e:
                logger.error(f"Error creating step result for step {step_code}: {str(e)}")
                continue

        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_document(
        self, db: Session, document_id: str, skip: int = 0, limit: int = 100
    ) -> List[AnalysisRun]:
        """Get all analysis runs for a specific document."""
        return (
            db.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(desc(self.model.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_status(
        self,
        db: Session,
        *,
        db_obj: AnalysisRun,
        status: AnalysisStatus,
        error_message: Optional[str] = None
    ) -> AnalysisRun:
        """Update the status of an analysis run."""
        db_obj.status = status
        if status == AnalysisStatus.COMPLETED:
            db_obj.completed_at = datetime.utcnow()
        if status == AnalysisStatus.IN_PROGRESS and not db_obj.started_at:
            db_obj.started_at = datetime.utcnow()
        if error_message:
            db_obj.error_message = error_message
        db_obj.updated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_filters(
        self,
        db: Session,
        *,
        filters: Dict[str, Any],
        skip: int = 0,
        limit: int = 100
    ) -> List[AnalysisRun]:
        """Get multiple analysis runs with filters."""
        query = db.query(self.model)
        
        # Join with Document if we need to filter by document type
        if "document_type" in filters:
            query = query.join(AnalysisRun.document)
            
        # Build filter conditions
        conditions = []
        
        if "user_id" in filters:
            conditions.append(AnalysisRun.document.has(user_id=filters["user_id"]))
            
        if "status" in filters:
            conditions.append(AnalysisRun.status == filters["status"])
            
        if "analysis_code" in filters:
            conditions.append(AnalysisRun.analysis_code == filters["analysis_code"])
            
        if "document_type" in filters:
            conditions.append(AnalysisRun.document.has(type=filters["document_type"]))
            
        if "start_date" in filters:
            conditions.append(AnalysisRun.created_at >= filters["start_date"])
            
        if "end_date" in filters:
            conditions.append(AnalysisRun.created_at <= filters["end_date"])
            
        # Apply all conditions
        if conditions:
            query = query.filter(and_(*conditions))
            
        # Order by creation date (newest first)
        query = query.order_by(desc(AnalysisRun.created_at))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        return query.all()

class CRUDStepExecutionResult(CRUDBase[StepExecutionResult, StepExecutionResultCreate, StepExecutionResultUpdate]):
    """CRUD operations for step execution results."""
    
    def get_by_analysis_run(
        self, db: Session, analysis_run_id: str
    ) -> List[StepExecutionResult]:
        """Get all step results for a specific analysis run."""
        return (
            db.query(self.model)
            .filter(self.model.analysis_run_id == analysis_run_id)
            .order_by(self.model.created_at)
            .all()
        )

    def update_result(
        self,
        db: Session,
        *,
        db_obj: StepExecutionResult,
        result: Union[Dict[str, Any], BaseResultSchema],
        status: AnalysisStatus = AnalysisStatus.COMPLETED,
        error_message: Optional[str] = None
    ) -> StepExecutionResult:
        """Update the result of a step execution."""
        db_obj.result = result if isinstance(result, dict) else result.dict()
        db_obj.status = status
        if status == AnalysisStatus.COMPLETED:
            db_obj.completed_at = datetime.utcnow()
        if status == AnalysisStatus.IN_PROGRESS and not db_obj.started_at:
            db_obj.started_at = datetime.utcnow()
        if error_message:
            db_obj.error_message = error_message
        db_obj.updated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update_user_corrections(
        self,
        db: Session,
        *,
        db_obj: StepExecutionResult,
        corrections: Dict[str, Any]
    ) -> StepExecutionResult:
        """Update user corrections for a step result."""
        db_obj.user_corrections = corrections
        db_obj.updated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

analysis_run = CRUDAnalysisRun(AnalysisRun)
step_execution_result = CRUDStepExecutionResult(StepExecutionResult)

def import_class(path: str) -> Any:
    """
    Dynamically import a class from a string path.
    
    Args:
        path: Full path to the class (e.g., 'app.services.analysis.implementations.my_step.MyStep')
    
    Returns:
        The class object
    """
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        logger.error(f"Error importing class {path}: {str(e)}")
        raise ImportError(f"Could not import class {path}")
