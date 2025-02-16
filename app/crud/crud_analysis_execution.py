from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import logging

from app.crud.base import CRUDBase
from app.db.models.analysis_execution import AnalysisRun, StepExecutionResult
from app.db.models.analysis_config import StepDefinition, AlgorithmDefinition
from app.analysis.schemas.types import Analysis as AnalysisSchema, AnalysisConfig
from app.analysis.schemas.steps import AnalysisStepResult as StepExecutionResultSchema

logger = logging.getLogger(__name__)

class CRUDAnalysisRun(CRUDBase[AnalysisRun, AnalysisSchema, AnalysisConfig]):
    def create_with_steps(
        self,
        db: Session,
        *,
        obj_in: AnalysisSchema,
        algorithm_configs: Dict[str, Dict[str, Any]]
    ) -> AnalysisRun:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.flush()  # Get the ID without committing

        # Get all active steps for this analysis definition
        steps = db.query(StepDefinition).filter(
            StepDefinition.analysis_definition_id == str(obj_in.analysis_definition_id),
            StepDefinition.is_active == True
        ).order_by(StepDefinition.order).all()

        # Create step results
        for step in steps:
            step_config = algorithm_configs.get(str(step.id), {})
            
            # Get algorithm based on configuration
            algorithm_id = None
            parameters = {}
            
            if "algorithm_code" in step_config and "algorithm_version" in step_config:
                # Find algorithm by code and version
                algorithm = db.query(AlgorithmDefinition).filter(
                    AlgorithmDefinition.step_id == step.id,
                    AlgorithmDefinition.code == step_config["algorithm_code"],
                    AlgorithmDefinition.version == step_config["algorithm_version"],
                    AlgorithmDefinition.is_active == True
                ).first()
                if algorithm:
                    algorithm_id = algorithm.id
                    parameters = step_config.get("parameters", {})
                    # Add any missing parameters with their defaults
                    for param in algorithm.parameters:
                        if isinstance(param, dict) and param['name'] not in parameters and param.get('default') is not None:
                            parameters[param['name']] = param['default']
                else:
                    logger.warning(
                        f"Specified algorithm {step_config['algorithm_code']} "
                        f"v{step_config['algorithm_version']} not found or not active"
                    )
            
            # If no algorithm specified or found, get the default
            if not algorithm_id:
                default_algorithm = db.query(AlgorithmDefinition).filter(
                    AlgorithmDefinition.step_id == step.id,
                    AlgorithmDefinition.is_active == True
                ).first()
                
                if default_algorithm:
                    algorithm_id = default_algorithm.id
                    # Get default parameters
                    parameters = {}
                    for param in default_algorithm.parameters:
                        if isinstance(param, dict) and param.get('default') is not None:
                            parameters[param['name']] = param['default']
                else:
                    logger.warning(f"No active algorithm found for step {step.id}")
                    continue

            # Create step result
            step_result = StepExecutionResult(
                analysis_run_id=db_obj.id,
                step_definition_id=step.id,
                algorithm_definition_id=algorithm_id,
                parameters=parameters,
                status="pending"
            )
            db.add(step_result)

        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_document(
        self, db: Session, document_id: str
    ) -> List[AnalysisRun]:
        return (
            db.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(desc(self.model.created_at))
            .all()
        )

    def update_status(
        self,
        db: Session,
        *,
        db_obj: AnalysisRun,
        status: str,
        error_message: Optional[str] = None
    ) -> AnalysisRun:
        db_obj.status = status
        if status == "completed":
            db_obj.completed_at = datetime.utcnow()
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
        query = db.query(self.model)
        
        # Join with Document if we need to filter by document type
        if "document_type" in filters:
            query = query.join(AnalysisRun.document)
            
        # Join with AnalysisDefinition if we need to filter by analysis type code
        if "analysis_type_code" in filters:
            query = query.join(AnalysisRun.analysis_definition)
        
        # Build filter conditions
        conditions = []
        
        if "user_id" in filters:
            conditions.append(AnalysisRun.document.has(user_id=filters["user_id"]))
            
        if "status" in filters:
            conditions.append(AnalysisRun.status == filters["status"])
            
        if "analysis_definition_id" in filters:
            conditions.append(AnalysisRun.analysis_definition_id == filters["analysis_definition_id"])
            
        if "analysis_type_code" in filters:
            conditions.append(AnalysisRun.analysis_definition.has(code=filters["analysis_type_code"]))
            
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

class CRUDStepExecutionResult(CRUDBase[StepExecutionResult, StepExecutionResultSchema, StepExecutionResultSchema]):
    def get_by_analysis_run(
        self, db: Session, analysis_run_id: str
    ) -> List[StepExecutionResult]:
        return (
            db.query(self.model)
            .join(StepDefinition)
            .filter(self.model.analysis_run_id == analysis_run_id)
            .order_by(StepDefinition.order)
            .all()
        )

    def update_result(
        self,
        db: Session,
        *,
        db_obj: StepExecutionResult,
        result: Dict[str, Any],
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> StepExecutionResult:
        db_obj.result = result
        db_obj.status = status
        if status == "completed":
            db_obj.completed_at = datetime.utcnow()
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
        db_obj.user_corrections = corrections
        db_obj.updated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

analysis_run = CRUDAnalysisRun(AnalysisRun)
step_execution_result = CRUDStepExecutionResult(StepExecutionResult)
