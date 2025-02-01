from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import logging

from app.crud.base import CRUDBase
from app.db.models.analysis import (
    AnalysisType, 
    AnalysisStep, 
    Algorithm, 
    Analysis,
    AnalysisStepResult
)
from app.schemas.analysis import (
    AnalysisTypeCreate,
    AnalysisTypeUpdate,
    AnalysisStepCreate,
    AnalysisStepUpdate,
    AnalysisCreate,
    AnalysisUpdate,
    AnalysisStepResultCreate,
    AnalysisStepResultUpdate
)

from app.schemas.algorithm import AlgorithmCreate, AlgorithmUpdate

logger = logging.getLogger(__name__)

class CRUDAnalysisType(CRUDBase[AnalysisType, AnalysisTypeCreate, AnalysisTypeUpdate]):
    def get_with_steps(self, db: Session, id: str) -> Optional[AnalysisType]:
        return db.query(self.model).filter(self.model.id == id).first()

    def get_by_name(self, db: Session, name: str) -> Optional[AnalysisType]:
        return db.query(self.model).filter(self.model.name == name).first()

class CRUDAnalysisStep(CRUDBase[AnalysisStep, AnalysisStepCreate, AnalysisStepUpdate]):
    def get_by_analysis_type(
        self, db: Session, analysis_type_id: str
    ) -> List[AnalysisStep]:
        return (
            db.query(self.model)
            .filter(self.model.analysis_type_id == analysis_type_id)
            .order_by(self.model.order)
            .all()
        )

    def get_by_name(
        self, db: Session, name: str, analysis_type_id: str
    ) -> Optional[AnalysisStep]:
        return (
            db.query(self.model)
            .filter(
                self.model.name == name,
                self.model.analysis_type_id == analysis_type_id
            )
            .first()
        )

class CRUDAlgorithm(CRUDBase[Algorithm, AlgorithmCreate, AlgorithmUpdate]):
    def get_by_step(self, db: Session, step_id: str) -> List[Algorithm]:
        return (
            db.query(self.model)
            .filter(
                self.model.step_id == step_id,
                self.model.is_active == True
            )
            .all()
        )

    def get_by_name_and_version(
        self, db: Session, name: str, version: str
    ) -> Optional[Algorithm]:
        return (
            db.query(self.model)
            .filter(
                self.model.name == name,
                self.model.version == version
            )
            .first()
        )

class CRUDAnalysis(CRUDBase[Analysis, AnalysisCreate, AnalysisUpdate]):
    def create_with_steps(
        self,
        db: Session,
        *,
        obj_in: AnalysisCreate,
        algorithm_configs: Dict[str, Dict[str, Any]]
    ) -> Analysis:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.flush()  # Get the ID without committing

        # Get all steps for this analysis type
        steps = db.query(AnalysisStep).filter(
            AnalysisStep.analysis_type_id == str(obj_in.analysis_type_id)
        ).order_by(AnalysisStep.order).all()

        # Create step results
        for step in steps:
            step_config = algorithm_configs.get(str(step.id), {})
            
            # If no algorithm specified, get the first active algorithm for this step
            algorithm_id = step_config.get("algorithm_id")
            if not algorithm_id:
                default_algorithm = db.query(Algorithm).filter(
                    Algorithm.step_id == step.id,
                    Algorithm.is_active == True
                ).first()
                if default_algorithm:
                    algorithm_id = default_algorithm.id
                    
                    # Get default parameters from the algorithm
                    parameters = {}
                    for param in default_algorithm.parameters:
                        if isinstance(param, dict) and param.get('default') is not None:
                            parameters[param['name']] = param['default']
                else:
                    # Log warning if no active algorithm found
                    logger.warning(f"No active algorithm found for step {step.id}")
                    continue
            else:
                # If algorithm is specified but no parameters, get defaults
                algorithm = db.query(Algorithm).filter(
                    Algorithm.id == algorithm_id,
                    Algorithm.is_active == True
                ).first()
                if algorithm:
                    parameters = step_config.get("parameters", {})
                    # Add any missing parameters with their defaults
                    for param in algorithm.parameters:
                        if isinstance(param, dict) and param['name'] not in parameters and param.get('default') is not None:
                            parameters[param['name']] = param['default']
                else:
                    # Log warning if specified algorithm not found or not active
                    logger.warning(f"Specified algorithm {algorithm_id} not found or not active")
                    continue

            step_result = AnalysisStepResult(
                analysis_id=db_obj.id,
                step_id=step.id,
                algorithm_id=algorithm_id,
                parameters=parameters,
                status="pending"
            )
            db.add(step_result)

        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_document(
        self, db: Session, document_id: str
    ) -> List[Analysis]:
        return (
            db.query(self.model)
            .filter(self.model.document_id == document_id)
            .all()
        )

    def update_status(
        self,
        db: Session,
        *,
        db_obj: Analysis,
        status: str,
        error_message: Optional[str] = None
    ) -> Analysis:
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
    ) -> List[Analysis]:
        """
        Get multiple analysis records with filtering options.
        
        Args:
            db: Database session
            filters: Dictionary of filter conditions
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of Analysis objects matching the filter criteria
        """
        query = db.query(self.model)
        
        # Join with Document if we need to filter by document type
        if "document_type" in filters:
            query = query.join(Analysis.document)
        
        # Build filter conditions
        conditions = []
        
        if "user_id" in filters:
            conditions.append(Analysis.document.has(user_id=filters["user_id"]))
            
        if "status" in filters:
            conditions.append(Analysis.status == filters["status"])
            
        if "analysis_type_id" in filters:
            conditions.append(Analysis.analysis_type_id == filters["analysis_type_id"])
            
        if "document_type" in filters:
            conditions.append(Analysis.document.has(type=filters["document_type"]))
            
        if "start_date" in filters:
            conditions.append(Analysis.created_at >= filters["start_date"])
            
        if "end_date" in filters:
            conditions.append(Analysis.created_at <= filters["end_date"])
            
        # Apply all conditions
        if conditions:
            query = query.filter(and_(*conditions))
            
        # Order by creation date (newest first)
        query = query.order_by(desc(Analysis.created_at))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        return query.all()

class CRUDAnalysisStepResult(CRUDBase[AnalysisStepResult, AnalysisStepResultCreate, AnalysisStepResultUpdate]):
    def get_by_analysis(
        self, db: Session, analysis_id: str
    ) -> List[AnalysisStepResult]:
        return (
            db.query(self.model)
            .filter(self.model.analysis_id == analysis_id)
            .all()
        )

    def update_result(
        self,
        db: Session,
        *,
        db_obj: AnalysisStepResult,
        result: Dict[str, Any],
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> AnalysisStepResult:
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
        db_obj: AnalysisStepResult,
        corrections: Dict[str, Any]
    ) -> AnalysisStepResult:
        db_obj.user_corrections = corrections
        db_obj.updated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

analysis_type = CRUDAnalysisType(AnalysisType)
analysis_step = CRUDAnalysisStep(AnalysisStep)
algorithm = CRUDAlgorithm(Algorithm)
analysis = CRUDAnalysis(Analysis)
analysis_step_result = CRUDAnalysisStepResult(AnalysisStepResult) 