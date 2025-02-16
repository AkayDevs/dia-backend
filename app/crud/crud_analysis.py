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
from app.analysis.schemas.types import (
    AnalysisTypeCreate,
    AnalysisTypeUpdate,
    Analysis as AnalysisSchema,
    AnalysisConfig
)
from app.analysis.schemas.steps import (
    AnalysisStepCreate,
    AnalysisStepUpdate,
    AnalysisStepResult as AnalysisStepResultSchema
)
from app.analysis.schemas.algorithms import (
    AlgorithmCreate,
    AlgorithmUpdate
)

logger = logging.getLogger(__name__)

class CRUDAnalysisType(CRUDBase[AnalysisType, AnalysisTypeCreate, AnalysisTypeUpdate]):
    def get_with_steps(self, db: Session, id: str) -> Optional[AnalysisType]:
        return db.query(self.model).filter(self.model.id == id).first()

    def get_by_code(self, db: Session, code: str) -> Optional[AnalysisType]:
        return db.query(self.model).filter(self.model.code == code).first()
    
    def get_by_code_and_version(
        self, db: Session, code: str, version: str
    ) -> Optional[AnalysisType]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.version == version
            )
            .first()
        )
    
    def get_active_types(self, db: Session) -> List[AnalysisType]:
        return (
            db.query(self.model)
            .filter(self.model.is_active == True)
            .all()
        )

class CRUDAnalysisStep(CRUDBase[AnalysisStep, AnalysisStepCreate, AnalysisStepUpdate]):
    def get_by_analysis_type(
        self, db: Session, analysis_type_id: str
    ) -> List[AnalysisStep]:
        return (
            db.query(self.model)
            .filter(
                self.model.analysis_type_id == analysis_type_id,
                self.model.is_active == True
            )
            .order_by(self.model.order)
            .all()
        )

    def get_by_code(
        self, db: Session, code: str, analysis_type_id: str
    ) -> Optional[AnalysisStep]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.analysis_type_id == analysis_type_id,
                self.model.is_active == True
            )
            .first()
        )
    
    def get_by_code_and_version(
        self, db: Session, code: str, version: str, analysis_type_id: str
    ) -> Optional[AnalysisStep]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.version == version,
                self.model.analysis_type_id == analysis_type_id,
                self.model.is_active == True
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

    def get_by_code_and_version(
        self, db: Session, code: str, version: str, step_id: str
    ) -> Optional[Algorithm]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.version == version,
                self.model.step_id == step_id,
                self.model.is_active == True
            )
            .first()
        )
    
    def get_default_for_step(
        self, db: Session, step_id: str
    ) -> Optional[Algorithm]:
        """Get the default (first active) algorithm for a step"""
        return (
            db.query(self.model)
            .filter(
                self.model.step_id == step_id,
                self.model.is_active == True
            )
            .first()
        )

class CRUDAnalysis(CRUDBase[Analysis, AnalysisSchema, AnalysisConfig]):
    def create_with_steps(
        self,
        db: Session,
        *,
        obj_in: AnalysisSchema,
        algorithm_configs: Dict[str, Dict[str, Any]]
    ) -> Analysis:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.flush()  # Get the ID without committing

        # Get all active steps for this analysis type
        steps = db.query(AnalysisStep).filter(
            AnalysisStep.analysis_type_id == str(obj_in.analysis_type_id),
            AnalysisStep.is_active == True
        ).order_by(AnalysisStep.order).all()

        # Create step results
        for step in steps:
            step_config = algorithm_configs.get(str(step.id), {})
            
            # Get algorithm based on configuration
            algorithm_id = None
            parameters = {}
            
            if "algorithm_code" in step_config and "algorithm_version" in step_config:
                # Find algorithm by code and version
                algorithm = db.query(Algorithm).filter(
                    Algorithm.step_id == step.id,
                    Algorithm.code == step_config["algorithm_code"],
                    Algorithm.version == step_config["algorithm_version"],
                    Algorithm.is_active == True
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
                default_algorithm = db.query(Algorithm).filter(
                    Algorithm.step_id == step.id,
                    Algorithm.is_active == True
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
            .order_by(desc(self.model.created_at))
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
        query = db.query(self.model)
        
        # Join with Document if we need to filter by document type
        if "document_type" in filters:
            query = query.join(Analysis.document)
            
        # Join with AnalysisType if we need to filter by analysis type code
        if "analysis_type_code" in filters:
            query = query.join(Analysis.analysis_type)
        
        # Build filter conditions
        conditions = []
        
        if "user_id" in filters:
            conditions.append(Analysis.document.has(user_id=filters["user_id"]))
            
        if "status" in filters:
            conditions.append(Analysis.status == filters["status"])
            
        if "analysis_type_id" in filters:
            conditions.append(Analysis.analysis_type_id == filters["analysis_type_id"])
            
        if "analysis_type_code" in filters:
            conditions.append(Analysis.analysis_type.has(code=filters["analysis_type_code"]))
            
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

class CRUDAnalysisStepResult(CRUDBase[AnalysisStepResult, AnalysisStepResultSchema, AnalysisStepResultSchema]):
    def get_by_analysis(
        self, db: Session, analysis_id: str
    ) -> List[AnalysisStepResult]:
        return (
            db.query(self.model)
            .join(AnalysisStep)
            .filter(self.model.analysis_id == analysis_id)
            .order_by(AnalysisStep.order)
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