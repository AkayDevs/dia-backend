from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder
from datetime import datetime

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
    AlgorithmCreate,
    AlgorithmUpdate,
    AnalysisCreate,
    AnalysisUpdate,
    AnalysisStepResultCreate,
    AnalysisStepResultUpdate
)

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
            AnalysisStep.analysis_type_id == obj_in.analysis_type_id
        ).order_by(AnalysisStep.order).all()

        # Create step results
        for step in steps:
            step_config = algorithm_configs.get(str(step.id), {})
            algorithm_id = step_config.get("algorithm_id")
            parameters = step_config.get("parameters", {})

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