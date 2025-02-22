from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging

from app.crud.base import CRUDBase
from app.db.models.analysis_config import (
    AnalysisDefinition,
    StepDefinition,
    AlgorithmDefinition
)
from app.schemas.analysis.configs.definitions import (
    AnalysisDefinitionCreate,
    AnalysisDefinitionUpdate
)
from app.schemas.analysis.configs.steps import (
    StepDefinitionCreate,
    StepDefinitionUpdate
)
from app.schemas.analysis.configs.algorithms import (
    AlgorithmDefinitionCreate,
    AlgorithmDefinitionUpdate
)

logger = logging.getLogger(__name__)

class CRUDAnalysisDefinition(CRUDBase[AnalysisDefinition, AnalysisDefinitionCreate, AnalysisDefinitionUpdate]):
    def get_with_steps(self, db: Session, id: str) -> Optional[AnalysisDefinition]:
        return db.query(self.model).filter(self.model.id == id).first()

    def get_by_code(self, db: Session, code: str) -> Optional[AnalysisDefinition]:
        return db.query(self.model).filter(self.model.code == code).first()
    
    def get_by_code_and_version(
        self, db: Session, code: str, version: str
    ) -> Optional[AnalysisDefinition]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.version == version
            )
            .first()
        )
    
    def get_active_definitions(self, db: Session) -> List[AnalysisDefinition]:
        return (
            db.query(self.model)
            .filter(self.model.is_active == True)
            .all()
        )

class CRUDStepDefinition(CRUDBase[StepDefinition, StepDefinitionCreate, StepDefinitionUpdate]):
    def get_by_analysis_definition(
        self, db: Session, analysis_definition_id: str
    ) -> List[StepDefinition]:
        return (
            db.query(self.model)
            .filter(
                self.model.analysis_definition_id == analysis_definition_id,
                self.model.is_active == True
            )
            .order_by(self.model.order)
            .all()
        )

    def get_by_code(
        self, db: Session, code: str, analysis_definition_id: str
    ) -> Optional[StepDefinition]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.analysis_definition_id == analysis_definition_id,
                self.model.is_active == True
            )
            .first()
        )
    
    def get_by_code_and_version(
        self, db: Session, code: str, version: str, analysis_definition_id: str
    ) -> Optional[StepDefinition]:
        return (
            db.query(self.model)
            .filter(
                self.model.code == code,
                self.model.version == version,
                self.model.analysis_definition_id == analysis_definition_id,
                self.model.is_active == True
            )
            .first()
        )

class CRUDAlgorithmDefinition(CRUDBase[AlgorithmDefinition, AlgorithmDefinitionCreate, AlgorithmDefinitionUpdate]):
    def get_by_step(self, db: Session, step_id: str) -> List[AlgorithmDefinition]:
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
    ) -> Optional[AlgorithmDefinition]:
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
    ) -> Optional[AlgorithmDefinition]:
        """Get the default (first active) algorithm for a step"""
        return (
            db.query(self.model)
            .filter(
                self.model.step_id == step_id,
                self.model.is_active == True
            )
            .first()
        )

analysis_definition = CRUDAnalysisDefinition(AnalysisDefinition)
step_definition = CRUDStepDefinition(StepDefinition)
algorithm_definition = CRUDAlgorithmDefinition(AlgorithmDefinition)
