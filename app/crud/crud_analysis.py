import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid
from datetime import datetime

from app.crud.base import CRUDBase
from app.db.models.analysis_result import AnalysisResult
from app.schemas.analysis import AnalysisType, AnalysisResultCreate, AnalysisResultUpdate

logger = logging.getLogger(__name__)

class CRUDAnalysisResult(CRUDBase[AnalysisResult, AnalysisResultCreate, AnalysisResultUpdate]):
    def create_result(
        self, 
        db: Session, 
        *, 
        document_id: str, 
        analysis_type: AnalysisType,
        result: Dict[str, Any]
    ) -> AnalysisResult:
        """Create a new analysis result with proper error handling."""
        try:
            db_obj = AnalysisResult(
                id=str(uuid.uuid4()),
                document_id=document_id,
                type=analysis_type,
                result=result,
                created_at=datetime.utcnow()
            )
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating analysis result: {str(e)}")
            raise

    def get_document_results(
        self, 
        db: Session, 
        *, 
        document_id: str,
        analysis_type: Optional[AnalysisType] = None
    ) -> List[AnalysisResult]:
        """Get all analysis results for a document with type filtering."""
        try:
            query = (
                db.query(self.model)
                .filter(AnalysisResult.document_id == document_id)
            )
            
            if analysis_type:
                query = query.filter(AnalysisResult.type == analysis_type)
                
            return query.order_by(desc(AnalysisResult.created_at)).all()
        except Exception as e:
            logger.error(f"Error fetching analysis results: {str(e)}")
            raise

    def get_latest_result(
        self,
        db: Session,
        *,
        document_id: str,
        analysis_type: AnalysisType
    ) -> Optional[AnalysisResult]:
        """Get the latest analysis result of a specific type for a document."""
        try:
            return (
                db.query(self.model)
                .filter(
                    AnalysisResult.document_id == document_id,
                    AnalysisResult.type == analysis_type
                )
                .order_by(desc(AnalysisResult.created_at))
                .first()
            )
        except Exception as e:
            logger.error(f"Error fetching latest analysis result: {str(e)}")
            raise

    def delete_document_results(
        self,
        db: Session,
        *,
        document_id: str
    ) -> int:
        """Delete all analysis results for a document."""
        try:
            result = (
                db.query(self.model)
                .filter(AnalysisResult.document_id == document_id)
                .delete()
            )
            db.commit()
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting analysis results: {str(e)}")
            raise


analysis_result = CRUDAnalysisResult(AnalysisResult) 