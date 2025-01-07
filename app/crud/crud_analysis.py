import logging
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid
from datetime import datetime
import json

from app.crud.base import CRUDBase
from app.db.models.analysis_result import AnalysisResult
from app.schemas.analysis import (
    AnalysisType,
    AnalysisStatus,
    AnalysisResultCreate,
    AnalysisResultUpdate,
    BatchAnalysisDocument
)

logger = logging.getLogger(__name__)

class CRUDAnalysisResult(CRUDBase[AnalysisResult, AnalysisResultCreate, AnalysisResultUpdate]):
    def create(self, db: Session, *, obj_in: AnalysisResultCreate) -> AnalysisResult:
        """Create a new analysis result."""
        db_obj = AnalysisResult(
            id=obj_in.id,
            document_id=obj_in.document_id,
            type=obj_in.type,
            status=obj_in.status,
            parameters=obj_in.parameters,
            progress=obj_in.progress,
            created_at=obj_in.created_at
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, *, analysis_id: str, obj_in: Union[AnalysisResultUpdate, Dict[str, Any]]) -> AnalysisResult:
        """Update an analysis result."""
        db_obj = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
        if not db_obj:
            return None
            
        # Convert input to dictionary if it's a Pydantic model
        update_data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
            
        for field, value in update_data.items():
            setattr(db_obj, field, value)
            
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_filter(
        self,
        db: Session,
        *,
        filters: Dict[str, Any],
        skip: int = 0,
        limit: int = 100
    ) -> List[AnalysisResult]:
        """Get multiple analysis results with filters."""
        query = db.query(AnalysisResult)
        for field, value in filters.items():
            query = query.filter(getattr(AnalysisResult, field) == value)
        return query.offset(skip).limit(limit).all()

    def create_result(
        self, 
        db: Session, 
        *, 
        document_id: str, 
        analysis_type: AnalysisType,
        parameters: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Create a new analysis result with proper error handling."""
        try:
            db_obj = AnalysisResult(
                id=str(uuid.uuid4()),
                document_id=document_id,
                type=analysis_type,
                status=AnalysisStatus.PENDING,
                parameters=parameters,
                result=result,
                progress=0.0,
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

    def create_batch(
        self,
        db: Session,
        *,
        batch_documents: List[BatchAnalysisDocument]
    ) -> List[AnalysisResult]:
        """Create multiple analysis results in a batch."""
        results = []
        try:
            for doc in batch_documents:
                result = self.create_result(
                    db,
                    document_id=doc.document_id,
                    analysis_type=doc.analysis_type,
                    parameters=doc.parameters
                )
                results.append(result)
            return results
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating batch analysis results: {str(e)}")
            raise

    def get_document_results(
        self, 
        db: Session, 
        *, 
        document_id: str,
        analysis_type: Optional[AnalysisType] = None,
        status: Optional[AnalysisStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AnalysisResult]:
        """Get all analysis results for a document with type and status filtering."""
        try:
            query = (
                db.query(self.model)
                .filter(AnalysisResult.document_id == document_id)
            )
            
            if analysis_type:
                query = query.filter(AnalysisResult.type == analysis_type)
                
            if status:
                query = query.filter(AnalysisResult.status == status)
                
            return (
                query.order_by(desc(AnalysisResult.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
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

    def update_progress(
        self,
        db: Session,
        *,
        analysis_id: str,
        progress: float,
        status_message: Optional[str] = None
    ) -> Optional[AnalysisResult]:
        """Update analysis progress and status message."""
        try:
            analysis = self.get(db, id=analysis_id)
            if not analysis:
                logger.warning(f"Analysis not found: {analysis_id}")
                return None

            update_data = {
                "progress": min(max(progress, 0.0), 100.0),
                "status_message": status_message
            }

            if progress >= 100.0:
                update_data["status"] = AnalysisStatus.COMPLETED
                update_data["completed_at"] = datetime.utcnow()

            return self.update(db, db_obj=analysis, obj_in=update_data)
        except Exception as e:
            logger.error(f"Error updating analysis progress: {str(e)}")
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

    def get_pending_analyses(
        self,
        db: Session,
        *,
        analysis_type: Optional[AnalysisType] = None,
        limit: int = 10
    ) -> List[AnalysisResult]:
        """Get pending analyses for processing."""
        try:
            query = (
                db.query(self.model)
                .filter(AnalysisResult.status == AnalysisStatus.PENDING)
            )
            
            if analysis_type:
                query = query.filter(AnalysisResult.type == analysis_type)
                
            return (
                query.order_by(AnalysisResult.created_at)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Error fetching pending analyses: {str(e)}")
            raise


analysis_result = CRUDAnalysisResult(AnalysisResult) 