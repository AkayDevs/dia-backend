from typing import List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid

from app.crud.base import CRUDBase
from app.db.models.document import Document, AnalysisResult, AnalysisStatus
from app.schemas.document import DocumentCreate, DocumentUpdate


class CRUDDocument(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def create_with_user(
        self, db: Session, *, obj_in: DocumentCreate, user_id: str
    ) -> Document:
        """Create a new document with user ID."""
        db_obj = Document(
            id=str(uuid.uuid4()),
            **obj_in.model_dump(),
            user_id=user_id,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_user(
        self,
        db: Session,
        *,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
        status: Optional[AnalysisStatus] = None,
    ) -> List[Document]:
        """Get all documents for a user with optional status filter."""
        query = db.query(self.model).filter(Document.user_id == user_id)
        
        if status:
            query = query.filter(Document.status == status)
        
        return (
            query.order_by(desc(Document.uploaded_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_document_with_results(
        self, db: Session, *, document_id: str, user_id: str
    ) -> Optional[Document]:
        """Get a document with its analysis results."""
        return (
            db.query(self.model)
            .filter(
                Document.id == document_id,
                Document.user_id == user_id
            )
            .first()
        )

    def update_status(
        self, db: Session, *, document_id: str, status: AnalysisStatus
    ) -> Document:
        """Update document status."""
        document = db.query(self.model).filter(Document.id == document_id).first()
        if document:
            document.status = status
            db.add(document)
            db.commit()
            db.refresh(document)
        return document


class CRUDAnalysisResult(CRUDBase[AnalysisResult, Any, Any]):
    def create_result(
        self, db: Session, *, document_id: str, type: str, result: Any
    ) -> AnalysisResult:
        """Create a new analysis result."""
        db_obj = AnalysisResult(
            id=str(uuid.uuid4()),
            document_id=document_id,
            type=type,
            result=result,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_document_results(
        self, db: Session, *, document_id: str
    ) -> List[AnalysisResult]:
        """Get all analysis results for a document."""
        return (
            db.query(self.model)
            .filter(AnalysisResult.document_id == document_id)
            .order_by(desc(AnalysisResult.created_at))
            .all()
        )


document = CRUDDocument(Document)
analysis_result = CRUDAnalysisResult(AnalysisResult) 