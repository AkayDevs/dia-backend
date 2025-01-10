import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid
from datetime import datetime

from app.crud.base import CRUDBase
from app.db.models.document import Document
from app.db.models.tag import Tag
from app.schemas.document import DocumentCreate, DocumentUpdate, DocumentType
from app.crud.crud_tag import tag as crud_tag

logger = logging.getLogger(__name__)

class CRUDDocument(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def create_with_user(
        self, 
        db: Session, 
        *, 
        obj_in: DocumentCreate, 
        user_id: str
    ) -> Document:
        """Create a new document with user ID and proper error handling."""
        try:
            db_obj = Document(
                id=str(uuid.uuid4()),
                name=obj_in.name,
                type=obj_in.type,
                size=obj_in.size,
                url=obj_in.url,
                user_id=user_id,
                uploaded_at=datetime.utcnow()
            )
            
            # Add default tag
            default_tag = crud_tag.get_default_tag(db)
            db_obj.tags = [default_tag]
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating document: {str(e)}")
            raise

    def get_multi_by_user(
        self,
        db: Session,
        *,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
        doc_type: Optional[DocumentType] = None
    ) -> List[Document]:
        """Get all documents for a user with type filtering."""
        try:
            query = db.query(self.model).filter(Document.user_id == user_id)
            
            if doc_type:
                query = query.filter(Document.type == doc_type)
            
            return (
                query.order_by(desc(Document.uploaded_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Error fetching user documents: {str(e)}")
            raise

    def get_document_with_results(
        self, 
        db: Session, 
        *, 
        document_id: str, 
        user_id: str
    ) -> Optional[Document]:
        """Get a document with its analysis results, ensuring user ownership."""
        try:
            return (
                db.query(self.model)
                .filter(
                    Document.id == document_id,
                    Document.user_id == user_id
                )
                .first()
            )
        except Exception as e:
            logger.error(f"Error fetching document with results: {str(e)}")
            raise

    def get_by_type(
        self,
        db: Session,
        *,
        doc_type: DocumentType,
        skip: int = 0,
        limit: int = 100
    ) -> List[Document]:
        """Get documents by type with pagination."""
        try:
            return (
                db.query(self.model)
                .filter(Document.type == doc_type)
                .order_by(desc(Document.uploaded_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Error fetching documents by type: {str(e)}")
            raise

    def update_with_tags(
        self,
        db: Session,
        *,
        db_obj: Document,
        obj_in: DocumentUpdate
    ) -> Document:
        """Update document with tags."""
        try:
            # Update basic fields
            update_data = obj_in.model_dump(exclude_unset=True)
            tag_ids = update_data.pop('tag_ids', None)
            
            # Update document fields
            for field in update_data:
                setattr(db_obj, field, update_data[field])
            
            # Update tags if provided
            if tag_ids is not None:
                if not tag_ids:  # Empty list
                    default_tag = crud_tag.get_default_tag(db)
                    db_obj.tags = [default_tag]
                else:
                    tags = db.query(Tag).filter(Tag.id.in_(tag_ids)).all()
                    if len(tags) != len(tag_ids):
                        raise ValueError("Some tag IDs are invalid")
                    db_obj.tags = tags
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating document: {str(e)}")
            raise

    def get_by_hash(
        self,
        db: Session,
        *,
        user_id: str,
        file_hash: str
    ) -> Optional[Document]:
        """Get a document by user ID and file hash."""
        try:
            return (
                db.query(self.model)
                .filter(
                    Document.user_id == user_id,
                    Document.file_hash == file_hash
                )
                .first()
            )
        except Exception as e:
            logger.error(f"Error fetching document by hash: {str(e)}")
            raise


document = CRUDDocument(Document) 