import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid
from datetime import datetime

from app.crud.base import CRUDBase
from app.db.models.tag import Tag
from app.db.models.document import Document
from app.schemas.tag import TagCreate, TagUpdate

logger = logging.getLogger(__name__)

class CRUDTag(CRUDBase[Tag, TagCreate, TagUpdate]):
    def create(self, db: Session, *, obj_in: TagCreate) -> Tag:
        """Create a new tag."""
        try:
            db_obj = Tag(
                id=str(uuid.uuid4()),
                name=obj_in.name,
                created_at=datetime.utcnow()
            )
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating tag: {str(e)}")
            raise

    def get_by_name(self, db: Session, *, name: str) -> Optional[Tag]:
        """Get a tag by name."""
        try:
            return db.query(Tag).filter(Tag.name == name).first()
        except Exception as e:
            logger.error(f"Error fetching tag by name: {str(e)}")
            raise

    def get_or_create(self, db: Session, *, name: str) -> Tag:
        """Get an existing tag by name or create a new one."""
        try:
            tag = self.get_by_name(db, name=name)
            if tag:
                return tag
            
            return self.create(db, obj_in=TagCreate(name=name))
        except Exception as e:
            logger.error(f"Error in get_or_create tag: {str(e)}")
            raise

    def get_default_tag(self, db: Session) -> Tag:
        """Get or create the default tag."""
        return self.get_or_create(db, name="untagged")

    def get_document_tags(
        self,
        db: Session,
        *,
        document_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Tag]:
        """Get all tags for a document."""
        try:
            return (
                db.query(Tag)
                .join(Tag.documents)
                .filter(Document.id == document_id)
                .offset(skip)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Error fetching document tags: {str(e)}")
            raise

    def update_document_tags(
        self,
        db: Session,
        *,
        document_id: str,
        tag_ids: List[str]
    ) -> List[Tag]:
        """Update tags for a document."""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Get all tags
            tags = db.query(Tag).filter(Tag.id.in_(tag_ids)).all()
            if len(tags) != len(tag_ids):
                raise ValueError("Some tag IDs are invalid")

            # Update document tags
            document.tags = tags
            db.add(document)
            db.commit()
            db.refresh(document)
            return document.tags
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating document tags: {str(e)}")
            raise


tag = CRUDTag(Tag) 