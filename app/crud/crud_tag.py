import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.crud.base import CRUDBase
from app.db.models.document import Tag
from app.schemas.document import TagCreate

logger = logging.getLogger(__name__)

class CRUDTag(CRUDBase[Tag, TagCreate, TagCreate]):
    def get_by_name(self, db: Session, *, name: str) -> Optional[Tag]:
        """
        Get a tag by its name.
        
        Args:
            db: Database session
            name: Tag name to search for
            
        Returns:
            Tag if found, None otherwise
        """
        return db.query(Tag).filter(Tag.name == name).first()

    def get_multi(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        name_filter: Optional[str] = None
    ) -> List[Tag]:
        """
        Get multiple tags with optional filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            name_filter: Optional partial name to filter tags
            
        Returns:
            List of tags
        """
        query = db.query(Tag)
        
        if name_filter:
            query = query.filter(Tag.name.ilike(f"%{name_filter}%"))
            
        return (
            query.order_by(Tag.name)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_tag_stats(self, db: Session, *, tag_id: int) -> dict:
        """
        Get statistics for a specific tag.
        
        Args:
            db: Database session
            tag_id: Tag ID to get stats for
            
        Returns:
            Dictionary containing tag statistics
        """
        tag = self.get(db, id=tag_id)
        if not tag:
            return None
            
        return {
            "id": tag.id,
            "name": tag.name,
            "document_count": len(tag.documents),
            "created_at": tag.created_at
        }

    def create(self, db: Session, *, obj_in: TagCreate) -> Tag:
        """
        Create a new tag.
        
        Args:
            db: Database session
            obj_in: Tag creation data
            
        Returns:
            Created tag
            
        Raises:
            ValueError: If tag with same name already exists
        """
        try:
            # Check if tag with same name exists
            existing = self.get_by_name(db, name=obj_in.name)
            if existing:
                raise ValueError(f"Tag with name '{obj_in.name}' already exists")
                
            return super().create(db, obj_in=obj_in)
        except Exception as e:
            logger.error(f"Error creating tag: {str(e)}")
            raise

    def remove(self, db: Session, *, id: int) -> Tag:
        """
        Remove a tag.
        
        Args:
            db: Database session
            id: Tag ID to remove
            
        Returns:
            Removed tag
            
        Note:
            This will automatically remove all document-tag associations
            due to the CASCADE delete in the association table
        """
        try:
            return super().remove(db, id=id)
        except Exception as e:
            logger.error(f"Error removing tag: {str(e)}")
            raise

    def search_tags(
        self,
        db: Session,
        *,
        query: str,
        limit: int = 10
    ) -> List[Tag]:
        """
        Search tags by name.
        
        Args:
            db: Database session
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching tags
        """
        return (
            db.query(Tag)
            .filter(Tag.name.ilike(f"%{query}%"))
            .order_by(Tag.name)
            .limit(limit)
            .all()
        )

    def get_popular_tags(
        self,
        db: Session,
        *,
        limit: int = 10
    ) -> List[dict]:
        """
        Get most used tags with document counts.
        
        Args:
            db: Database session
            limit: Maximum number of tags to return
            
        Returns:
            List of tags with their usage counts
        """
        try:
            # Count documents per tag using the association table
            results = (
                db.query(
                    Tag,
                    func.count(Tag.documents).label('doc_count')
                )
                .group_by(Tag.id)
                .order_by(func.count(Tag.documents).desc())
                .limit(limit)
                .all()
            )
            
            return [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "document_count": doc_count,
                    "created_at": tag.created_at
                }
                for tag, doc_count in results
            ]
        except Exception as e:
            logger.error(f"Error getting popular tags: {str(e)}")
            raise


tag = CRUDTag(Tag) 