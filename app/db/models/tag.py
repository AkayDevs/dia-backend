from sqlalchemy import String, Table, Column, ForeignKey, Index, DateTime
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import List, TYPE_CHECKING
from datetime import datetime

from app.db.base_class import Base

if TYPE_CHECKING:
    from app.db.models.document import Document

# Association table for many-to-many relationship
document_tags = Table(
    'document_tags',
    Base.metadata,
    Column('document_id', String, ForeignKey('documents.id', ondelete="CASCADE"), primary_key=True),
    Column('tag_id', String, ForeignKey('tags.id', ondelete="CASCADE"), primary_key=True),
    Index('ix_document_tags_document_id', 'document_id'),
    Index('ix_document_tags_tag_id', 'tag_id')
)

class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Document relationship
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        secondary=document_tags,
        back_populates="tags",
        lazy="selectin"
    )

    # Indexes for common queries
    __table_args__ = (
        # Index for tag name searches
        Index('ix_tags_name', name),
    ) 