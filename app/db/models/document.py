from sqlalchemy import String, Integer, DateTime, Enum, ForeignKey, Index, Table, Column, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import List, TYPE_CHECKING, Optional
import uuid

from app.db.base_class import Base
from app.enums.document import DocumentType
from app.db.models.analysis_execution import AnalysisRun

if TYPE_CHECKING:
    from app.db.models.user import User

# Association table for document-tag many-to-many relationship
document_tags = Table(
    'document_tags',
    Base.metadata,
    Column('document_id', String, ForeignKey('documents.id', ondelete="CASCADE"), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete="CASCADE"), primary_key=True),
    Index('ix_document_tags_document_id', 'document_id'),
    Index('ix_document_tags_tag_id', 'tag_id')
)

class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationship with documents
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        secondary=document_tags,
        back_populates="tags",
        lazy="selectin"
    )

class Document(Base):
    """Document model."""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[DocumentType] = mapped_column(Enum(DocumentType), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)

    previous_version_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    uploaded_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    archived_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_until: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="documents")
    analysis_runs: Mapped[List["AnalysisRun"]] = relationship(
        AnalysisRun,
        back_populates="document",
        cascade="all, delete-orphan"
    )
    tags: Mapped[List["Tag"]] = relationship("Tag", secondary=document_tags, back_populates="documents")

    __table_args__ = (
        Index("ix_documents_id", "id"),
        Index("ix_documents_name", "name"),
        Index("ix_documents_user_id_uploaded_at", "user_id", "uploaded_at"),
        Index("ix_documents_type", "type"),
    )