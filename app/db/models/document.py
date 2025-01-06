from sqlalchemy import String, Integer, DateTime, Enum, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import List
from app.db.session import Base
from app.schemas.document import DocumentType
from app.db.models.user import User
from app.db.models.analysis_result import AnalysisResult
from app.schemas.analysis import AnalysisStatus

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[DocumentType] = mapped_column(Enum(DocumentType), nullable=False)
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False
    )
    uploaded_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False)  # Size in bytes
    url: Mapped[str] = mapped_column(String, nullable=False)
    
    # User relationship
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    user: Mapped["User"] = relationship(
        "User",
        back_populates="documents",
        lazy="selectin"
    )
    
    # Analysis results relationship
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    # Indexes for common queries
    __table_args__ = (
        # Index for user's documents list with sorting by upload date
        Index('ix_documents_user_id_uploaded_at', user_id, uploaded_at.desc()),
        # Index for filtering by status and type
        Index('ix_documents_status_type', status, type),
        # Index for name search
        Index('ix_documents_name', name),
    )