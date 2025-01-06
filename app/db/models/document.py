from sqlalchemy import String, Integer, DateTime, Enum, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import List
import enum
from app.db.session import Base


class DocumentType(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"


class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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


class AnalysisType(str, enum.Enum):
    TABLE_DETECTION = "table_detection"
    TEXT_EXTRACTION = "text_extraction"
    TEXT_SUMMARIZATION = "text_summarization"
    TEMPLATE_CONVERSION = "template_conversion"


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    type: Mapped[AnalysisType] = mapped_column(Enum(AnalysisType), nullable=False)
    result: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON string of results
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="analysis_results",
        lazy="selectin"
    )

    # Indexes for common queries
    __table_args__ = (
        # Index for getting all analysis results for a document
        Index('ix_analysis_results_document_id_type', document_id, type),
        # Index for filtering by analysis type
        Index('ix_analysis_results_type', type),
        # Index for sorting by creation date
        Index('ix_analysis_results_created_at', created_at.desc()),
    ) 