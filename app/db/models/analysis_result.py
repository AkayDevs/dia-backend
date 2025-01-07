from sqlalchemy import String, ForeignKey, Index, Enum, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime
from sqlalchemy import func

from app.db.base_class import Base
from app.schemas.analysis import AnalysisType, AnalysisStatus
from typing import TYPE_CHECKING, Dict, Any
import uuid

if TYPE_CHECKING:
    from app.db.models.document import Document


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(
        String, 
        primary_key=True, 
        index=True, 
        default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    type: Mapped[AnalysisType] = mapped_column(
        Enum(AnalysisType), 
        nullable=False,
        index=True
    )
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True
    )
    result: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=None
    )
    error: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )
    progress: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )
    status_message: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )
    parameters: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=None
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    completed_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
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
    ) 