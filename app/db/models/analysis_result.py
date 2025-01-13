from sqlalchemy import String, ForeignKey, Index, Enum, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime
from sqlalchemy import func

from app.db.base_class import Base
from app.enums.analysis import AnalysisType, AnalysisStatus, AnalysisMode
from app.enums.table_analysis import TableAnalysisStep
from typing import TYPE_CHECKING, Dict, Any, Optional
import uuid

if TYPE_CHECKING:
    from app.db.models.document import Document


class AnalysisResult(Base):
    """
    Model for storing analysis results.
    
    In automatic mode, step_results contains all results at once.
    In step-by-step mode, step_results contains results for each step separately.
    """
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
    
    # Fields for granular analysis
    mode: Mapped[AnalysisMode] = mapped_column(
        Enum(AnalysisMode),
        default=AnalysisMode.AUTOMATIC,
        nullable=False,
        index=True
    )
    current_step: Mapped[str | None] = mapped_column(
        String,  # Using String to support different step types for different analyses
        nullable=True,
        index=True
    )
    step_results: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Stores analysis results and status for each step. Format: {step_name: {status, result, created_at, completed_at, error, accuracy}}"
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
        # Index for filtering by mode and status
        Index('ix_analysis_results_mode_status', mode, status),
        # Index for step-based queries
        Index('ix_analysis_results_current_step', current_step),
    ) 