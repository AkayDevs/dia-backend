from sqlalchemy import String, ForeignKey, Index, Enum, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime
from sqlalchemy import func

from app.db.base_class import Base
from app.schemas.analysis import AnalysisType, AnalysisStatus
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.models.document import Document


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    type: Mapped[AnalysisType] = mapped_column(Enum(AnalysisType), nullable=False)
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False
    )
    result: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON string of results
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status_message: Mapped[str | None] = mapped_column(String, nullable=True)
    parameters: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON string of parameters
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
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
        # Index for filtering by analysis type
        Index('ix_analysis_results_type', type),
        # Index for filtering by status
        Index('ix_analysis_results_status', status),
        # Index for sorting by creation date
        Index('ix_analysis_results_created_at', created_at.desc()),
    ) 