from sqlalchemy import String, ForeignKey, Index, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime
from sqlalchemy import func

from app.db.base_class import Base
from app.schemas.analysis import AnalysisType
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