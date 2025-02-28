from sqlalchemy import Column, String, DateTime, ForeignKey, func, JSON
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from uuid import uuid4


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    analysis_code = Column(String(100), nullable=False)
    mode = Column(String(20), nullable=False)  # "automatic" or "step_by_step"
    status = Column(String(20), nullable=False)  # "pending", "in_progress", "completed", "failed"
    config = Column(JSON, default=dict)  # Configuration for the analysis run
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(String(500))
    
    # Relationships
    document = relationship("Document", back_populates="analysis_runs")
    analysis_definition = relationship("AnalysisDefinition", back_populates="analysis_runs")
    step_results = relationship("StepExecutionResult", back_populates="analysis_run", cascade="all, delete-orphan")


class StepExecutionResult(Base):
    __tablename__ = "step_execution_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    analysis_run_id = Column(String(36), ForeignKey("analysis_runs.id"), nullable=False)
    step_code = Column(String(100), nullable=False)
    algorithm_code = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # "pending", "in_progress", "completed", "failed"
    parameters = Column(JSON)  # Parameters used for this execution
    result = Column(JSON)  # The actual result data
    user_corrections = Column(JSON)  # Any corrections made by the user
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(String(500))
    
    # Relationships
    analysis_run = relationship("AnalysisRun", back_populates="step_results")
    step_definition = relationship("StepDefinition", back_populates="step_results")
    algorithm_definition = relationship("AlgorithmDefinition", back_populates="step_results") 