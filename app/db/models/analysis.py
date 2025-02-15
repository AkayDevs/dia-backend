from sqlalchemy import Column, String, Integer, ForeignKey, JSON, Enum, Boolean, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import List
from uuid import uuid4
from app.enums.analysis import AnalysisTypeEnum, AnalysisStepEnum

from app.db.base_class import Base


class AnalysisType(Base):
    __tablename__ = "analysis_types"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    code = Column(String(100), nullable=False, unique=True)  # Identifier code from registry
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    description = Column(String(500))
    supported_document_types = Column(JSON)
    implementation_path = Column(String(255), nullable=False)  # Path to implementation class
    is_active = Column(Boolean, nullable=False, server_default='1')  # Match database schema
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    steps = relationship("AnalysisStep", back_populates="analysis_type", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="analysis_type")
    
    __table_args__ = (
        # Ensure unique combination of code and version
        UniqueConstraint('code', 'version', name='uix_analysis_type_code_version'),
    )

class AnalysisStep(Base):
    __tablename__ = "analysis_steps"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    code = Column(String(100), nullable=False)  # Identifier code from registry
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    description = Column(String(500))
    order = Column(Integer, nullable=False)  # Order in the analysis pipeline
    analysis_type_id = Column(String(36), ForeignKey("analysis_types.id"), nullable=False)
    result_schema = Column(String(255), nullable=False)  # Python path to result schema class
    base_parameters = Column(JSON)  # List of Parameter objects
    implementation_path = Column(String(255), nullable=False)  # Path to implementation class
    is_active = Column(Boolean, nullable=False, server_default='1')  # Match database schema
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    analysis_type = relationship("AnalysisType", back_populates="steps")
    algorithms = relationship("Algorithm", back_populates="step", cascade="all, delete-orphan")
    step_results = relationship("AnalysisStepResult", back_populates="step")
    
    __table_args__ = (
        # Ensure unique combination of code and version within an analysis type
        UniqueConstraint('analysis_type_id', 'code', 'version', name='uix_step_analysis_code_version'),
    )

class Algorithm(Base):
    __tablename__ = "algorithms"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    code = Column(String(100), nullable=False)  # Identifier code from registry
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    version = Column(String(20), nullable=False)
    step_id = Column(String(36), ForeignKey("analysis_steps.id"), nullable=False)
    supported_document_types = Column(JSON)  # List of supported DocumentType values
    parameters = Column(JSON)  # List of Parameter objects
    implementation_path = Column(String(255), nullable=False)  # Path to implementation class
    is_active = Column(Boolean, nullable=False, server_default='1')  # Match database schema
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    step = relationship("AnalysisStep", back_populates="algorithms")
    step_results = relationship("AnalysisStepResult", back_populates="algorithm")
    
    __table_args__ = (
        # Ensure unique combination of code and version within a step
        UniqueConstraint('step_id', 'code', 'version', name='uix_algorithm_step_code_version'),
    )

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    analysis_type_id = Column(String(36), ForeignKey("analysis_types.id"), nullable=False)
    mode = Column(String(20), nullable=False)  # "automatic" or "step_by_step"
    status = Column(String(20), nullable=False)  # "pending", "in_progress", "completed", "failed"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(String(500))
    
    # Relationships
    document = relationship("Document", back_populates="analyses")
    analysis_type = relationship("AnalysisType", back_populates="analyses")
    step_results = relationship("AnalysisStepResult", back_populates="analysis", cascade="all, delete-orphan")

class AnalysisStepResult(Base):
    __tablename__ = "analysis_step_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    analysis_id = Column(String(36), ForeignKey("analyses.id"), nullable=False)
    step_id = Column(String(36), ForeignKey("analysis_steps.id"), nullable=False)
    algorithm_id = Column(String(36), ForeignKey("algorithms.id"), nullable=False)
    status = Column(String(20), nullable=False)  # "pending", "in_progress", "completed", "failed"
    parameters = Column(JSON)  # Parameters used for this execution
    result = Column(JSON)  # The actual result data
    user_corrections = Column(JSON)  # Any corrections made by the user
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(String(500))
    
    # Relationships
    analysis = relationship("Analysis", back_populates="step_results")
    step = relationship("AnalysisStep", back_populates="step_results")
    algorithm = relationship("Algorithm", back_populates="step_results") 