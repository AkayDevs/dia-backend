from sqlalchemy import Column, String, DateTime, Boolean, JSON, UniqueConstraint, func, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from uuid import uuid4

class AnalysisDefinition(Base):
    __tablename__ = "analysis_definitions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    code = Column(String(100), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    description = Column(String(500))
    supported_document_types = Column(JSON)
    implementation_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default='1')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    steps = relationship("StepDefinition", back_populates="analysis_definition", cascade="all, delete-orphan")
    analysis_runs = relationship("AnalysisRun", back_populates="analysis_definition")
    
    __table_args__ = (
        # Ensure unique combination of code and version
        UniqueConstraint('code', 'version', name='uix_analysis_type_code_version'),
    )


class StepDefinition(Base):
    __tablename__ = "step_definitions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    code = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    description = Column(String(500))
    order = Column(Integer, nullable=False)
    analysis_definition_id = Column(String(36), ForeignKey("analysis_definitions.id"), nullable=False)
    result_schema = Column(String(255), nullable=False)
    base_parameters = Column(JSON)
    implementation_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default='1')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    analysis_definition = relationship("AnalysisDefinition", back_populates="steps")
    algorithms = relationship("AlgorithmDefinition", back_populates="step", cascade="all, delete-orphan")
    step_results = relationship("StepExecutionResult", back_populates="step_definition")
    
    __table_args__ = (
        # Ensure unique combination of code and version within an analysis type
        UniqueConstraint('analysis_definition_id', 'code', 'version', name='uix_step_analysis_code_version'),
    )


class AlgorithmDefinition(Base):
    __tablename__ = "algorithm_definitions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    code = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    version = Column(String(20), nullable=False)
    step_id = Column(String(36), ForeignKey("step_definitions.id"), nullable=False)
    supported_document_types = Column(JSON)
    parameters = Column(JSON)
    implementation_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default='1')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    step = relationship("StepDefinition", back_populates="algorithms")
    step_results = relationship("StepExecutionResult", back_populates="algorithm_definition")
    
    __table_args__ = (
        # Ensure unique combination of code and version within a step
        UniqueConstraint('step_id', 'code', 'version', name='uix_algorithm_step_code_version'),
    )