from sqlalchemy import Column, String, Integer, ForeignKey, JSON, Enum, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from typing import List
from uuid import uuid4

from app.db.base_class import Base

class AnalysisTypeEnum(str, enum.Enum):
    TABLE_ANALYSIS = "table_analysis"
    TEXT_ANALYSIS = "text_analysis"
    TEMPLATE_CONVERSION = "template_conversion"

class AnalysisStepEnum(str, enum.Enum):
    # Table Analysis Steps
    TABLE_DETECTION = "table_detection"
    TABLE_STRUCTURE_RECOGNITION = "table_structure_recognition"
    TABLE_DATA_EXTRACTION = "table_data_extraction"
    
    # Text Analysis Steps
    TEXT_DETECTION = "text_detection"
    TEXT_RECOGNITION = "text_recognition"
    TEXT_CLASSIFICATION = "text_classification"
    
    # Template Conversion Steps
    TEMPLATE_DETECTION = "template_detection"
    TEMPLATE_MATCHING = "template_matching"
    TEMPLATE_EXTRACTION = "template_extraction"

class AnalysisType(Base):
    __tablename__ = "analysis_types"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(Enum(AnalysisTypeEnum), nullable=False)
    description = Column(String(500))
    supported_document_types = Column(JSON)  # List of supported DocumentType values
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    steps = relationship("AnalysisStep", back_populates="analysis_type", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="analysis_type")

class AnalysisStep(Base):
    __tablename__ = "analysis_steps"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(Enum(AnalysisStepEnum), nullable=False)
    description = Column(String(500))
    order = Column(Integer, nullable=False)  # Order in the analysis pipeline
    analysis_type_id = Column(String(36), ForeignKey("analysis_types.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Base parameters that all algorithms for this step must support
    base_parameters = Column(JSON)  # List of Parameter objects
    
    # Relationships
    analysis_type = relationship("AnalysisType", back_populates="steps")
    algorithms = relationship("Algorithm", back_populates="step", cascade="all, delete-orphan")
    step_results = relationship("AnalysisStepResult", back_populates="step")

class Algorithm(Base):
    __tablename__ = "algorithms"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    version = Column(String(20), nullable=False)
    step_id = Column(String(36), ForeignKey("analysis_steps.id"), nullable=False)
    supported_document_types = Column(JSON)  # List of supported DocumentType values
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Algorithm-specific parameters in addition to base parameters
    parameters = Column(JSON)  # List of Parameter objects
    
    # Relationships
    step = relationship("AnalysisStep", back_populates="algorithms")
    step_results = relationship("AnalysisStepResult", back_populates="algorithm")

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