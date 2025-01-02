from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from .session import Base

class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentType(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    documents = relationship("Document", back_populates="owner")
    notification_settings = relationship("NotificationSettings", back_populates="user", uselist=False)

class NotificationSettings(Base):
    __tablename__ = "notification_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    email_notifications = Column(Boolean, default=True)
    analysis_complete = Column(Boolean, default=True)
    document_shared = Column(Boolean, default=True)
    security_alerts = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="notification_settings")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String, Enum(DocumentType))
    size = Column(Integer)  # in bytes
    status = Column(String, Enum(DocumentStatus), default=DocumentStatus.PENDING)
    url = Column(String)
    file_path = Column(String)
    metadata = Column(JSON, nullable=True)
    
    # File processing metadata
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    has_tables = Column(Boolean, nullable=True)
    has_images = Column(Boolean, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # User relationship
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="documents")
    
    # Analysis results
    analysis_results = relationship("AnalysisResult", back_populates="document", cascade="all, delete-orphan")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    analysis_type = Column(String)  # table_detection, text_extraction, etc.
    result_data = Column(Text)  # JSON string of results
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="analysis_results") 