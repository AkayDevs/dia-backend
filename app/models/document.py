from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.db.base_class import Base

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

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String, Enum(DocumentType))
    size = Column(Integer)  # in bytes
    status = Column(String, Enum(DocumentStatus), default=DocumentStatus.PENDING)
    url = Column(String)
    metadata = Column(JSON, nullable=True)
    
    # File processing metadata
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    has_tables = Column(Boolean, nullable=True)
    has_images = Column(Boolean, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # User relationship
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="documents")
    
    # Analysis results
    analyses = relationship("Analysis", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "status": self.status,
            "url": self.url,
            "metadata": {
                "pageCount": self.page_count,
                "wordCount": self.word_count,
                "hasTables": self.has_tables,
                "hasImages": self.has_images
            },
            "uploadedAt": self.uploaded_at.isoformat(),
            "processedAt": self.processed_at.isoformat() if self.processed_at else None
        } 