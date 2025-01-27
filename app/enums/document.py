from enum import Enum

class DocumentType(str, Enum):
    """Document types supported by the system."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"