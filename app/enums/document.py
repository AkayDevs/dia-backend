from enum import Enum

class DocumentType(str, Enum):
    """Document types supported by the system."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    IMAGE = "image"
    UNKNOWN = "unknown"



# Mappings ------------------------------------------------------------------


MIME_TYPES = {
    'application/pdf': DocumentType.PDF,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
    'application/msword': DocumentType.DOCX,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentType.XLSX,
    'image/jpeg': DocumentType.IMAGE,
    'image/png': DocumentType.IMAGE,
    'image/jpg': DocumentType.IMAGE,
    'image/bmp': DocumentType.IMAGE,
    'image/gif': DocumentType.IMAGE,
    'image/tiff': DocumentType.IMAGE,
    'image/webp': DocumentType.IMAGE,
}