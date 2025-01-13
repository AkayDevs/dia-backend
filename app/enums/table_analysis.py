import enum

class TableAnalysisStep(str, enum.Enum):
    """Steps in table analysis process."""
    DETECTION = "detection"
    STRUCTURE_RECOGNITION = "structure_recognition"
    DATA_EXTRACTION = "data_extraction"

class TableDetectionAlgorithm(str, enum.Enum):
    """Enumeration of supported table detection algorithms."""
    MSA = "msa"
    PYTHON_CUSTOM = "python_custom"
    YOLO = "yolo"

class TableStructureRecognitionAlgorithm(str, enum.Enum):
    """Enumeration of supported table structure recognition algorithms."""
    MSA = "msa"
    PYTHON_CUSTOM = "python_custom"
    YOLO = "yolo"

class TableDataExtractionAlgorithm(str, enum.Enum):
    """Enumeration of supported table data extraction algorithms."""
    MSA = "msa"
    PYTHON_CUSTOM = "python_custom"
    YOLO = "yolo"
