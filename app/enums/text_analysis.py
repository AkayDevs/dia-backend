import enum

class TextAnalysisStep(str, enum.Enum):
    """Steps in text analysis process."""
    SEGMENTATION = "segmentation"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"


class TextSegmentationAlgorithm(str, enum.Enum):
    """Enumeration of supported text segmentation algorithms."""
    MSA = "msa"
    PYTHON_CUSTOM = "python_custom"
    YOLO = "yolo"

class TextExtractionAlgorithm(str, enum.Enum):
    """Enumeration of supported text extraction algorithms."""
    BERT = "bert"
    GPT = "gpt"
    PYTHON_CUSTOM = "python_custom"

class TextSummarizationAlgorithm(str, enum.Enum):
    """Enumeration of supported text summarization algorithms."""
    BERT = "bert"
    GPT = "gpt"
    PYTHON_CUSTOM = "python_custom"

