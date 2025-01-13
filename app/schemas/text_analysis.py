from app.schemas.analysis import AnalysisParameters


# Text Segmentation Parameters --------------------------------------------------

class BaseTextSegmentationParameters(AnalysisParameters):
    """Base parameters for text segmentation."""
    pass

class TextSegmentationParameters(BaseTextSegmentationParameters):
    """Parameters specific to text segmentation."""
    pass


# Text Extraction Parameters --------------------------------------------------

class BaseTextExtractionParameters(AnalysisParameters):
    """Base parameters for text extraction."""
    pass

class TextExtractionParameters(BaseTextExtractionParameters):
    """Parameters specific to text extraction."""
    pass


# Text Summarization Parameters --------------------------------------------------

class BaseTextSummarizationParameters(AnalysisParameters):
    """Base parameters for text summarization."""
    pass

class TextSummarizationParameters(BaseTextSummarizationParameters):
    """Parameters specific to text summarization."""
    pass


# Text Analysis Parameters --------------------------------------------------

class TextAnalysisParameters(AnalysisParameters):
    """Parameters specific to text analysis."""
    pass
