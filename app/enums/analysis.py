from enum import Enum

class AnalysisTypeEnum(str, Enum):
    """
    Enum for the type of analysis.
    """
    TABLE_ANALYSIS = "table_analysis"
    TEXT_ANALYSIS = "text_analysis"
    TEMPLATE_CONVERSION = "template_conversion"


class AnalysisStepEnum(str, Enum):
    """
    Enum for the steps of an analysis.
    """
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
