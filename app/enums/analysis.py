import enum

class AnalysisMode(str, enum.Enum):
    """Mode of analysis execution."""
    AUTOMATIC = "automatic"  
    STEP_BY_STEP = "step_by_step"

class AnalysisStatus(str, enum.Enum):
    """Status of an analysis."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVED = "approved"
    REJECTED = "rejected"

class AnalysisType(str, enum.Enum):
    """Types of analysis supported by the system."""
    TABLE_ANALYSIS = "table_analysis"
    TEXT_ANALYSIS = "text_analysis"
    TEMPLATE_CONVERSION = "template_conversion"