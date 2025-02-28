from enum import Enum


class AnalysisProcessingType(str, Enum):
    """
    Enum for the processing type of an analysis.
    """
    SINGLE = "single"
    BATCH = "batch"

class AnalysisMode(str, Enum):
    """
    Enum for the mode of an analysis.
    """
    AUTOMATIC = "automatic"
    STEP_BY_STEP = "step_by_step"


class AnalysisStatus(str, Enum):
    """
    Enum for the status of an analysis.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"