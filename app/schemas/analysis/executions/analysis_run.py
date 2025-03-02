from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from app.enums.analysis import AnalysisStatus, AnalysisMode
from .step_result import StepExecutionResultInfo
from app.schemas.analysis.configs.algorithms import AlgorithmSelection


# Configs for Analysis Run

class StepConfig(BaseModel):
    """Configuration for a specific step in the analysis."""
    algorithm: Optional[AlgorithmSelection] = Field(
        None,
        description="Algorithm configuration. If not provided, default algorithm will be used."
    )
    enabled: bool = Field(
        default=True,
        description="Whether this step should be executed"
    )
    timeout: Optional[int] = Field(
        None,
        description="Maximum execution time in seconds for this step"
    )
    retry: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="Number of retry attempts on failure"
    )

class NotificationConfig(BaseModel):
    """Configuration for analysis run notifications via websockets."""
    notify_on_completion: bool = Field(
        default=True,
        description="Whether to send notification on completion"
    )
    notify_on_failure: bool = Field(
        default=True,
        description="Whether to send notification on failure"
    )
    websocket_channel: Optional[str] = Field(
        default=None,
        description="WebSocket channel ID for real-time notifications"
    )

class AnalysisRunConfig(BaseModel):
    """Complete configuration for an analysis run."""
    steps: Dict[str, StepConfig] = Field(
        default_factory=dict,
        description="Configuration for each step, keyed by step ID"
    )
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="WebSocket notification configuration for real-time updates"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the analysis run, including batch information"
    )

    class Config:
        schema_extra = {
            "example_single": {
                "steps": {
                    "step_1": {
                        "algorithm": {
                            "algorithm_code": "table_detection_v2",
                            "algorithm_version": "1.0.0",
                            "parameters": {
                                "confidence_threshold": 0.8,
                                "max_tables_per_page": 10
                            }
                        },
                        "enabled": True,
                        "timeout": 300,
                        "retry": 2
                    }
                },
                "notifications": {
                    "notify_on_completion": True,
                    "notify_on_failure": True,
                    "websocket_channel": "user_123_analysis_456"
                },
                "metadata": {
                    "analysis_type": "single",
                    "priority": "high",
                    "source": "api"
                }
            },
            "example_batch": {
                "steps": {
                    "step_1": {
                        "algorithm": {
                            "algorithm_code": "table_detection_v2",
                            "algorithm_version": "1.0.0",
                            "parameters": {
                                "confidence_threshold": 0.8,
                                "max_tables_per_page": 10
                            }
                        },
                        "enabled": True,
                        "timeout": 300,
                        "retry": 2
                    }
                },
                "notifications": {
                    "notify_on_completion": True,
                    "notify_on_failure": True,
                    "websocket_channel": "user_123_analysis_456"
                },
                "metadata": {
                    "analysis_type": "batch",
                    "batch_id": "batch_123",
                    "batch_email": "user@example.com",
                    "total_documents": 50,
                    "document_index": 5,
                    "priority": "normal",
                    "source": "batch_upload"
                }
            }
        }


class AnalysisRunBase(BaseModel):
    """Base schema for analysis run data"""
    document_id: str = Field(..., description="ID of the document being analyzed")
    analysis_code: str = Field(..., description="Code of the analysis definition")
    mode: AnalysisMode = Field(..., description="Analysis execution mode")
    config: AnalysisRunConfig = Field(
        default_factory=AnalysisRunConfig,
        description="Configuration for the analysis run"
    )

class AnalysisRunCreate(AnalysisRunBase):
    """Schema for creating a new analysis run"""
    pass

class AnalysisRunUpdate(BaseModel):
    """Schema for updating an analysis run"""
    status: Optional[AnalysisStatus] = None
    config: Optional[AnalysisRunConfig] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class AnalysisRunInDB(AnalysisRunBase):
    """Schema for analysis run as stored in database"""
    id: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class AnalysisRunInfo(AnalysisRunInDB):
    """Schema for analysis run with basic information"""
    pass

class AnalysisRunWithResults(AnalysisRunInfo):
    """Schema for analysis run with step results"""
    step_results: List[StepExecutionResultInfo] = Field(
        default_factory=list,
        description="Results of individual analysis steps"
    )

    class Config:
        orm_mode = True
        from_attributes = True
        arbitrary_types_allowed = True 