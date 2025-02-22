from typing import Any
from app.admin.base import BaseModelView
from app.db.models.analysis_config import (
    AnalysisDefinition,
    StepDefinition,
    AlgorithmDefinition
)
from app.db.models.analysis_execution import (
    AnalysisRun,
    StepExecutionResult
)
from app.enums.analysis import AnalysisStatus
from app.core.config import settings

class AnalysisDefinitionAdmin(BaseModelView, model=AnalysisDefinition):
    """Admin interface for AnalysisDefinition model."""
    
    name = "Analysis Definition"
    name_plural = "Analysis Definitions"
    icon = "fa-solid fa-cube"
    
    # List view configuration
    column_list = [
        AnalysisDefinition.id,
        AnalysisDefinition.code,
        AnalysisDefinition.name,
        AnalysisDefinition.version,
        AnalysisDefinition.description,
        AnalysisDefinition.supported_document_types,
        AnalysisDefinition.is_active,
        AnalysisDefinition.created_at,
        AnalysisDefinition.updated_at
    ]
    
    column_searchable_list = [
        AnalysisDefinition.code,
        AnalysisDefinition.name,
        AnalysisDefinition.description
    ]
    
    column_sortable_list = [
        AnalysisDefinition.code,
        AnalysisDefinition.name,
        AnalysisDefinition.version,
        AnalysisDefinition.is_active,
        AnalysisDefinition.created_at,
        AnalysisDefinition.updated_at
    ]
    
    column_formatters = {
        AnalysisDefinition.is_active: lambda m, a: "✓" if m.is_active else "✗",
        AnalysisDefinition.supported_document_types: lambda m, a: ", ".join(m.supported_document_types)
    }
    
    # Form configuration
    form_columns = [
        AnalysisDefinition.code,
        AnalysisDefinition.name,
        AnalysisDefinition.version,
        AnalysisDefinition.description,
        AnalysisDefinition.supported_document_types,
        AnalysisDefinition.implementation_path,
        AnalysisDefinition.is_active
    ]
    
    # Security settings
    can_create = True
    can_delete = False  # Prevent deletion for data integrity
    can_edit = True
    can_export = True

class StepDefinitionAdmin(BaseModelView, model=StepDefinition):
    """Admin interface for StepDefinition model."""
    
    name = "Analysis Step"
    name_plural = "Analysis Steps"
    icon = "fa-solid fa-list-ol"
    
    # List view configuration
    column_list = [
        StepDefinition.id,
        StepDefinition.code,
        StepDefinition.name,
        StepDefinition.version,
        StepDefinition.order,
        StepDefinition.description,
        StepDefinition.is_active,
        StepDefinition.analysis_definition_id,
        StepDefinition.created_at,
        StepDefinition.updated_at
    ]
    
    column_searchable_list = [
        StepDefinition.code,
        StepDefinition.name,
        StepDefinition.description,
        StepDefinition.analysis_definition_id
    ]
    
    column_sortable_list = [
        StepDefinition.code,
        StepDefinition.name,
        StepDefinition.version,
        StepDefinition.order,
        StepDefinition.is_active,
        StepDefinition.created_at,
        StepDefinition.updated_at
    ]
    
    column_formatters = {
        StepDefinition.is_active: lambda m, a: "✓" if m.is_active else "✗"
    }
    
    # Form configuration
    form_columns = [
        StepDefinition.code,
        StepDefinition.name,
        StepDefinition.version,
        StepDefinition.description,
        StepDefinition.order,
        StepDefinition.analysis_definition_id,
        StepDefinition.base_parameters,
        StepDefinition.result_schema_path,
        StepDefinition.implementation_path,
        StepDefinition.is_active
    ]
    
    # Security settings
    can_create = True
    can_delete = False
    can_edit = True
    can_export = True

class AlgorithmDefinitionAdmin(BaseModelView, model=AlgorithmDefinition):
    """Admin interface for AlgorithmDefinition model."""
    
    name = "Algorithm"
    name_plural = "Algorithms"
    icon = "fa-solid fa-microchip"
    
    # List view configuration
    column_list = [
        AlgorithmDefinition.id,
        AlgorithmDefinition.code,
        AlgorithmDefinition.name,
        AlgorithmDefinition.version,
        AlgorithmDefinition.description,
        AlgorithmDefinition.step_id,
        AlgorithmDefinition.is_active,
        AlgorithmDefinition.created_at,
        AlgorithmDefinition.updated_at
    ]
    
    column_searchable_list = [
        AlgorithmDefinition.code,
        AlgorithmDefinition.name,
        AlgorithmDefinition.description,
        AlgorithmDefinition.step_id
    ]
    
    column_sortable_list = [
        AlgorithmDefinition.code,
        AlgorithmDefinition.name,
        AlgorithmDefinition.version,
        AlgorithmDefinition.is_active,
        AlgorithmDefinition.created_at,
        AlgorithmDefinition.updated_at
    ]
    
    column_formatters = {
        AlgorithmDefinition.is_active: lambda m, a: "✓" if m.is_active else "✗",
        AlgorithmDefinition.supported_document_types: lambda m, a: ", ".join(m.supported_document_types)
    }
    
    # Form configuration
    form_columns = [
        AlgorithmDefinition.code,
        AlgorithmDefinition.name,
        AlgorithmDefinition.version,
        AlgorithmDefinition.description,
        AlgorithmDefinition.step_id,
        AlgorithmDefinition.supported_document_types,
        AlgorithmDefinition.parameters,
        AlgorithmDefinition.implementation_path,
        AlgorithmDefinition.is_active
    ]
    
    # Security settings
    can_create = True
    can_delete = False
    can_edit = True
    can_export = True

class AnalysisRunAdmin(BaseModelView, model=AnalysisRun):
    """Admin interface for AnalysisRun model."""
    
    name = "Analysis Run"
    name_plural = "Analysis Runs"
    icon = "fa-solid fa-microscope"
    
    # List view configuration
    column_list = [
        AnalysisRun.id,
        AnalysisRun.document_id,
        AnalysisRun.analysis_definition_id,
        AnalysisRun.mode,
        AnalysisRun.status,
        AnalysisRun.created_at,
        AnalysisRun.started_at,
        AnalysisRun.completed_at
    ]
    
    column_searchable_list = [
        AnalysisRun.document_id,
        AnalysisRun.analysis_definition_id,
        AnalysisRun.status
    ]
    
    column_sortable_list = [
        AnalysisRun.created_at,
        AnalysisRun.started_at,
        AnalysisRun.completed_at,
        AnalysisRun.status
    ]
    
    column_formatters = {
        AnalysisRun.mode: lambda m, a: f"<span class='badge badge-primary'>{m.mode}</span>",
        AnalysisRun.status: lambda m, a: f"<span class='badge badge-{_get_status_badge(m.status)}'>{m.status}</span>"
    }
    
    # Form configuration - Read-only view
    form_columns = [
        AnalysisRun.document_id,
        AnalysisRun.analysis_definition_id,
        AnalysisRun.mode,
        AnalysisRun.status,
        AnalysisRun.error_message,
        AnalysisRun.config
    ]
    
    # Security settings
    can_create = False
    can_delete = False
    can_edit = False
    can_export = True
    can_view_details = True

class StepExecutionResultAdmin(BaseModelView, model=StepExecutionResult):
    """Admin interface for StepExecutionResult model."""
    
    name = "Step Result"
    name_plural = "Step Results"
    icon = "fa-solid fa-clipboard-check"
    
    # List view configuration
    column_list = [
        StepExecutionResult.id,
        StepExecutionResult.analysis_run_id,
        StepExecutionResult.step_definition_id,
        StepExecutionResult.algorithm_definition_id,
        StepExecutionResult.status,
        StepExecutionResult.created_at,
        StepExecutionResult.started_at,
        StepExecutionResult.completed_at
    ]
    
    column_searchable_list = [
        StepExecutionResult.analysis_run_id,
        StepExecutionResult.step_definition_id,
        StepExecutionResult.algorithm_definition_id,
        StepExecutionResult.status
    ]
    
    column_sortable_list = [
        StepExecutionResult.created_at,
        StepExecutionResult.started_at,
        StepExecutionResult.completed_at,
        StepExecutionResult.status
    ]
    
    column_formatters = {
        StepExecutionResult.status: lambda m, a: f"<span class='badge badge-{_get_status_badge(m.status)}'>{m.status}</span>"
    }
    
    # Form configuration - Read-only view
    form_columns = [
        StepExecutionResult.analysis_run_id,
        StepExecutionResult.step_definition_id,
        StepExecutionResult.algorithm_definition_id,
        StepExecutionResult.status,
        StepExecutionResult.parameters,
        StepExecutionResult.result,
        StepExecutionResult.user_corrections,
        StepExecutionResult.error_message
    ]
    
    # Security settings
    can_create = False
    can_delete = False
    can_edit = False
    can_export = True
    can_view_details = True

def _get_status_badge(status: str) -> str:
    """Get appropriate Bootstrap badge class for status."""
    return {
        AnalysisStatus.PENDING: "secondary",
        AnalysisStatus.IN_PROGRESS: "info",
        AnalysisStatus.COMPLETED: "success",
        AnalysisStatus.FAILED: "danger",
        AnalysisStatus.CANCELLED: "warning"
    }.get(status, "secondary") 