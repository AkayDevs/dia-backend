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
import logging

logger = logging.getLogger(__name__)

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
    
    column_labels = {
        AnalysisDefinition.id: "ID",
        AnalysisDefinition.code: "Code",
        AnalysisDefinition.name: "Name",
        AnalysisDefinition.version: "Version",
        AnalysisDefinition.description: "Description",
        AnalysisDefinition.supported_document_types: "Supported Types",
        AnalysisDefinition.is_active: "Active",
        AnalysisDefinition.created_at: "Created",
        AnalysisDefinition.updated_at: "Updated"
    }
    
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
        AnalysisDefinition.supported_document_types: lambda m, a: ", ".join(m.supported_document_types),
        AnalysisDefinition.created_at: lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M"),
        AnalysisDefinition.updated_at: lambda m, a: m.updated_at.strftime("%Y-%m-%d %H:%M")
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
    
    form_widget_args = {
        AnalysisDefinition.code: {
            "placeholder": "unique_analysis_code"
        },
        AnalysisDefinition.name: {
            "placeholder": "Analysis Name"
        },
        AnalysisDefinition.version: {
            "placeholder": "1.0.0"
        },
        AnalysisDefinition.description: {
            "placeholder": "Description of what this analysis does"
        },
        AnalysisDefinition.implementation_path: {
            "placeholder": "path.to.implementation"
        }
    }
    
    # Column descriptions
    column_descriptions = {
        AnalysisDefinition.code: "Unique identifier for this analysis type",
        AnalysisDefinition.name: "Human-readable name",
        AnalysisDefinition.version: "Semantic version number",
        AnalysisDefinition.description: "Detailed description of the analysis",
        AnalysisDefinition.supported_document_types: "Document types this analysis can process",
        AnalysisDefinition.implementation_path: "Python path to the implementation class",
        AnalysisDefinition.is_active: "Whether this analysis definition is active"
    }
    
    # Security settings
    can_create = True
    can_delete = False
    can_edit = True
    can_export = True
    can_view_details = True
    
    # List configuration
    page_size = 25
    can_set_page_size = True

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
    
    column_labels = {
        StepDefinition.id: "ID",
        StepDefinition.code: "Code",
        StepDefinition.name: "Name",
        StepDefinition.version: "Version",
        StepDefinition.order: "Order",
        StepDefinition.description: "Description",
        StepDefinition.is_active: "Active",
        StepDefinition.analysis_definition_id: "Analysis",
        StepDefinition.created_at: "Created",
        StepDefinition.updated_at: "Updated"
    }
    
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
        StepDefinition.is_active: lambda m, a: "✓" if m.is_active else "✗",
        StepDefinition.created_at: lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M"),
        StepDefinition.updated_at: lambda m, a: m.updated_at.strftime("%Y-%m-%d %H:%M")
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
    
    form_widget_args = {
        StepDefinition.code: {
            "placeholder": "step_code"
        },
        StepDefinition.name: {
            "placeholder": "Step Name"
        },
        StepDefinition.version: {
            "placeholder": "1.0.0"
        },
        StepDefinition.description: {
            "placeholder": "Description of this step"
        },
        StepDefinition.implementation_path: {
            "placeholder": "path.to.implementation"
        }
    }
    
    # Column descriptions
    column_descriptions = {
        StepDefinition.code: "Unique identifier for this step",
        StepDefinition.name: "Human-readable name",
        StepDefinition.version: "Semantic version number",
        StepDefinition.description: "Detailed description of the step",
        StepDefinition.order: "Execution order in the analysis",
        StepDefinition.analysis_definition_id: "Parent analysis definition",
        StepDefinition.base_parameters: "Default parameters for this step",
        StepDefinition.result_schema_path: "Path to the result schema definition",
        StepDefinition.implementation_path: "Python path to the implementation class",
        StepDefinition.is_active: "Whether this step is active"
    }
    
    # Security settings
    can_create = True
    can_delete = False
    can_edit = True
    can_export = True
    can_view_details = True
    
    # List configuration
    page_size = 25
    can_set_page_size = True

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
    
    column_labels = {
        AlgorithmDefinition.id: "ID",
        AlgorithmDefinition.code: "Code",
        AlgorithmDefinition.name: "Name",
        AlgorithmDefinition.version: "Version",
        AlgorithmDefinition.description: "Description",
        AlgorithmDefinition.step_id: "Step",
        AlgorithmDefinition.is_active: "Active",
        AlgorithmDefinition.created_at: "Created",
        AlgorithmDefinition.updated_at: "Updated"
    }
    
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
        AlgorithmDefinition.supported_document_types: lambda m, a: ", ".join(m.supported_document_types),
        AlgorithmDefinition.created_at: lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M"),
        AlgorithmDefinition.updated_at: lambda m, a: m.updated_at.strftime("%Y-%m-%d %H:%M")
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
    
    form_widget_args = {
        AlgorithmDefinition.code: {
            "placeholder": "algorithm_code"
        },
        AlgorithmDefinition.name: {
            "placeholder": "Algorithm Name"
        },
        AlgorithmDefinition.version: {
            "placeholder": "1.0.0"
        },
        AlgorithmDefinition.description: {
            "placeholder": "Description of this algorithm"
        },
        AlgorithmDefinition.implementation_path: {
            "placeholder": "path.to.implementation"
        }
    }
    
    # Column descriptions
    column_descriptions = {
        AlgorithmDefinition.code: "Unique identifier for this algorithm",
        AlgorithmDefinition.name: "Human-readable name",
        AlgorithmDefinition.version: "Semantic version number",
        AlgorithmDefinition.description: "Detailed description of the algorithm",
        AlgorithmDefinition.step_id: "Parent analysis step",
        AlgorithmDefinition.supported_document_types: "Document types this algorithm can process",
        AlgorithmDefinition.parameters: "Default parameters for this algorithm",
        AlgorithmDefinition.implementation_path: "Python path to the implementation class",
        AlgorithmDefinition.is_active: "Whether this algorithm is active"
    }
    
    # Security settings
    can_create = True
    can_delete = False
    can_edit = True
    can_export = True
    can_view_details = True
    
    # List configuration
    page_size = 25
    can_set_page_size = True

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
    
    column_labels = {
        AnalysisRun.id: "ID",
        AnalysisRun.document_id: "Document",
        AnalysisRun.analysis_definition_id: "Analysis",
        AnalysisRun.mode: "Mode",
        AnalysisRun.status: "Status",
        AnalysisRun.created_at: "Created",
        AnalysisRun.started_at: "Started",
        AnalysisRun.completed_at: "Completed"
    }
    
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
        AnalysisRun.status: lambda m, a: f"<span class='badge badge-{_get_status_badge(m.status)}'>{m.status}</span>",
        AnalysisRun.created_at: lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M"),
        AnalysisRun.started_at: lambda m, a: m.started_at.strftime("%Y-%m-%d %H:%M") if m.started_at else "-",
        AnalysisRun.completed_at: lambda m, a: m.completed_at.strftime("%Y-%m-%d %H:%M") if m.completed_at else "-"
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
    
    # Column descriptions
    column_descriptions = {
        AnalysisRun.document_id: "Document being analyzed",
        AnalysisRun.analysis_definition_id: "Analysis being performed",
        AnalysisRun.mode: "Analysis execution mode",
        AnalysisRun.status: "Current status of the analysis",
        AnalysisRun.error_message: "Error message if analysis failed",
        AnalysisRun.config: "Analysis configuration"
    }
    
    # Security settings
    can_create = False
    can_delete = False
    can_edit = False
    can_export = True
    can_view_details = True
    
    # List configuration
    page_size = 25
    can_set_page_size = True

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
    
    column_labels = {
        StepExecutionResult.id: "ID",
        StepExecutionResult.analysis_run_id: "Analysis Run",
        StepExecutionResult.step_definition_id: "Step",
        StepExecutionResult.algorithm_definition_id: "Algorithm",
        StepExecutionResult.status: "Status",
        StepExecutionResult.created_at: "Created",
        StepExecutionResult.started_at: "Started",
        StepExecutionResult.completed_at: "Completed"
    }
    
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
        StepExecutionResult.status: lambda m, a: f"<span class='badge badge-{_get_status_badge(m.status)}'>{m.status}</span>",
        StepExecutionResult.created_at: lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M"),
        StepExecutionResult.started_at: lambda m, a: m.started_at.strftime("%Y-%m-%d %H:%M") if m.started_at else "-",
        StepExecutionResult.completed_at: lambda m, a: m.completed_at.strftime("%Y-%m-%d %H:%M") if m.completed_at else "-"
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
    
    # Column descriptions
    column_descriptions = {
        StepExecutionResult.analysis_run_id: "Parent analysis run",
        StepExecutionResult.step_definition_id: "Step being executed",
        StepExecutionResult.algorithm_definition_id: "Algorithm being used",
        StepExecutionResult.status: "Current status of the step",
        StepExecutionResult.parameters: "Execution parameters",
        StepExecutionResult.result: "Step execution results",
        StepExecutionResult.user_corrections: "User-provided corrections",
        StepExecutionResult.error_message: "Error message if step failed"
    }
    
    # Security settings
    can_create = False
    can_delete = False
    can_edit = False
    can_export = True
    can_view_details = True
    
    # List configuration
    page_size = 25
    can_set_page_size = True

def _get_status_badge(status: str) -> str:
    """Get appropriate Bootstrap badge class for status."""
    return {
        AnalysisStatus.PENDING: "secondary",
        AnalysisStatus.IN_PROGRESS: "info",
        AnalysisStatus.COMPLETED: "success",
        AnalysisStatus.FAILED: "danger",
        AnalysisStatus.CANCELLED: "warning"
    }.get(status, "secondary") 