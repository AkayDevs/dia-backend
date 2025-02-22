"""Table analysis module initialization and registration."""

from app.services.analysis.configs.registry import AnalysisRegistry
from .analysis import TableAnalysis
from .steps.table_detection import TableDetectionStep
from .steps.table_structure import TableStructureStep
from .steps.table_data import TableDataStep
from .algorithms.basic_detection import BasicTableDetectionAlgorithm

def register_components():
    """Register all table analysis components."""
    # Register analysis definition
    analysis_info = TableAnalysis().get_info()
    AnalysisRegistry.register_analysis_definition(analysis_info)

    # Register steps
    table_detection_step = TableDetectionStep()
    table_structure_step = TableStructureStep()
    table_data_step = TableDataStep()

    AnalysisRegistry.register_step(table_detection_step.get_info(), analysis_info.code)
    AnalysisRegistry.register_step(table_structure_step.get_info(), analysis_info.code)
    AnalysisRegistry.register_step(table_data_step.get_info(), analysis_info.code)

    # Register algorithms
    basic_detection_algo = BasicTableDetectionAlgorithm()
    AnalysisRegistry.register_algorithm(
        basic_detection_algo.get_info(),
        f"{analysis_info.code}.{table_detection_step.get_info().code}"
    )
