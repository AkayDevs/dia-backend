from typing import Dict, Any, Optional, List
from app.services.analysis.configs.base import BaseStep
from app.schemas.analysis.configs.steps import StepDefinitionBase
from app.schemas.analysis.configs.algorithms import AlgorithmParameter, AlgorithmSelection

class TableStructureStep(BaseStep):
    """Table structure step implementation"""
    
    def get_info(self) -> StepDefinitionBase:
        return StepDefinitionBase(
            code="table_structure",
            name="Table Structure",
            version="1.0.0",
            description="Extract table structure from the detected tables",
            order=2,
            base_parameters=[
                AlgorithmParameter(
                    name="consider_headers",
                    description="Whether to consider headers in the table structure",
                    type="boolean",
                    required=False,
                    default=True
                ),
                AlgorithmParameter(
                    name="consider_merged_cells",
                    description="Whether to consider merged cells in the table structure",
                    type="boolean",
                    required=False,
                    default=True
                )
            ],
            result_schema_path="app.schemas.analysis.results.table_structure.TableStructureResult",
            implementation_path="app.services.analysis.configs.definitions.table_analysis.steps.table_structure.TableStructureStep",
            is_active=True
        )
    
    def get_default_algorithm(self) -> Optional[AlgorithmSelection]:
        return AlgorithmSelection(
            code="table_structure",
            version="1.0.0",
            parameters=[]
        )
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        pass
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if not input_data.get("tables"):
            raise ValueError("No tables provided in input data")
    
    async def validate_result(self, result: Dict[str, Any]) -> None:
        """Validate step execution result"""
        if not result.get("structures"):
            raise ValueError("No table structures detected in the result")
    
    async def prepare_execution(self, document_path: str, previous_results: Dict[str, Dict[str, Any]] = {}) -> Dict[str, Any]:
        """Prepare data for step execution"""
        table_detection_results = previous_results.get("table_detection", {})
        return {
            "document_path": document_path,
            "tables": table_detection_results.get("tables", []),
            "previous_results": previous_results
        }
    
    async def post_process_result(self, result: Dict[str, Any], previous_results: Dict[str, Dict[str, Any]] = {}) -> Dict[str, Any]:
        """Post-process algorithm execution result"""
        return {
            "structures": result.get("structures", []),
            "metadata": result.get("metadata", {})
        }
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass
        