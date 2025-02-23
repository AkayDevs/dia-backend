from typing import Dict, Any, Optional, List
from app.services.analysis.configs.base import BaseStep
from app.schemas.analysis.configs.steps import StepDefinitionBase
from app.schemas.analysis.configs.algorithms import AlgorithmParameter

class TableDataStep(BaseStep):
    """Table data step implementation"""
    
    def get_info(self) -> StepDefinitionBase:
        return StepDefinitionBase(
            code="table_data",
            name="Table Data",
            version="1.0.0",
            description="Extract data from the detected tables",
            order=3,
            base_parameters=[
                AlgorithmParameter(
                    name="extract_formulas",
                    description="Whether to extract formulas from cells",
                    type="boolean",
                    required=False,
                    default=False
                )
            ],
            result_schema_path="app.schemas.analysis.results.table_data.TableDataResult",
            implementation_path="app.services.analysis.configs.definitions.table_analysis.steps.table_data.TableDataStep",
            is_active=True
        )
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        pass
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if not input_data.get("structures"):
            raise ValueError("No table structures provided in input data")
    
    async def validate_result(self, result: Dict[str, Any]) -> None:
        """Validate step execution result"""
        if not result.get("tables"):
            raise ValueError("No table data extracted in the result")
    
    async def prepare_execution(self, document_path: str, previous_results: Dict[str, Dict[str, Any]] = {}) -> Dict[str, Any]:
        """Prepare data for step execution"""
        structure_results = previous_results.get("table_structure", {})
        return {
            "document_path": document_path,
            "structures": structure_results.get("structures", []),
            "previous_results": previous_results
        }
    
    async def post_process_result(self, result: Dict[str, Any], previous_results: Dict[str, Dict[str, Any]] = {}) -> Dict[str, Any]:
        """Post-process algorithm execution result"""
        return {
            "tables": result.get("tables", []),
            "metadata": result.get("metadata", {})
        }
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass
        