from typing import Dict, Any, Optional, List
from app.services.analysis.configs.base.base_step import BaseStep
from app.schemas.analysis.configs.steps import StepDefinitionInfo
from app.schemas.analysis.configs.algorithms import AlgorithmParameter

class TableDetectionStep(BaseStep):
    """Table detection step implementation"""
    
    def get_info(self) -> StepDefinitionInfo:
        return StepDefinitionInfo(
            code="table_detection",
            name="Table Detection",
            version="1.0.0",
            description="Detect table locations in the document",
            order=1,
            base_parameters=[
                AlgorithmParameter(
                    name="page_range",
                    description="Range of pages to process (e.g., '1-5' or '1,3,5')",
                    type="string",
                    required=False,
                    default="all"
                )
            ],
            result_schema_path="app.schemas.analysis.results.table_detection.TableDetectionResult",
            implementation_path="app.services.analysis.configs.definitions.table_analysis.steps.table_detection.TableDetectionStep",
            is_active=True
        )
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        pass
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        required_keys = ["document_id", "document_path"]
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required input keys: {', '.join(missing_keys)}")
    
    async def validate_result(self, result: Dict[str, Any]) -> None:
        """Validate step execution result"""
        if not result.get("tables"):
            raise ValueError("No tables detected in the result")
    
    async def prepare_execution(self, document_path: str, previous_results: Dict[str, Dict[str, Any]] = {}) -> Dict[str, Any]:
        """Prepare data for step execution"""
        return {
            "document_path": document_path,
            "previous_results": previous_results
        }
    
    async def post_process_result(self, result: Dict[str, Any], previous_results: Dict[str, Dict[str, Any]] = {}) -> Dict[str, Any]:
        """Post-process algorithm execution result"""
        return {
            "tables": result.get("tables", []),
            "page_info": result.get("page_info", {})
        }
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass 