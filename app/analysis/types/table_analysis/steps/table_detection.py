from typing import Dict, Any, Optional
from app.analysis.base.base_step import BaseStep
from app.analysis.registry.components import AnalysisStepInfo, AnalysisIdentifier
from app.schemas.analysis import Parameter, AnalysisStepResult
from app.enums.document import DocumentType
from app.schemas.results.table_detection import TableDetectionOutput

class TableDetectionStep(BaseStep):
    """Table detection step implementation"""
    
    def get_info(self) -> AnalysisStepInfo:
        return AnalysisStepInfo(
            identifier=AnalysisIdentifier(
                name="Table Detection",
                code="table_detection",
                version="1.0.0"
            ),
            description="Detect table locations in the document",
            order=1,
            base_parameters=[
                Parameter(
                    name="page_range",
                    description="Range of pages to process (e.g., '1-5' or '1,3,5')",
                    type="string",
                    required=False,
                    default="all"
                )
            ],
            result_schema="app.schemas.results.table_detection.TableDetectionOutput",
            algorithms=[]
        )
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_keys = ["document_id", "pages"]
        return all(key in input_data for key in required_keys)
    
    async def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate step result"""
        try:
            TableDetectionOutput(**result)
            return True
        except Exception:
            return False
    
    async def prepare_step(self, previous_result: Optional[AnalysisStepResult] = None) -> Dict[str, Any]:
        """Prepare data for step execution"""
        if previous_result:
            return previous_result.result
        return {}
    
    async def cleanup(self) -> None:
        """Cleanup temporary resources"""
        pass 