
from app.analysis.base.base_step import BaseStep
from app.analysis.registry.components import AnalysisStepInfo, AnalysisIdentifier
from app.schemas.analysis import AnalysisStepResult
from typing import Dict, Any, Optional

class TableDataStep(BaseStep):
    """Table data step implementation"""
    
    def get_info(self) -> AnalysisStepInfo:
        return AnalysisStepInfo(
            identifier=AnalysisIdentifier(
                name="Table Data",
                code="table_data",
                version="1.0.0"
            ),
            description="Extract data from the detected tables",
            order=3,
            base_parameters=[],
            result_schema="app.schemas.results.table_data.TableDataOutput",
            algorithms=[]
        )
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data for the step"""
        return True
    
    async def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate the result of the step"""
        return True
    
    async def prepare_step(self, previous_result: Optional[AnalysisStepResult] = None) -> Dict[str, Any]:
        """Prepare data for step execution"""
        if previous_result:
            return previous_result.result
        return {}
    
    async def cleanup(self) -> None:
        """Cleanup temporary resources"""
        pass
        