
from app.analysis.base.base_step import BaseStep
from app.analysis.registry.components import AnalysisStepInfo, AnalysisIdentifier
from app.schemas.analysis import Parameter, AnalysisStepResult
from typing import Dict, Any, Optional

class TableStructureStep(BaseStep):
    """Table structure step implementation"""
    
    def get_info(self) -> AnalysisStepInfo:
        return AnalysisStepInfo(
            identifier=AnalysisIdentifier(
                name="Table Structure",
                code="table_structure",
                version="1.0.0"
            ),
            description="Extract table structure from the detected tables",
            order=2,
            base_parameters=[
                Parameter(
                    name="consider_headers",
                    description="Whether to consider headers in the table structure",
                    type="boolean",
                    required=False,
                    default=True
                ),
                Parameter(
                    name="consider_merged_cells",
                    description="Whether to consider merged cells in the table structure",
                    type="boolean",
                    required=False,
                    default=True
                )
            ],
            result_schema="app.schemas.results.table_structure.TableStructureOutput",
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
        