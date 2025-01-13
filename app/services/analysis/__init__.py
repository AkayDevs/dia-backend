from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from sqlalchemy.orm import Session
import uuid
import logging

from app.enums.analysis import AnalysisType, AnalysisStatus, AnalysisMode
from app.enums.table_analysis import TableAnalysisStep
from app.schemas.analysis import (
    AnalysisResult,
    AnalysisParameters,
    StepApprovalRequest
)
from app.schemas.table_analysis import (
    TableAnalysisParameters,
    TableAnalysisDetectionResult,
    TableAnalysisStructureRecognitionResult,
    TableAnalysisDataExtractionResult,
    DetectedTable,
    RecognizedTableStructure,
    ExtractedTableData
)
from app.schemas.text_analysis import TextAnalysisParameters
from app.schemas.template_conversion import TemplateConversionParameters
from app.services.analysis.ml.factory import get_analysis_factory
from app.crud.crud_analysis import analysis_result as crud_analysis
from app.exceptions.analysis import AnalysisError

logger = logging.getLogger(__name__)



# Mapping for Analysis Parameters -----------------------------------------------------

PARAMETER_MAPPINGS = {
    AnalysisType.TABLE_ANALYSIS: TableAnalysisParameters,
    AnalysisType.TEXT_ANALYSIS: TextAnalysisParameters,
    AnalysisType.TEMPLATE_CONVERSION: TemplateConversionParameters
} 


class AnalysisOrchestrator:
    """Orchestrates document analysis operations."""

    def __init__(self, db: Session):
        self.db = db

    async def start_analysis(
        self,
        document_id: str,
        analysis_type: AnalysisType,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Start a new analysis task."""
        try:
            # Validate parameters using mapping
            parameter_model = PARAMETER_MAPPINGS.get(analysis_type)
            if not parameter_model:
                raise ValueError(f"No parameter model found for analysis type: {analysis_type}")
            
            validated_params = parameter_model(**parameters)

            # Create analysis record
            analysis_id = str(uuid.uuid4())
            mode = validated_params.mode
            
            analysis = crud_analysis.create(
                self.db,
                obj_in={
                    "id": analysis_id,
                    "document_id": document_id,
                    "type": analysis_type,
                    "status": AnalysisStatus.PENDING,
                    "parameters": validated_params.model_dump(),
                    "created_at": datetime.utcnow(),
                    "mode": mode
                }
            )

            # Initialize step results for table analysis
            if analysis_type == AnalysisType.TABLE_ANALYSIS:
                step_results = {}
                for step in TableAnalysisStep:
                    step_results[step] = {
                        "status": AnalysisStatus.PENDING,
                        "created_at": datetime.utcnow()
                    }
                analysis.step_results = step_results
                if mode == AnalysisMode.STEP_BY_STEP:
                    analysis.current_step = TableAnalysisStep.DETECTION

            return analysis

        except Exception as e:
            logger.error(f"Error starting analysis: {str(e)}")
            raise AnalysisError(f"Failed to start analysis: {str(e)}")

    async def process_analysis(
        self,
        analysis_id: str,
        step: Optional[str] = None,
        step_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Process an analysis task.
        
        Args:
            analysis_id: ID of the analysis to process
            step: Optional specific step to process (for rerunning a step)
            step_parameters: Optional parameters for the specific step
        """
        try:
            # Get analysis record
            analysis = crud_analysis.get(self.db, id=analysis_id)
            if not analysis:
                raise ValueError(f"Analysis not found: {analysis_id}")

            # Update status to processing
            crud_analysis.update(
                self.db,
                db_obj=analysis,
                obj_in={"status": AnalysisStatus.PROCESSING}
            )

            # Get factory for analysis type
            factory = get_analysis_factory(analysis.type)

            if analysis.type == AnalysisType.TABLE_ANALYSIS:
                if step:
                    # Process specific step
                    await self._process_table_analysis_step(
                        analysis,
                        factory,
                        step,
                        step_parameters
                    )
                elif analysis.mode == AnalysisMode.STEP_BY_STEP:
                    # Process current step in step-by-step mode
                    await self._process_table_analysis_step(
                        analysis,
                        factory,
                        analysis.current_step,
                        step_parameters
                    )
                else:
                    # Process all steps in automatic mode
                    await self._process_table_analysis_automatic(
                        analysis,
                        factory
                    )
            else:
                # Handle other analysis types
                await self._process_generic_analysis(analysis, factory)

        except Exception as e:
            logger.error(f"Error processing analysis {analysis_id}: {str(e)}")
            crud_analysis.update(
                self.db,
                db_obj=analysis,
                obj_in={
                    "status": AnalysisStatus.FAILED,
                    "error": str(e)
                }
            )
            raise

    async def _process_table_analysis_automatic(
        self,
        analysis: AnalysisResult,
        factory: Any
    ) -> None:
        """Process all steps of table analysis in automatic mode."""
        try:
            params = TableAnalysisParameters(**analysis.parameters)
            
            # Execute detection step
            detection_result = await self._execute_table_detection(
                factory,
                analysis.document_id,
                params.table_detection_parameters
            )
            analysis.step_results[TableAnalysisStep.DETECTION] = detection_result.model_dump()

            # Execute structure recognition step
            structure_result = await self._execute_table_structure_recognition(
                factory,
                analysis.document_id,
                params.table_structure_recognition_parameters,
                detection_result
            )
            analysis.step_results[TableAnalysisStep.STRUCTURE_RECOGNITION] = structure_result.model_dump()

            # Execute data extraction step
            extraction_result = await self._execute_table_data_extraction(
                factory,
                analysis.document_id,
                params.table_data_extraction_parameters,
                structure_result
            )
            analysis.step_results[TableAnalysisStep.DATA_EXTRACTION] = extraction_result.model_dump()

            # Update analysis record
            crud_analysis.update(
                self.db,
                db_obj=analysis,
                obj_in={
                    "status": AnalysisStatus.COMPLETED,
                    "step_results": analysis.step_results,
                    "completed_at": datetime.utcnow(),
                    "progress": 1.0
                }
            )

        except Exception as e:
            raise AnalysisError(f"Error in automatic table analysis: {str(e)}")

    async def _process_table_analysis_step(
        self,
        analysis: AnalysisResult,
        factory: Any,
        step: str,
        step_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Process a single step of table analysis."""
        try:
            params = TableAnalysisParameters(**analysis.parameters)
            
            # Get step parameters
            if step_parameters:
                # Update parameters with provided ones
                if step == TableAnalysisStep.DETECTION:
                    params.table_detection_parameters = step_parameters
                elif step == TableAnalysisStep.STRUCTURE_RECOGNITION:
                    params.table_structure_recognition_parameters = step_parameters
                elif step == TableAnalysisStep.DATA_EXTRACTION:
                    params.table_data_extraction_parameters = step_parameters

            # Execute step
            if step == TableAnalysisStep.DETECTION:
                result = await self._execute_table_detection(
                    factory,
                    analysis.document_id,
                    params.table_detection_parameters
                )
            elif step == TableAnalysisStep.STRUCTURE_RECOGNITION:
                prev_result = TableAnalysisDetectionResult(**analysis.step_results[TableAnalysisStep.DETECTION])
                result = await self._execute_table_structure_recognition(
                    factory,
                    analysis.document_id,
                    params.table_structure_recognition_parameters,
                    prev_result
                )
            else:  # DATA_EXTRACTION
                prev_result = TableAnalysisStructureRecognitionResult(
                    **analysis.step_results[TableAnalysisStep.STRUCTURE_RECOGNITION]
                )
                result = await self._execute_table_data_extraction(
                    factory,
                    analysis.document_id,
                    params.table_data_extraction_parameters,
                    prev_result
                )

            # Update step result
            analysis.step_results[step] = result.model_dump()
            
            if analysis.mode == AnalysisMode.AUTOMATIC:
                status = AnalysisStatus.COMPLETED
            else:
                status = AnalysisStatus.WAITING_FOR_APPROVAL

            # Update analysis record
            crud_analysis.update(
                self.db,
                db_obj=analysis,
                obj_in={
                    "step_results": analysis.step_results,
                    "progress": self._calculate_progress(analysis),
                    "status": status
                }
            )

        except Exception as e:
            raise AnalysisError(f"Error processing table analysis step {step}: {str(e)}")

    async def process_step_approval(
        self,
        analysis_id: str,
        approval: StepApprovalRequest,
        modifications: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Process step approval/rejection with optional modifications."""
        try:
            analysis = crud_analysis.get(self.db, id=analysis_id)
            if not analysis:
                raise ValueError(f"Analysis not found: {analysis_id}")

            step_result = analysis.step_results[approval.step]
            
            if approval.action == "approve":
                # Apply modifications if any
                if modifications:
                    step_result = await self._apply_step_modifications(
                        analysis,
                        approval.step,
                        modifications
                    )

                # Update step status
                step_result["status"] = AnalysisStatus.APPROVED
                analysis.step_results[approval.step] = step_result

                if analysis.mode == AnalysisMode.STEP_BY_STEP:
                    # Move to next step
                    next_step = self._get_next_step(approval.step)
                    if next_step:
                        analysis.current_step = next_step
                        await self.process_analysis(analysis_id)
                    else:
                        # Complete analysis
                        crud_analysis.update(
                            self.db,
                            db_obj=analysis,
                            obj_in={
                                "status": AnalysisStatus.COMPLETED,
                                "completed_at": datetime.utcnow(),
                                "progress": 1.0
                            }
                        )
            else:
                # Handle rejection
                step_result["status"] = AnalysisStatus.REJECTED
                analysis.step_results[approval.step] = step_result

            crud_analysis.update(
                self.db,
                db_obj=analysis,
                obj_in={"step_results": analysis.step_results}
            )

            return analysis

        except Exception as e:
            logger.error(f"Error processing step approval: {str(e)}")
            raise AnalysisError(f"Failed to process step approval: {str(e)}")

    async def _apply_step_modifications(
        self,
        analysis: AnalysisResult,
        step: str,
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply user modifications to step results and calculate accuracy."""
        step_result = analysis.step_results[step]
        
        if step == TableAnalysisStep.DETECTION:
            # Handle detection modifications (e.g., adding/removing tables)
            modified_result = await self._apply_detection_modifications(
                step_result,
                modifications
            )
        elif step == TableAnalysisStep.STRUCTURE_RECOGNITION:
            # Handle structure modifications
            modified_result = await self._apply_structure_modifications(
                step_result,
                modifications
            )
        else:
            # Handle data extraction modifications
            modified_result = await self._apply_extraction_modifications(
                step_result,
                modifications
            )

        # Calculate accuracy based on modifications
        accuracy = self._calculate_step_accuracy(modified_result, modifications)
        modified_result["accuracy"] = accuracy

        return modified_result

    def _calculate_step_accuracy(
        self,
        step_result: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> float:
        """Calculate accuracy based on modifications."""
        # Implementation depends on your accuracy calculation logic
        # This is a placeholder
        return 0.95

    def _get_next_step(self, current_step: str) -> Optional[str]:
        """Get next step in the analysis process."""
        steps = list(TableAnalysisStep)
        try:
            current_index = steps.index(current_step)
            if current_index < len(steps) - 1:
                return steps[current_index + 1]
        except ValueError:
            pass
        return None

    def _calculate_progress(self, analysis: AnalysisResult) -> float:
        """Calculate overall analysis progress."""
        if not analysis.step_results:
            return 0.0

        total_steps = len(TableAnalysisStep)
        completed_steps = sum(
            1 for result in analysis.step_results.values()
            if result["status"] in [AnalysisStatus.APPROVED, AnalysisStatus.COMPLETED]
        )
        return completed_steps / total_steps

    async def _execute_table_detection(
        self,
        factory: Any,
        document_id: str,
        parameters: Dict[str, Any]
    ) -> TableAnalysisDetectionResult:
        """Execute table detection step."""
        # Implementation for table detection
        pass

    async def _execute_table_structure_recognition(
        self,
        factory: Any,
        document_id: str,
        parameters: Dict[str, Any],
        detection_result: TableAnalysisDetectionResult
    ) -> TableAnalysisStructureRecognitionResult:
        """Execute table structure recognition step."""
        # Implementation for structure recognition
        pass

    async def _execute_table_data_extraction(
        self,
        factory: Any,
        document_id: str,
        parameters: Dict[str, Any],
        structure_result: TableAnalysisStructureRecognitionResult
    ) -> TableAnalysisDataExtractionResult:
        """Execute table data extraction step."""
        # Implementation for data extraction
        pass 