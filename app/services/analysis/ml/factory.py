from typing import Dict, Any, Optional, Type, List
from abc import ABC, abstractmethod
import logging

from app.enums.analysis import AnalysisType, AnalysisMode
from app.enums.table_analysis import (
    TableAnalysisStep,
    TableDetectionAlgorithm,
    TableStructureRecognitionAlgorithm,
    TableDataExtractionAlgorithm
)
from app.schemas.table_analysis import (
    TableAnalysisDetectionResult,
    TableAnalysisStructureRecognitionResult,
    TableAnalysisDataExtractionResult,
    DetectedTable,
    RecognizedTableStructure,
    ExtractedTableData
)
from app.exceptions.analysis import AnalysisError

logger = logging.getLogger(__name__)

class BaseAnalysisFactory(ABC):
    """Base factory for analysis operations."""

    @abstractmethod
    async def analyze(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute analysis operation."""
        pass


class TableAnalysisFactory(BaseAnalysisFactory):
    """Factory for table analysis operations."""

    def __init__(self):
        self.detection_handlers = {
            TableDetectionAlgorithm.MSA: self._handle_msa_detection,
            TableDetectionAlgorithm.PYTHON_CUSTOM: self._handle_custom_detection,
            TableDetectionAlgorithm.YOLO: self._handle_yolo_detection
        }
        
        self.structure_handlers = {
            TableStructureRecognitionAlgorithm.MSA: self._handle_msa_structure,
            TableStructureRecognitionAlgorithm.PYTHON_CUSTOM: self._handle_custom_structure,
            TableStructureRecognitionAlgorithm.YOLO: self._handle_yolo_structure
        }
        
        self.extraction_handlers = {
            TableDataExtractionAlgorithm.MSA: self._handle_msa_extraction,
            TableDataExtractionAlgorithm.PYTHON_CUSTOM: self._handle_custom_extraction,
            TableDataExtractionAlgorithm.YOLO: self._handle_yolo_extraction
        }

    async def analyze(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute table analysis operation."""
        try:
            mode = parameters.get("mode", AnalysisMode.AUTOMATIC)
            
            if mode == AnalysisMode.AUTOMATIC:
                return await self._handle_automatic_analysis(document_path, parameters)
            else:
                current_step = parameters.get("current_step")
                if not current_step:
                    raise ValueError("Current step must be specified for step-by-step mode")
                return await self._handle_step_analysis(
                    document_path,
                    current_step,
                    parameters,
                    previous_result
                )
                
        except Exception as e:
            logger.error(f"Error in table analysis: {str(e)}")
            raise AnalysisError(f"Table analysis failed: {str(e)}")

    async def _handle_automatic_analysis(
        self,
        document_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle automatic mode analysis."""
        try:
            # Execute detection
            detection_result = await self._handle_detection(
                document_path,
                parameters.get("table_detection_parameters", {})
            )

            # Execute structure recognition
            structure_result = await self._handle_structure(
                document_path,
                parameters.get("table_structure_recognition_parameters", {}),
                detection_result
            )

            # Execute data extraction
            extraction_result = await self._handle_extraction(
                document_path,
                parameters.get("table_data_extraction_parameters", {}),
                structure_result
            )

            return {
                "detection": detection_result,
                "structure": structure_result,
                "extraction": extraction_result
            }

        except Exception as e:
            logger.error(f"Error in automatic table analysis: {str(e)}")
            raise

    async def _handle_step_analysis(
        self,
        document_path: str,
        current_step: str,
        parameters: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle step-by-step mode analysis."""
        try:
            if current_step == TableAnalysisStep.DETECTION:
                return await self._handle_detection(
                    document_path,
                    parameters.get("table_detection_parameters", {})
                )
            elif current_step == TableAnalysisStep.STRUCTURE_RECOGNITION:
                if not previous_result:
                    raise ValueError("Previous detection result required for structure recognition")
                return await self._handle_structure(
                    document_path,
                    parameters.get("table_structure_recognition_parameters", {}),
                    previous_result
                )
            elif current_step == TableAnalysisStep.DATA_EXTRACTION:
                if not previous_result:
                    raise ValueError("Previous structure result required for data extraction")
                return await self._handle_extraction(
                    document_path,
                    parameters.get("table_data_extraction_parameters", {}),
                    previous_result
                )
            else:
                raise ValueError(f"Invalid step: {current_step}")

        except Exception as e:
            logger.error(f"Error in step-by-step table analysis: {str(e)}")
            raise

    async def _handle_detection(
        self,
        document_path: str,
        parameters: Dict[str, Any]
    ) -> TableAnalysisDetectionResult:
        """Handle table detection step."""
        try:
            algorithm = parameters.get("algorithm", TableDetectionAlgorithm.MSA)
            handler = self.detection_handlers.get(algorithm)
            if not handler:
                raise ValueError(f"No handler found for detection algorithm: {algorithm}")
            
            return await handler(document_path, parameters.get("parameters", {}))
            
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            raise

    async def _handle_structure(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        detection_result: TableAnalysisDetectionResult
    ) -> TableAnalysisStructureRecognitionResult:
        """Handle table structure recognition step."""
        try:
            algorithm = parameters.get("algorithm", TableStructureRecognitionAlgorithm.MSA)
            handler = self.structure_handlers.get(algorithm)
            if not handler:
                raise ValueError(f"No handler found for structure recognition algorithm: {algorithm}")
            
            return await handler(document_path, parameters.get("parameters", {}), detection_result)
            
        except Exception as e:
            logger.error(f"Error in structure recognition: {str(e)}")
            raise

    async def _handle_extraction(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        structure_result: TableAnalysisStructureRecognitionResult
    ) -> TableAnalysisDataExtractionResult:
        """Handle table data extraction step."""
        try:
            algorithm = parameters.get("algorithm", TableDataExtractionAlgorithm.MSA)
            handler = self.extraction_handlers.get(algorithm)
            if not handler:
                raise ValueError(f"No handler found for data extraction algorithm: {algorithm}")
            
            return await handler(document_path, parameters.get("parameters", {}), structure_result)
            
        except Exception as e:
            logger.error(f"Error in data extraction: {str(e)}")
            raise

    # Algorithm-specific handlers for detection
    async def _handle_msa_detection(
        self,
        document_path: str,
        parameters: Dict[str, Any]
    ) -> TableAnalysisDetectionResult:
        """Handle MSA table detection."""
        # Implementation for MSA detection
        pass

    async def _handle_custom_detection(
        self,
        document_path: str,
        parameters: Dict[str, Any]
    ) -> TableAnalysisDetectionResult:
        """Handle custom table detection."""
        # Implementation for custom detection
        pass

    async def _handle_yolo_detection(
        self,
        document_path: str,
        parameters: Dict[str, Any]
    ) -> TableAnalysisDetectionResult:
        """Handle YOLO table detection."""
        # Implementation for YOLO detection
        pass

    # Algorithm-specific handlers for structure recognition
    async def _handle_msa_structure(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        detection_result: TableAnalysisDetectionResult
    ) -> TableAnalysisStructureRecognitionResult:
        """Handle MSA structure recognition."""
        # Implementation for MSA structure recognition
        pass

    async def _handle_custom_structure(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        detection_result: TableAnalysisDetectionResult
    ) -> TableAnalysisStructureRecognitionResult:
        """Handle custom structure recognition."""
        # Implementation for custom structure recognition
        pass

    async def _handle_yolo_structure(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        detection_result: TableAnalysisDetectionResult
    ) -> TableAnalysisStructureRecognitionResult:
        """Handle YOLO structure recognition."""
        # Implementation for YOLO structure recognition
        pass

    # Algorithm-specific handlers for data extraction
    async def _handle_msa_extraction(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        structure_result: TableAnalysisStructureRecognitionResult
    ) -> TableAnalysisDataExtractionResult:
        """Handle MSA data extraction."""
        # Implementation for MSA extraction
        pass

    async def _handle_custom_extraction(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        structure_result: TableAnalysisStructureRecognitionResult
    ) -> TableAnalysisDataExtractionResult:
        """Handle custom data extraction."""
        # Implementation for custom extraction
        pass

    async def _handle_yolo_extraction(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        structure_result: TableAnalysisStructureRecognitionResult
    ) -> TableAnalysisDataExtractionResult:
        """Handle YOLO data extraction."""
        # Implementation for YOLO extraction
        pass


class TextAnalysisFactory(BaseAnalysisFactory):
    """Factory for text extraction operations."""
    # ... existing code ...
    pass

class TemplateConversionFactory(BaseAnalysisFactory):
    """Factory for template conversion operations."""
    # ... existing code ...
    pass

def get_analysis_factory(analysis_type: AnalysisType) -> BaseAnalysisFactory:
    """Get appropriate factory for analysis type."""
    factories: Dict[AnalysisType, Type[BaseAnalysisFactory]] = {
        AnalysisType.TABLE_ANALYSIS: TableAnalysisFactory,
        AnalysisType.TEXT_ANALYSIS: TextAnalysisFactory,
        AnalysisType.TEMPLATE_CONVERSION: TemplateConversionFactory
    }

    factory_class = factories.get(analysis_type)
    if not factory_class:
        raise ValueError(f"No factory found for analysis type: {analysis_type}")

    return factory_class() 