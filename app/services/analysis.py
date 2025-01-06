from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from sqlalchemy.orm import Session
import os
from pathlib import Path
import logging
import asyncio
from fastapi import HTTPException, status

from app.db.models.document import Document
from app.db.models.analysis_result import AnalysisResult
from app.core.config import settings
from app.crud.crud_analysis import analysis_result as crud_analysis
from app.crud.crud_document import document as crud_document
from app.schemas.analysis import (
    AnalysisType,
    AnalysisStatus,
    AnalysisRequest,
    AnalysisResult as AnalysisResultSchema,
    AnalysisResultCreate,
    AnalysisResultUpdate,
    TableDetectionParameters,
    TextExtractionParameters,
    TextSummarizationParameters,
    TemplateConversionParameters
)
from app.services.ml.factory import (
    TableDetectionFactory,
    TextExtractionFactory,
    TextSummarizationFactory,
    TemplateConversionFactory
)

logger = logging.getLogger(__name__)

class DocumentNotFoundError(Exception):
    """Raised when document is not found."""
    pass

class ProgressTracker:
    """Track progress of long-running analysis tasks."""
    
    def __init__(self, analysis_id: str, db: Session):
        self.analysis_id = analysis_id
        self.db = db
        self._progress = 0
        
    async def update(self, progress: float, status_message: str = None):
        """Update analysis progress."""
        self._progress = min(max(progress, 0), 100)  # Ensure between 0-100
        
        # Update analysis record with progress
        update_data = {
            "progress": self._progress,
            "status_message": status_message
        }
        
        if self._progress >= 100:
            update_data["status"] = AnalysisStatus.COMPLETED
            
        crud_analysis.update(
            self.db,
            analysis_id=self.analysis_id,
            obj_in=update_data
        )

class AnalysisService:
    """Service for handling document analysis with proper error handling and validation."""

    def __init__(self, db: Session):
        """Initialize service with database session."""
        self.db = db
        self.factories = {
            AnalysisType.TABLE_DETECTION: TableDetectionFactory(),
            AnalysisType.TEXT_EXTRACTION: TextExtractionFactory(),
            AnalysisType.TEXT_SUMMARIZATION: TextSummarizationFactory(),
            AnalysisType.TEMPLATE_CONVERSION: TemplateConversionFactory()
        }

    async def process_analysis(self, analysis_id: str) -> AnalysisResult:
        """
        Process an analysis task with comprehensive error handling and progress tracking.
        
        Args:
            analysis_id: ID of the analysis to process
            
        Returns:
            Processed analysis result
            
        Raises:
            Various exceptions based on processing errors
        """
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis not found: {analysis_id}")

        progress_tracker = ProgressTracker(analysis_id, self.db)
        
        try:
            # Update status to processing
            self.update_analysis_status(
                analysis_id,
                AnalysisStatus.PROCESSING,
                status_message="Initializing analysis"
            )

            # Get document
            document = self.db.query(Document).filter(
                Document.id == analysis.document_id
            ).first()
            
            if not document:
                raise ValueError(f"Document not found: {analysis.document_id}")

            # Get appropriate factory
            factory = self.factories.get(analysis.type)
            if not factory:
                raise ValueError(f"Unsupported analysis type: {analysis.type}")

            # Initialize progress
            await progress_tracker.update(0, "Loading models")
            
            # Process based on analysis type
            result = await self._process_with_type(
                factory,
                document,
                analysis.type,
                analysis.parameters,
                progress_tracker
            )

            # Update with final result
            self.update_analysis_status(
                analysis_id,
                AnalysisStatus.COMPLETED,
                result=result
            )
            
            await progress_tracker.update(100, "Analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Error processing analysis {analysis_id}: {str(e)}")
            self.update_analysis_status(
                analysis_id,
                AnalysisStatus.FAILED,
                error=str(e)
            )
            raise

    async def _process_with_type(
        self,
        factory: Any,
        document: Document,
        analysis_type: AnalysisType,
        parameters: Dict[str, Any],
        progress_tracker: ProgressTracker
    ) -> Dict[str, Any]:
        """Process document with specific analysis type."""
        
        if analysis_type == AnalysisType.TABLE_DETECTION:
            await progress_tracker.update(20, "Detecting tables")
            result = await factory.process(document.file_path, parameters)
            
        elif analysis_type == AnalysisType.TEXT_EXTRACTION:
            await progress_tracker.update(20, "Extracting text")
            result = await factory.process(document.file_path, parameters)
            
            # Additional NLP processing if needed
            if parameters.get("perform_nlp", False):
                await progress_tracker.update(60, "Performing NLP analysis")
                # Add NLP processing here
                
        elif analysis_type == AnalysisType.TEXT_SUMMARIZATION:
            # First extract text if needed
            if not document.extracted_text:
                await progress_tracker.update(20, "Extracting text for summarization")
                text_factory = TextExtractionFactory()
                text_result = await text_factory.process(document.file_path, {})
                text = text_result["text"]
            else:
                text = document.extracted_text
                
            await progress_tracker.update(40, "Generating summary")
            result = await factory.process(text, parameters)
            
        elif analysis_type == AnalysisType.TEMPLATE_CONVERSION:
            await progress_tracker.update(30, "Converting template")
            result = await factory.process(document.file_path, parameters)
            
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
        await progress_tracker.update(90, "Finalizing results")
        return result

    def create_analysis(
        self,
        document_id: str,
        analysis_type: AnalysisType,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Create a new analysis task with validation."""
        logger.info(f"Creating analysis task - Document: {document_id}, Type: {analysis_type}")

        try:
            # Verify document exists
            document = crud_document.get(self.db, id=document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Create analysis record
            analysis_create = AnalysisResultCreate(
                id=str(uuid.uuid4()),
                document_id=document_id,
                type=analysis_type,
                status=AnalysisStatus.PENDING,
                parameters=parameters,
                progress=0,
                created_at=datetime.utcnow()
            )
            
            analysis = crud_analysis.create(self.db, obj_in=analysis_create)
            logger.info(f"Analysis task created successfully - ID: {analysis.id}")
            return analysis

        except Exception as e:
            logger.error(f"Error creating analysis task: {str(e)}")
            raise

    def update_analysis_status(
        self,
        analysis_id: str,
        status: AnalysisStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        status_message: Optional[str] = None
    ) -> AnalysisResult:
        """Update analysis status and results with validation."""
        try:
            analysis = crud_analysis.get(self.db, id=analysis_id)
            if not analysis:
                raise ValueError(f"Analysis not found: {analysis_id}")

            # Prepare update data
            update_data = AnalysisResultUpdate(
                status=status,
                result=result if status == AnalysisStatus.COMPLETED else None,
                error=error if status == AnalysisStatus.FAILED else None,
                status_message=status_message,
                completed_at=datetime.utcnow() if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED] else None
            )

            analysis = crud_analysis.update(self.db, db_obj=analysis, obj_in=update_data)
            logger.info(f"Analysis {analysis_id} status updated to {status}")
            return analysis

        except Exception as e:
            logger.error(f"Error updating analysis status: {str(e)}")
            raise

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """
        Get analysis by ID with proper error handling.
        
        Args:
            analysis_id: ID of the analysis to retrieve
            
        Returns:
            Analysis result record if found, None otherwise
        """
        try:
            return crud_analysis.get(self.db, id=analysis_id)
        except Exception as e:
            logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
            raise

    def get_document_analyses(
        self,
        document_id: str,
        analysis_type: Optional[AnalysisType] = None,
        status: Optional[AnalysisStatus] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[AnalysisResult]:
        """
        Get all analyses for a document with filtering and pagination.
        
        Args:
            document_id: ID of the document
            analysis_type: Optional filter by analysis type
            status: Optional filter by status
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of analysis results
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        try:
            # Verify document exists
            document = crud_document.get(self.db, id=document_id)
            if not document:
                raise DocumentNotFoundError(f"Document not found: {document_id}")

            return crud_analysis.get_document_results(
                self.db,
                document_id=document_id,
                analysis_type=analysis_type,
                status=status,
                skip=skip,
                limit=limit
            )

        except Exception as e:
            logger.error(f"Error retrieving analyses for document {document_id}: {str(e)}")
            raise

    def _validate_parameters(self, analysis_type: AnalysisType, parameters: Dict[str, Any]) -> None:
        """Validate analysis parameters based on type."""
        parameter_models = {
            AnalysisType.TABLE_DETECTION: TableDetectionParameters,
            AnalysisType.TEXT_EXTRACTION: TextExtractionParameters,
            AnalysisType.TEXT_SUMMARIZATION: TextSummarizationParameters,
            AnalysisType.TEMPLATE_CONVERSION: TemplateConversionParameters,
        }

        model = parameter_models.get(analysis_type)
        if model:
            try:
                model(**parameters)
            except Exception as e:
                raise ValueError(f"Invalid parameters for {analysis_type}: {str(e)}")

    def _validate_status_transition(
        self,
        current_status: AnalysisStatus,
        new_status: AnalysisStatus
    ) -> None:
        """Validate analysis status transition."""
        valid_transitions = {
            AnalysisStatus.PENDING: {AnalysisStatus.PROCESSING},
            AnalysisStatus.PROCESSING: {AnalysisStatus.COMPLETED, AnalysisStatus.FAILED},
            AnalysisStatus.COMPLETED: set(),  # No transitions from completed
            AnalysisStatus.FAILED: set(),     # No transitions from failed
        }

        if new_status not in valid_transitions[current_status]:
            raise ValueError(
                f"Invalid status transition from {current_status} to {new_status}"
            )

    def _get_processor(self, analysis_type: AnalysisType):
        """Get appropriate processor for analysis type."""
        return {
            AnalysisType.TABLE_DETECTION: self._process_table_detection,
            AnalysisType.TEXT_EXTRACTION: self._process_text_extraction,
            AnalysisType.TEXT_SUMMARIZATION: self._process_text_summarization,
            AnalysisType.TEMPLATE_CONVERSION: self._process_template_conversion,
            AnalysisType.DOCUMENT_CLASSIFICATION: self._process_document_classification,
            AnalysisType.ENTITY_EXTRACTION: self._process_entity_extraction,
            AnalysisType.DOCUMENT_COMPARISON: self._process_document_comparison,
        }.get(analysis_type)

    async def _process_table_detection(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process table detection."""
        logger.debug(f"Processing table detection for document: {document.id}")
        
        try:
            # Validate parameters
            confidence_threshold = float(parameters.get("confidence_threshold", 0.5))
            min_row_count = int(parameters.get("min_row_count", 2))
            
            # Validate document path
            if not os.path.exists(document.file_path):
                raise ValueError(f"Document file not found: {document.file_path}")
            
            # Get appropriate detector
            detector = TableDetectionFactory.get_detector(document.file_path)
            if not detector:
                raise ValueError("Unsupported file type for table detection")
            
            # Validate file
            if not detector.validate_file(document.file_path):
                raise ValueError("Invalid file or unsupported format")
            
            # Detect tables
            tables = detector.detect_tables(
                document.file_path,
                confidence_threshold=confidence_threshold,
                min_row_count=min_row_count
            )
            
            # Format results
            return {
                "tables": tables,
                "page_numbers": [table.get("page_number", 1) for table in tables],
                "confidence_scores": [table["confidence"] for table in tables]
            }
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            raise

    async def _process_text_extraction(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text extraction."""
        logger.debug(f"Processing text extraction for document: {document.id}")
        
        try:
            # Validate parameters
            extract_layout = bool(parameters.get("extract_layout", True))
            detect_lists = bool(parameters.get("detect_lists", True))
            
            # Validate document path
            if not os.path.exists(document.file_path):
                raise ValueError(f"Document file not found: {document.file_path}")
            
            # Get appropriate extractor
            extractor = TextExtractionFactory.get_extractor(document.file_path)
            if not extractor:
                raise ValueError("Unsupported file type for text extraction")
            
            # Validate file
            if not extractor.validate_file(document.file_path):
                raise ValueError("Invalid file or unsupported format")
            
            # Extract text
            return extractor.extract_text(
                document.file_path,
                extract_layout=extract_layout,
                detect_lists=detect_lists
            )
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise

    async def _process_text_summarization(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text summarization."""
        logger.debug(f"Processing text summarization for document: {document.id}")
        
        try:
            # First extract text from document
            text_result = await self._process_text_extraction(document, {
                "extract_layout": False,
                "detect_lists": False
            })
            
            # Get text content
            text = text_result.get("text", "")
            if not text:
                raise ValueError("No text content found in document")
            
            # Get summarizer
            summarizer = TextSummarizationFactory.get_summarizer()
            if not summarizer:
                raise ValueError("Text summarization service not available")
            
            # Summarize text
            return summarizer.summarize_text(
                text,
                max_length=parameters.get("max_length", 150),
                min_length=parameters.get("min_length", 50)
            )
            
        except Exception as e:
            logger.error(f"Text summarization failed: {str(e)}")
            raise

    async def _process_template_conversion(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process template conversion."""
        logger.debug(f"Processing template conversion for document: {document.id}")
        
        try:
            # Get source format from file extension
            source_format = document.file_path.split('.')[-1].lower()
            
            # Get converter
            converter = TemplateConversionFactory.get_converter(source_format)
            if not converter:
                raise ValueError(f"Unsupported source format: {source_format}")
            
            # Validate file
            if not converter.validate_file(document.file_path):
                raise ValueError("Invalid file or unsupported format")
            
            # Convert document
            return converter.convert_template(
                document.file_path,
                target_format=parameters.get("target_format", "docx"),
                preserve_styles=parameters.get("preserve_styles", True)
            )
            
        except Exception as e:
            logger.error(f"Template conversion failed: {str(e)}")
            raise