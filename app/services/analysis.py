import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from sqlalchemy.orm import Session
import os

from app.db.models.document import Document, AnalysisResult
from app.schemas.analysis import (
    AnalysisType,
    AnalysisStatus,
    AnalysisRequest,
    TableDetectionParameters,
    TextExtractionParameters,
    TextSummarizationParameters,
    TemplateConversionParameters,
)
from app.services.ml.factory import (
    TableDetectionFactory,
    TextExtractionFactory,
    TextSummarizationFactory,
    TemplateConversionFactory,
    DocumentClassificationFactory,
    EntityExtractionFactory,
    DocumentComparisonFactory
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for handling document analysis."""

    def __init__(self, db: Session):
        self.db = db

    def create_analysis(
        self, document_id: str, analysis_type: AnalysisType, parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Create a new analysis task."""
        logger.debug(f"Creating analysis task - Document: {document_id}, Type: {analysis_type}")

        # Create analysis record
        analysis = AnalysisResult(
            id=str(uuid.uuid4()),
            document_id=document_id,
            analysis_type=analysis_type,
            status=AnalysisStatus.PENDING,
            parameters=parameters,
            created_at=datetime.utcnow(),
        )
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)

        logger.info(f"Analysis task created - ID: {analysis.id}")
        return analysis

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis by ID."""
        return self.db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()

    def get_document_analyses(self, document_id: str) -> list[AnalysisResult]:
        """Get all analyses for a document."""
        return self.db.query(AnalysisResult).filter(
            AnalysisResult.document_id == document_id
        ).all()

    def update_analysis_status(
        self,
        analysis_id: str,
        status: AnalysisStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> AnalysisResult:
        """Update analysis status and results."""
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis not found: {analysis_id}")

        analysis.status = status
        if status == AnalysisStatus.COMPLETED:
            analysis.result = result
            analysis.completed_at = datetime.utcnow()
        elif status == AnalysisStatus.FAILED:
            analysis.error = error
            analysis.completed_at = datetime.utcnow()

        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis

    async def process_analysis(self, analysis_id: str) -> AnalysisResult:
        """Process an analysis task."""
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis not found: {analysis_id}")

        try:
            # Update status to processing
            analysis = self.update_analysis_status(analysis_id, AnalysisStatus.PROCESSING)

            # Get document
            document = self.db.query(Document).filter(Document.id == analysis.document_id).first()
            if not document:
                raise ValueError(f"Document not found: {analysis.document_id}")

            # Process based on analysis type
            processor_map = {
                AnalysisType.TABLE_DETECTION: self._process_table_detection,
                AnalysisType.TEXT_EXTRACTION: self._process_text_extraction,
                AnalysisType.TEXT_SUMMARIZATION: self._process_text_summarization,
                AnalysisType.TEMPLATE_CONVERSION: self._process_template_conversion,
                AnalysisType.DOCUMENT_CLASSIFICATION: self._process_document_classification,
                AnalysisType.ENTITY_EXTRACTION: self._process_entity_extraction,
                AnalysisType.DOCUMENT_COMPARISON: self._process_document_comparison,
            }

            processor = processor_map.get(analysis.analysis_type)
            if not processor:
                raise ValueError(f"Unsupported analysis type: {analysis.analysis_type}")

            # Process document
            result = await processor(document, analysis.parameters)

            # Update with success
            return self.update_analysis_status(
                analysis_id,
                AnalysisStatus.COMPLETED,
                result=result
            )

        except Exception as e:
            logger.error(f"Analysis failed - ID: {analysis_id}, Error: {str(e)}")
            # Update with failure
            return self.update_analysis_status(
                analysis_id,
                AnalysisStatus.FAILED,
                error=str(e)
            )

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

    async def _process_document_classification(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document classification."""
        logger.debug(f"Processing document classification for document: {document.id}")
        
        try:
            # Get classifier
            classifier = DocumentClassificationFactory.get_classifier()
            if not classifier:
                raise ValueError("Document classification service not available")
            
            # Validate file
            if not classifier.validate_file(document.file_path):
                raise ValueError("Invalid file or unsupported format")
            
            # Classify document
            return classifier.classify_document(
                document.file_path,
                confidence_threshold=parameters.get("confidence_threshold", 0.5)
            )
            
        except Exception as e:
            logger.error(f"Document classification failed: {str(e)}")
            raise

    async def _process_entity_extraction(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process entity extraction."""
        logger.debug(f"Processing entity extraction for document: {document.id}")
        
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
            
            # Get entity extractor
            extractor = EntityExtractionFactory.get_extractor()
            if not extractor:
                raise ValueError("Entity extraction service not available")
            
            # Extract entities
            return extractor.extract_entities(
                text,
                entity_types=parameters.get("entity_types"),
                confidence_threshold=parameters.get("confidence_threshold", 0.5)
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            raise

    async def _process_document_comparison(
        self, document: Document, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document comparison."""
        logger.debug(f"Processing document comparison for document: {document.id}")
        
        try:
            # Get comparison document
            comparison_doc_id = parameters.get("comparison_document_id")
            if not comparison_doc_id:
                raise ValueError("No comparison document specified")
            
            comparison_doc = self.db.query(Document).filter(
                Document.id == comparison_doc_id
            ).first()
            if not comparison_doc:
                raise ValueError(f"Comparison document not found: {comparison_doc_id}")
            
            # Get comparer
            comparer = DocumentComparisonFactory.get_comparer()
            if not comparer:
                raise ValueError("Document comparison service not available")
            
            # Validate files
            if not comparer.validate_file(document.file_path):
                raise ValueError("Invalid source file or unsupported format")
            if not comparer.validate_file(comparison_doc.file_path):
                raise ValueError("Invalid comparison file or unsupported format")
            
            # Compare documents
            return comparer.compare_documents(
                document.file_path,
                comparison_doc.file_path,
                comparison_type=parameters.get("comparison_type", "content")
            )
            
        except Exception as e:
            logger.error(f"Document comparison failed: {str(e)}")
            raise 