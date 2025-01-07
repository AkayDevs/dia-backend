from typing import Optional, Dict, Any, List, Type
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
    AnalysisResultUpdate
)
from app.services.ml.factory import (
    TableDetectionFactory,
    TextExtractionFactory,
    TextSummarizationFactory,
    TemplateConversionFactory,
    FACTORY_MAP
)

logger = logging.getLogger(__name__)

class DocumentNotFoundError(Exception):
    """Exception raised when a document is not found."""
    pass

class AnalysisError(Exception):
    """Base exception for analysis-related errors."""
    pass

class ParameterValidationError(AnalysisError):
    """Exception raised when analysis parameters are invalid."""
    pass

class UnsupportedAnalysisError(AnalysisError):
    """Exception raised when analysis type is not supported for document type."""
    pass

class AnalysisOrchestrator:
    """Orchestrates the complete analysis workflow."""
    
    def __init__(self, db: Session):
        self.db = db
        self._factories = {}
        
    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis by ID."""
        return crud_analysis.get(self.db, id=analysis_id)

    def get_document_analyses(
        self,
        document_id: str,
        analysis_type: Optional[AnalysisType] = None,
        status: Optional[AnalysisStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AnalysisResult]:
        """Get all analyses for a document with filtering."""
        filters = {"document_id": document_id}
        if analysis_type:
            filters["type"] = analysis_type
        if status:
            filters["status"] = status
        return crud_analysis.get_multi_by_filter(
            self.db,
            filters=filters,
            skip=skip,
            limit=limit
        )

    def delete_analysis(self, analysis_id: str) -> None:
        """Delete an analysis record."""
        crud_analysis.remove(self.db, id=analysis_id)

    def get_supported_parameters(self, analysis_type: AnalysisType, document_type: str) -> Dict[str, Any]:
        """Get supported parameters for a specific analysis type and document type."""
        factory = self._get_factory(analysis_type)
        if not factory:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
        return factory.get_supported_parameters(document_type)
        
    def validate_parameters(self, analysis_type: AnalysisType, document_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a specific analysis type and document type."""
        factory = self._get_factory(analysis_type)
        if not factory:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
        return factory.validate_parameters(document_type, parameters)
        
    async def start_analysis(self, document_id: str, analysis_type: AnalysisType, parameters: Dict[str, Any]) -> AnalysisResult:
        """Start an analysis process."""
        try:
            # 1. Get document and validate
            document = self._get_and_validate_document(document_id)
            document_type = self._get_document_type(document.url)
            
            # 2. Get factory and check supported formats
            factory = self._get_factory(analysis_type)
            if not factory:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
            if document_type not in factory.supported_formats:
                supported = ", ".join(factory.supported_formats.keys())
                raise ValueError(
                    f"Document type '{document_type}' is not supported for {analysis_type}. "
                    f"Supported formats are: {supported}"
                )
            
            # 3. Validate parameters
            if not self.validate_parameters(analysis_type, document_type, parameters):
                raise ValueError("Invalid parameters for analysis")
                
            # 4. Create analysis record
            analysis = self._create_analysis_record(document_id, analysis_type, parameters)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to start analysis: {str(e)}")
            raise
            
    async def process_analysis(self, analysis_id: str):
        """Public method to process the analysis in background."""
        return await self._process_analysis(analysis_id)

    async def _process_analysis(self, analysis_id: str):
        """Process the analysis in background."""
        try:
            analysis = crud_analysis.get(self.db, id=analysis_id)
            if not analysis:
                raise ValueError(f"Analysis not found: {analysis_id}")
                
            # 1. Update status to processing
            self._update_analysis_status(analysis_id, AnalysisStatus.PROCESSING)
            
            # 2. Get document
            document = self._get_and_validate_document(analysis.document_id)
            
            # 3. Get factory and process
            factory = self._get_factory(analysis.type)
            if not factory:
                raise ValueError(f"Unsupported analysis type: {analysis.type}")
                
            # 4. Process with progress tracking
            progress_tracker = ProgressTracker(analysis_id, self.db)
            result = await factory.process(
                document.url,
                analysis.parameters,
                progress_callback=progress_tracker.update
            )
            
            # 5. Store results
            self._update_analysis_status(
                analysis_id,
                AnalysisStatus.COMPLETED,
                result=result
            )
            
        except Exception as e:
            logger.error(f"Analysis processing failed: {str(e)}")
            self._update_analysis_status(
                analysis_id,
                AnalysisStatus.FAILED,
                error=str(e)
            )
            
    def _get_factory(self, analysis_type: AnalysisType) -> Any:
        """Get or create factory for analysis type."""
        if analysis_type not in self._factories:
            factory_class = FACTORY_MAP.get(analysis_type)
            if not factory_class:
                return None
            self._factories[analysis_type] = factory_class()
        return self._factories[analysis_type]
        
    def _get_and_validate_document(self, document_id: str) -> Document:
        """Get and validate document exists."""
        document = crud_document.get(self.db, id=document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        return document
        
    def _get_document_type(self, url: str) -> str:
        """Get document type from URL."""
        ext = Path(url).suffix.lower()
        return ext[1:] if ext else ""
        
    def _create_analysis_record(
        self,
        document_id: str,
        analysis_type: AnalysisType,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Create analysis record in database."""
        analysis_create = AnalysisResultCreate(
            id=str(uuid.uuid4()),
            document_id=document_id,
            type=analysis_type,
            status=AnalysisStatus.PENDING,
            parameters=parameters,
            progress=0.0,
            created_at=datetime.utcnow()
        )
        try:
            return crud_analysis.create(self.db, obj_in=analysis_create)
        except Exception as e:
            logger.error(f"Failed to create analysis record: {str(e)}")
            raise ValueError(f"Failed to create analysis record: {str(e)}")
        
    def _update_analysis_status(
        self,
        analysis_id: str,
        status: AnalysisStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Update analysis status and results."""
        update_data = AnalysisResultUpdate(
            status=status,
            result=result,
            error=error,
            completed_at=datetime.utcnow() if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED] else None
        )
        crud_analysis.update(self.db, analysis_id=analysis_id, obj_in=update_data)

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
            
        crud_analysis.update(
            self.db,
            analysis_id=self.analysis_id,
            obj_in=update_data
        )