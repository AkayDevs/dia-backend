from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
import logging

from app.core.auth import get_current_active_verified_user
from app.db import deps
from app.db.models.user import User
from app.db.models.document import Document
from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisType,
    AnalysisStatus,
)
from app.services.analysis import AnalysisService

router = APIRouter()
logger = logging.getLogger("app.api.analysis")


@router.post("/{document_id}/analyze", response_model=AnalysisResult)
async def create_analysis(
    *,
    document_id: str,
    analysis_in: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Create a new analysis task for a document.
    """
    logger.debug(f"Creating analysis - Document: {document_id}, Type: {analysis_in.analysis_type}")
    
    # Check if document exists and belongs to user
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Create analysis service
    analysis_service = AnalysisService(db)
    
    # Create analysis task
    analysis = analysis_service.create_analysis(
        document_id=document_id,
        analysis_type=analysis_in.analysis_type,
        parameters=analysis_in.parameters,
    )
    
    # Add processing task to background tasks
    background_tasks.add_task(analysis_service.process_analysis, analysis.id)
    
    return analysis


@router.get("/{document_id}/analyses", response_model=List[AnalysisResult])
async def list_analyses(
    document_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    List all analyses for a document.
    """
    logger.debug(f"Listing analyses for document: {document_id}")
    
    # Check if document exists and belongs to user
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get analyses
    analysis_service = AnalysisService(db)
    return analysis_service.get_document_analyses(document_id)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Get a specific analysis result.
    """
    logger.debug(f"Fetching analysis: {analysis_id}")
    
    # Get analysis
    analysis_service = AnalysisService(db)
    analysis = analysis_service.get_analysis(analysis_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Check if document belongs to user
    document = db.query(Document).filter(
        Document.id == analysis.document_id,
        Document.user_id == current_user.id
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return analysis


@router.get("/types", response_model=List[dict])
async def list_analysis_types() -> Any:
    """
    List all available analysis types with their parameters.
    """
    logger.debug("Listing available analysis types")
    
    return [
        {
            "type": AnalysisType.TABLE_DETECTION,
            "name": "Table Detection",
            "description": "Detect and extract tables from documents",
            "supported_formats": ["pdf", "png", "jpg", "jpeg"],
            "parameters": {
                "confidence_threshold": {
                    "type": "float",
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Minimum confidence score for table detection"
                },
                "min_row_count": {
                    "type": "integer",
                    "default": 2,
                    "min": 1,
                    "description": "Minimum number of rows to consider a valid table"
                },
                "detect_headers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to detect table headers"
                }
            }
        },
        {
            "type": AnalysisType.TEXT_EXTRACTION,
            "name": "Text Extraction",
            "description": "Extract text content from documents",
            "supported_formats": ["pdf", "png", "jpg", "jpeg", "docx"],
            "parameters": {
                "confidence_threshold": {
                    "type": "float",
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Minimum confidence score for text extraction"
                },
                "extract_layout": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to preserve document layout"
                },
                "detect_lists": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to detect and format lists"
                }
            }
        },
        {
            "type": AnalysisType.TEXT_SUMMARIZATION,
            "name": "Text Summarization",
            "description": "Generate concise summaries of text content",
            "supported_formats": ["pdf", "docx", "txt"],
            "parameters": {
                "max_length": {
                    "type": "integer",
                    "default": 150,
                    "min": 50,
                    "max": 500,
                    "description": "Maximum length of the summary in words"
                },
                "min_length": {
                    "type": "integer",
                    "default": 50,
                    "min": 20,
                    "max": 200,
                    "description": "Minimum length of the summary in words"
                }
            }
        },
        {
            "type": AnalysisType.TEMPLATE_CONVERSION,
            "name": "Template Conversion",
            "description": "Convert documents to different formats",
            "supported_formats": ["pdf", "docx"],
            "parameters": {
                "target_format": {
                    "type": "string",
                    "default": "docx",
                    "enum": ["pdf", "docx"],
                    "description": "Target format for conversion"
                },
                "preserve_styles": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to preserve document styles"
                }
            }
        }
    ] 