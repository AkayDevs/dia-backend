from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Query
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
    AnalysisParameters,
    BatchAnalysisRequest,
    BatchAnalysisResponse
)
from app.services.analysis import AnalysisService

router = APIRouter()
logger = logging.getLogger("app.api.analysis")


@router.post("/batch", response_model=BatchAnalysisResponse)
async def create_batch_analysis(
    *,
    batch_request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Create multiple analysis tasks for multiple documents.
    
    Args:
        batch_request: Batch analysis request containing document IDs and analysis parameters
        background_tasks: Background tasks runner
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Batch analysis response with created analysis tasks
        
    Raises:
        HTTPException: If any document is not found or validation fails
    """
    logger.info(f"Creating batch analysis for {len(batch_request.documents)} documents")
    
    analysis_service = AnalysisService(db)
    results = []
    errors = []

    for doc_request in batch_request.documents:
        try:
            # Verify document ownership
            document = db.query(Document).filter(
                Document.id == doc_request.document_id,
                Document.user_id == current_user.id
            ).first()
            
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {doc_request.document_id}"
                )

            # Create analysis task
            analysis = analysis_service.create_analysis(
                document_id=doc_request.document_id,
                analysis_type=doc_request.analysis_type,
                parameters=doc_request.parameters
            )
            
            # Schedule processing
            background_tasks.add_task(
                analysis_service.process_analysis,
                analysis.id
            )
            
            results.append(analysis)
            
        except Exception as e:
            errors.append({
                "document_id": doc_request.document_id,
                "error": str(e)
            })

    if errors and not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "All analysis requests failed", "errors": errors}
        )

    return {
        "results": results,
        "errors": errors if errors else None,
        "total_submitted": len(results),
        "total_failed": len(errors)
    }


@router.post("/{document_id}", response_model=AnalysisResult)
async def create_analysis(
    *,
    document_id: str,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Create a new analysis task for a document.
    
    Args:
        document_id: Document ID
        analysis_request: Analysis request parameters
        background_tasks: Background tasks runner
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Created analysis task
        
    Raises:
        HTTPException: If document is not found or validation fails
    """
    logger.info(f"Creating analysis - Document: {document_id}, Type: {analysis_request.analysis_type}")
    
    # Verify document ownership
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Create analysis service
        analysis_service = AnalysisService(db)
        
        # Create analysis task
        analysis = analysis_service.create_analysis(
            document_id=document_id,
            analysis_type=analysis_request.analysis_type,
            parameters=analysis_request.parameters
        )
        
        # Schedule processing
        background_tasks.add_task(
            analysis_service.process_analysis,
            analysis.id
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error creating analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/document/{document_id}", response_model=List[AnalysisResult])
async def list_document_analyses(
    document_id: str,
    analysis_type: Optional[AnalysisType] = None,
    status: Optional[AnalysisStatus] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    List all analyses for a document with optional filtering.
    
    Args:
        document_id: Document ID
        analysis_type: Filter by analysis type
        status: Filter by analysis status
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        List of analysis results
        
    Raises:
        HTTPException: If document is not found
    """
    logger.info(f"Listing analyses for document: {document_id}")
    
    # Verify document ownership
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get analyses with filters
    analysis_service = AnalysisService(db)
    return analysis_service.get_document_analyses(
        document_id=document_id,
        analysis_type=analysis_type,
        status=status,
        skip=skip,
        limit=limit
    )


@router.get("/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Get a specific analysis result.
    
    Args:
        analysis_id: Analysis ID
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Analysis result
        
    Raises:
        HTTPException: If analysis is not found or user doesn't have access
    """
    logger.info(f"Fetching analysis: {analysis_id}")
    
    analysis_service = AnalysisService(db)
    analysis = analysis_service.get_analysis(analysis_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Verify document ownership
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
    
    Returns:
        List of analysis types with their configurations
    """
    logger.info("Listing available analysis types")
    
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