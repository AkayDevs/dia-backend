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
    BatchAnalysisResponse,
    BatchAnalysisError
)
from app.services.analysis import AnalysisOrchestrator, DocumentNotFoundError

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
    """
    logger.info(f"Creating batch analysis for {len(batch_request.documents)} documents")
    
    analysis_orchestrator = AnalysisOrchestrator(db)
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
                raise DocumentNotFoundError(f"Document not found: {doc_request.document_id}")

            # Get supported parameters for validation
            supported_params = analysis_orchestrator.get_supported_parameters(
                doc_request.analysis_type,
                document.file_type
            )
            
            # Validate parameters
            analysis_orchestrator.validate_parameters(
                doc_request.analysis_type,
                document.file_type,
                doc_request.parameters
            )

            # Create analysis task
            analysis = analysis_orchestrator.start_analysis(
                document_id=doc_request.document_id,
                analysis_type=doc_request.analysis_type,
                parameters=doc_request.parameters
            )
            
            # Schedule processing
            background_tasks.add_task(
                analysis_orchestrator.process_analysis,
                analysis.id
            )
            
            results.append(analysis)
            
        except DocumentNotFoundError as e:
            errors.append(BatchAnalysisError(
                document_id=doc_request.document_id,
                error=str(e)
            ))
        except ValueError as e:
            errors.append(BatchAnalysisError(
                document_id=doc_request.document_id,
                error=f"Parameter validation failed: {str(e)}"
            ))
        except Exception as e:
            logger.error(f"Error processing document {doc_request.document_id}: {str(e)}")
            errors.append(BatchAnalysisError(
                document_id=doc_request.document_id,
                error="Internal server error"
            ))

    if errors and not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "All analysis requests failed", "errors": errors}
        )

    return BatchAnalysisResponse(
        results=results,
        errors=errors if errors else None,
        total_submitted=len(results),
        total_failed=len(errors)
    )


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
        # Create analysis orchestrator
        analysis_orchestrator = AnalysisOrchestrator(db)
        
        # Get supported parameters for validation
        supported_params = analysis_orchestrator.get_supported_parameters(
            analysis_request.analysis_type,
            document.file_type
        )
        
        # Validate parameters
        analysis_orchestrator.validate_parameters(
            analysis_request.analysis_type,
            document.file_type,
            analysis_request.parameters
        )
        
        # Create analysis task
        analysis = analysis_orchestrator.start_analysis(
            document_id=document_id,
            analysis_type=analysis_request.analysis_type,
            parameters=analysis_request.parameters
        )
        
        # Schedule processing
        background_tasks.add_task(
            analysis_orchestrator.process_analysis,
            analysis.id
        )
        
        return analysis
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating analysis task"
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
    
    try:
        # Get analyses with filters
        analysis_orchestrator = AnalysisOrchestrator(db)
        return analysis_orchestrator.get_document_analyses(
            document_id=document_id,
            analysis_type=analysis_type,
            status=status,
            skip=skip,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Error listing analyses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving analysis results"
        )


@router.get("/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Get a specific analysis result.
    """
    logger.info(f"Fetching analysis: {analysis_id}")
    
    try:
        analysis_orchestrator = AnalysisOrchestrator(db)
        analysis = analysis_orchestrator.get_analysis(analysis_id)
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving analysis result"
        )


@router.get("/parameters/{document_id}/{analysis_type}", response_model=dict)
async def get_analysis_parameters(
    document_id: str,
    analysis_type: AnalysisType,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> Any:
    """
    Get supported parameters for a specific analysis type and document.
    """
    logger.info(f"Getting parameters for {analysis_type} on document: {document_id}")
    
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
        analysis_orchestrator = AnalysisOrchestrator(db)
        return analysis_orchestrator.get_supported_parameters(
            analysis_type,
            document.file_type
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting parameters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving parameters"
        )


@router.get("/types", response_model=List[dict])
async def list_analysis_types() -> Any:
    """
    List all available analysis types with their parameters.
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
        },
        {
            "type": AnalysisType.DOCUMENT_CLASSIFICATION,
            "name": "Document Classification",
            "description": "Classify documents into predefined categories",
            "supported_formats": ["pdf", "docx", "txt"],
            "parameters": {
                "confidence_threshold": {
                    "type": "float",
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Minimum confidence score for classification"
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include document metadata in classification"
                }
            }
        },
        {
            "type": AnalysisType.ENTITY_EXTRACTION,
            "name": "Entity Extraction",
            "description": "Extract named entities from documents",
            "supported_formats": ["pdf", "docx", "txt"],
            "parameters": {
                "confidence_threshold": {
                    "type": "float",
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Minimum confidence score for entity extraction"
                },
                "entity_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["person", "organization", "location", "date", "money"]
                    },
                    "default": ["person", "organization", "location"],
                    "description": "Types of entities to extract"
                },
                "include_context": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include surrounding context for entities"
                }
            }
        },
        {
            "type": AnalysisType.DOCUMENT_COMPARISON,
            "name": "Document Comparison",
            "description": "Compare two documents for similarities and differences",
            "supported_formats": ["pdf", "docx"],
            "parameters": {
                "comparison_type": {
                    "type": "string",
                    "default": "content",
                    "enum": ["content", "structure", "visual"],
                    "description": "Type of comparison to perform"
                },
                "include_visual_diff": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include visual difference markers"
                }
            }
        }
    ]


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user),
) -> None:
    """
    Delete a specific analysis result.
    """
    logger.info(f"Deleting analysis: {analysis_id}")
    
    try:
        analysis_orchestrator = AnalysisOrchestrator(db)
        analysis = analysis_orchestrator.get_analysis(analysis_id)
        
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
        
        analysis_orchestrator.delete_analysis(analysis_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting analysis result"
        ) 