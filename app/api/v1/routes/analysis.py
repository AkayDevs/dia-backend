from typing import Any, List, Optional, Dict
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
from app.services.analysis import AnalysisOrchestrator
from app.exceptions import DocumentNotFoundError

router = APIRouter()
logger = logging.getLogger("app.api.analysis")


@router.get("/types", response_model=List[dict])
async def list_analysis_types(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(get_current_active_verified_user)
) -> Any:
    """
    List all available analysis types with their parameters.
    Any logged-in user can access this endpoint.
    """
    logger.info(f"User {current_user.id} listing available analysis types")
    
    try:
        # Create orchestrator to access factories
        analysis_orchestrator = AnalysisOrchestrator(db)
        
        # Build response for each analysis type
        analysis_types = []
        for analysis_type in AnalysisType:
            try:
                # Get factory instance
                factory = analysis_orchestrator._get_factory(analysis_type)
                if not factory:
                    logger.info(f"Factory not found for {analysis_type}")
                    continue
                
                # Get description and supported formats directly from factory
                description = factory.get_description()
                supported_formats = list(factory.supported_formats)
                
                # Get parameters using orchestrator
                try:
                    parameters = analysis_orchestrator.get_supported_parameters(analysis_type)
                except ValueError as e:
                    logger.warning(f"Error getting parameters for {analysis_type}: {str(e)}")
                    continue
                
                analysis_types.append({
                    "type": analysis_type,
                    "name": analysis_type.value.replace("_", " ").title(),
                    "description": description,
                    "supported_formats": supported_formats,
                    "parameters": parameters
                })
                
            except Exception as e:
                logger.warning(f"Error getting info for analysis type {analysis_type}: {str(e)}")
                continue
        
        return analysis_types
        
    except Exception as e:
        logger.error(f"Error listing analysis types: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving analysis types"
        )


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
                document.type
            )
            
            # Validate parameters
            analysis_orchestrator.validate_parameters(
                doc_request.analysis_type,
                document.type,
                doc_request.parameters
            )

            # Create analysis task
            analysis = await analysis_orchestrator.start_analysis(
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
            analysis_request.analysis_type
        )
        
        # Validate parameters
        analysis_orchestrator.validate_parameters(
            analysis_request.analysis_type,
            document.type,
            analysis_request.parameters
        )
        
        # Create analysis task
        analysis = await analysis_orchestrator.start_analysis(
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


@router.get("/types/{analysis_type}/parameters")
async def get_analysis_parameters(
    analysis_type: AnalysisType,
    db: Session = Depends(deps.get_db)
) -> Dict[str, Any]:
    """Get supported parameters for an analysis type."""
    try:
        analysis_orchestrator = AnalysisOrchestrator(db)
        return analysis_orchestrator.get_supported_parameters(analysis_type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


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