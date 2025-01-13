from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
import logging
from app.db.models.user import User
from app.db import deps
from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResult,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BatchAnalysisError,
    StepApprovalRequest
)
from app.enums.analysis import AnalysisMode, AnalysisStatus, AnalysisType
from app.services.analysis import AnalysisOrchestrator
from app.crud.crud_analysis import analysis_result as crud_analysis
from app.crud.crud_document import document as crud_document

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/{document_id}/analyze", response_model=AnalysisResult)
async def analyze_document(
    document_id: str,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> AnalysisResult:
    """
    Start analysis on a document.
    
    Args:
        document_id: ID of document to analyze
        request: Analysis request details
        background_tasks: FastAPI background tasks
        db: Database session
        current_user: Current user
        
    Returns:
        AnalysisResult: Created analysis task
    """
    try:
        # Verify document exists and user has access
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this document")

        # Create orchestrator
        orchestrator = AnalysisOrchestrator(db)
        
        # Start analysis
        analysis = await orchestrator.start_analysis(
            document_id=document_id,
            analysis_type=request.analysis_type,
            parameters=request.parameters
        )
        
        # Process analysis in background if automatic mode
        if request.mode == AnalysisMode.AUTOMATIC:
            background_tasks.add_task(
                orchestrator.process_analysis,
                analysis_id=analysis.id
            )
        
        return analysis

    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/analysis/{analysis_id}/step", response_model=AnalysisResult)
async def execute_analysis_step(
    document_id: str,
    analysis_id: str,
    step: Optional[str] = None,
    step_parameters: Optional[Dict[str, Any]] = None,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> AnalysisResult:
    """
    Execute a specific step of an analysis.
    
    Args:
        document_id: ID of document
        analysis_id: ID of analysis
        step: Optional specific step to execute
        step_parameters: Optional parameters for the step
        db: Database session
        current_user: Current user
        
    Returns:
        AnalysisResult: Updated analysis task
    """
    try:
        # Verify document exists and user has access
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this document")

        # Verify analysis exists
        analysis = crud_analysis.get(db, id=analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if analysis.document_id != document_id:
            raise HTTPException(status_code=400, detail="Analysis does not belong to document")

        # Create orchestrator
        orchestrator = AnalysisOrchestrator(db)
        
        # Execute step
        await orchestrator.process_analysis(
            analysis_id=analysis_id,
            step=step,
            step_parameters=step_parameters
        )
        
        # Get updated analysis
        updated_analysis = crud_analysis.get(db, id=analysis_id)
        return updated_analysis

    except Exception as e:
        logger.error(f"Error executing analysis step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/analysis/{analysis_id}/approve", response_model=AnalysisResult)
async def approve_analysis_step(
    document_id: str,
    analysis_id: str,
    approval: StepApprovalRequest,
    modifications: Optional[Dict[str, Any]] = None,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> AnalysisResult:
    """
    Approve or reject a step in the analysis process.
    
    Args:
        document_id: ID of document
        analysis_id: ID of analysis
        approval: Step approval details
        modifications: Optional modifications to the step results
        db: Database session
        current_user: Current user
        
    Returns:
        AnalysisResult: Updated analysis task
    """
    try:
        # Verify document exists and user has access
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this document")

        # Verify analysis exists
        analysis = crud_analysis.get(db, id=analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if analysis.document_id != document_id:
            raise HTTPException(status_code=400, detail="Analysis does not belong to document")
        if analysis.status != AnalysisStatus.WAITING_FOR_APPROVAL:
            raise HTTPException(status_code=400, detail="Analysis step not waiting for approval")

        # Create orchestrator
        orchestrator = AnalysisOrchestrator(db)
        
        # Process approval
        updated_analysis = await orchestrator.process_step_approval(
            analysis_id=analysis_id,
            approval=approval,
            modifications=modifications
        )
        
        return updated_analysis

    except Exception as e:
        logger.error(f"Error processing step approval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/analysis", response_model=List[AnalysisResult])
async def list_analyses(
    document_id: str,
    analysis_type: Optional[AnalysisType] = None,
    status: Optional[AnalysisStatus] = None,
    mode: Optional[AnalysisMode] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> List[AnalysisResult]:
    """
    List analyses for a document.
    
    Args:
        document_id: ID of document
        analysis_type: Optional filter by analysis type
        status: Optional filter by status
        mode: Optional filter by mode
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Current user
        
    Returns:
        List[AnalysisResult]: List of analysis tasks
    """
    try:
        # Verify document exists and user has access
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this document")

        # Build filters
        filters = {"document_id": document_id}
        if analysis_type:
            filters["type"] = analysis_type
        if status:
            filters["status"] = status
        if mode:
            filters["mode"] = mode

        # Get analyses
        analyses = crud_analysis.get_multi(
            db,
            skip=skip,
            limit=limit,
            filters=filters
        )
        
        return analyses

    except Exception as e:
        logger.error(f"Error listing analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    document_id: str,
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> AnalysisResult:
    """
    Get details of a specific analysis.
    
    Args:
        document_id: ID of document
        analysis_id: ID of analysis
        db: Database session
        current_user: Current user
        
    Returns:
        AnalysisResult: Analysis task details
    """
    try:
        # Verify document exists and user has access
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this document")

        # Get analysis
        analysis = crud_analysis.get(db, id=analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if analysis.document_id != document_id:
            raise HTTPException(status_code=400, detail="Analysis does not belong to document")
        
        return analysis

    except Exception as e:
        logger.error(f"Error getting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> BatchAnalysisResponse:
    """
    Start batch analysis on multiple documents.
    
    Args:
        request: Batch analysis request details
        background_tasks: FastAPI background tasks
        db: Database session
        current_user_id: ID of current user
        
    Returns:
        BatchAnalysisResponse: Results of batch submission
    """
    try:
        orchestrator = AnalysisOrchestrator(db)
        results = []
        errors = []
        
        for doc in request.documents:
            try:
                # Verify document exists and user has access
                document = crud_document.get(db, id=doc.document_id)
                if not document:
                    raise ValueError("Document not found")
                if document.user_id != current_user.id:
                    raise ValueError("Not authorized to access this document")

                # Start analysis
                analysis = await orchestrator.start_analysis(
                    document_id=doc.document_id,
                    analysis_type=doc.analysis_type,
                    parameters=doc.parameters
                )
                
                # Process in background if automatic mode
                if doc.parameters.get("mode", AnalysisMode.AUTOMATIC) == AnalysisMode.AUTOMATIC:
                    background_tasks.add_task(
                        orchestrator.process_analysis,
                        analysis_id=analysis.id
                    )
                
                results.append(analysis)
                
            except Exception as e:
                errors.append(BatchAnalysisError(
                    document_id=doc.document_id,
                    error=str(e)
                ))
        
        return BatchAnalysisResponse(
            results=results,
            errors=errors,
            total_submitted=len(results),
            total_failed=len(errors)
        )

    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
