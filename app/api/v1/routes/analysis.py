from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from app.db import deps
from app.crud import crud_analysis, crud_document
from app.schemas.analysis import (
    AnalysisType,
    AnalysisStep,
    Algorithm,
    Analysis,
    AnalysisStepResult,
    AnalysisRequest,
    StepExecutionRequest
)
from app.db.models.user import User
from app.core.analysis import AnalysisOrchestrator

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/types", response_model=List[AnalysisType])
async def list_analysis_types(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[AnalysisType]:
    """
    List all available analysis types with their steps and algorithms.
    """
    return crud_analysis.analysis_type.get_multi(db)

@router.get("/types/{analysis_type_id}", response_model=AnalysisType)
async def get_analysis_type(
    analysis_type_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisType:
    """
    Get detailed information about a specific analysis type.
    """
    analysis_type = crud_analysis.analysis_type.get_with_steps(db, analysis_type_id)
    if not analysis_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis type not found"
        )
    return analysis_type

@router.get("/steps/{step_id}/algorithms", response_model=List[Algorithm])
async def list_step_algorithms(
    step_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Algorithm]:
    """
    List all available algorithms for a specific analysis step.
    """
    return crud_analysis.algorithm.get_by_step(db, step_id)

@router.post("/documents/{document_id}/analyze", response_model=Analysis)
async def start_analysis(
    document_id: str,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Analysis:
    """
    Start a new analysis for a document.
    """
    # Verify document exists and user has access
    document = crud_document.document.get(db, id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Verify analysis type exists
    analysis_type = crud_analysis.analysis_type.get(db, id=str(analysis_request.analysis_type_id))
    if not analysis_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis type not found"
        )

    # Verify document type is supported
    if document.type not in analysis_type.supported_document_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document type {document.type} is not supported by this analysis type"
        )

    try:
        # Create analysis with steps
        analysis_obj = crud_analysis.analysis.create_with_steps(
            db=db,
            obj_in={
                "document_id": document_id,
                "analysis_type_id": str(analysis_request.analysis_type_id),
                "mode": analysis_request.mode,
                "status": "pending"
            },
            algorithm_configs=analysis_request.algorithm_configs
        )

        # Start analysis in background for automatic mode
        if analysis_request.mode == "automatic":
            orchestrator = AnalysisOrchestrator()
            background_tasks.add_task(
                orchestrator.run_analysis,
                db,
                analysis_obj.id
            )

        return analysis_obj

    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error starting analysis"
        )

@router.get("/documents/{document_id}/analyses", response_model=List[Analysis])
async def list_document_analyses(
    document_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Analysis]:
    """
    List all analyses for a specific document.
    """
    # Verify document exists and user has access
    document = crud_document.document.get(db, id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return crud_analysis.analysis.get_by_document(db, document_id)

@router.get("/analyses/{analysis_id}", response_model=Analysis)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Analysis:
    """
    Get detailed information about a specific analysis.
    """
    analysis_obj = crud_analysis.analysis.get(db, id=analysis_id)
    if not analysis_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    # Verify user has access to the document
    document = crud_document.document.get(db, id=analysis_obj.document_id)
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return analysis_obj

@router.post("/analyses/{analysis_id}/steps/{step_id}/execute", response_model=AnalysisStepResult)
async def execute_step(
    analysis_id: str,
    step_id: str,
    execution_request: StepExecutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisStepResult:
    """
    Execute a specific step in step-by-step mode.
    """
    # Verify analysis exists and is in step-by-step mode
    analysis_obj = crud_analysis.analysis.get(db, id=analysis_id)
    if not analysis_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    if analysis_obj.mode != "step_by_step":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This analysis is not in step-by-step mode"
        )

    # Verify user has access to the document
    document = crud_document.document.get(db, id=analysis_obj.document_id)
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Get step result
    step_result = next(
        (r for r in analysis_obj.step_results if str(r.step_id) == step_id),
        None
    )
    if not step_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Step not found in this analysis"
        )

    # Update step configuration
    step_result.algorithm_id = execution_request.algorithm_id
    step_result.parameters = execution_request.parameters
    step_result.status = "pending"
    db.add(step_result)
    db.commit()
    db.refresh(step_result)

    # Execute step in background
    orchestrator = AnalysisOrchestrator()
    background_tasks.add_task(
        orchestrator.run_step,
        db,
        step_result.id
    )

    return step_result

@router.put("/analyses/{analysis_id}/steps/{step_id}/corrections", response_model=AnalysisStepResult)
async def update_step_corrections(
    analysis_id: str,
    step_id: str,
    corrections: Dict[str, Any],
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisStepResult:
    """
    Update user corrections for a step result.
    """
    # Verify analysis exists
    analysis_obj = crud_analysis.analysis.get(db, id=analysis_id)
    if not analysis_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    # Verify user has access to the document
    document = crud_document.document.get(db, id=analysis_obj.document_id)
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Get step result
    step_result = next(
        (r for r in analysis_obj.step_results if str(r.step_id) == step_id),
        None
    )
    if not step_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Step not found in this analysis"
        )

    # Update corrections
    return crud_analysis.analysis_step_result.update_user_corrections(
        db=db,
        db_obj=step_result,
        corrections=corrections
    ) 