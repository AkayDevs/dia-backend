from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from app.db import deps
from app.crud import crud_analysis_config, crud_document, crud_analysis_execution
from app.schemas.analysis.configs.definitions import (
    AnalysisDefinitionInfo,
    AnalysisDefinitionWithSteps,
    AnalysisDefinitionWithStepsAndAlgorithms
)
from app.schemas.analysis.configs.steps import StepDefinitionWithAlgorithms
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionInfo
from app.schemas.analysis.executions import (
    AnalysisRunCreate,
    AnalysisRunInfo,
    AnalysisRunWithResults,
    StepExecutionResultInfo,
    StepExecutionResultUpdate,
    AnalysisRunConfig
)
from app.db.models.user import User
from app.services.analysis.executions.orchestrator import AnalysisOrchestrator
from app.schemas.document import DocumentType
from app.enums.analysis import AnalysisMode, AnalysisStatus, AnalysisProcessingType
from app.services.analysis.configs.utils import prepare_analysis_config
from app.services.analysis.configs.registry import AnalysisRegistry
router = APIRouter()

@router.get("/definitions", response_model=List[AnalysisDefinitionInfo])
async def list_analysis_definitions(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[AnalysisDefinitionInfo]:
    """
    List all available analysis definitions.
    """
    return crud_analysis_config.analysis_definition.get_active_definitions(db)

@router.get("/definitions/{definition_id}", response_model=AnalysisDefinitionWithStepsAndAlgorithms)
async def get_analysis_definition(
    definition_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisDefinitionWithStepsAndAlgorithms:
    """
    Get detailed information about a specific analysis definition.
    """
    definition = crud_analysis_config.analysis_definition.get(db, id=definition_id)
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis definition not found"
        )
    return definition

@router.get("/steps/{step_id}/algorithms", response_model=List[AlgorithmDefinitionInfo])
async def list_step_algorithms(
    step_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[AlgorithmDefinitionInfo]:
    """
    List all available algorithms for a specific analysis step.
    """
    return crud_analysis_config.algorithm_definition.get_by_step(db, step_id)

@router.post("/documents/{document_id}/analyze", response_model=AnalysisRunInfo)
async def start_analysis(
    document_id: str,
    analysis_code: str,
    mode: AnalysisMode = AnalysisMode.AUTOMATIC,
    config: Optional[AnalysisRunConfig] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisRunInfo:
    """
    Start a new analysis for a document.
    
    Args:
        document_id: ID of the document to analyze
        analysis_code: Code of the analysis definition to use
        mode: Analysis execution mode (automatic or step_by_step)
        config: Optional configuration for the analysis run including:
            - steps: Configuration for each step (algorithms, parameters, etc.)
            - notifications: WebSocket notification settings
            - metadata: Additional metadata for the run
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

    # Verify analysis definition exists and is active
    definition = AnalysisRegistry.get_analysis_definition(analysis_code)
    if not definition or not definition.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis definition not found or inactive"
        )

    # Verify document type is supported
    if document.type not in definition.supported_document_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document type {document.type} is not supported by this analysis"
        )

    try:
        # Prepare complete configuration with defaults
        complete_config = prepare_analysis_config(
            user=current_user,
            document=document,
            analysis_code=definition.code,
            analysis_processing_type=AnalysisProcessingType.SINGLE,
            user_config=config
        )
        
        # Create analysis run with complete configuration
        analysis_create = AnalysisRunCreate(
            document_id=document_id,
            analysis_code=definition.code,
            mode=mode,
            config=complete_config
        )
        
        analysis_run = crud_analysis_execution.analysis_run.create_with_steps(
            db=db,
            obj_in=analysis_create
        )

        # Start analysis in background for automatic mode
        if mode == AnalysisMode.AUTOMATIC and background_tasks:
            orchestrator = AnalysisOrchestrator()
            background_tasks.add_task(
                orchestrator.run_analysis,
                db,
                analysis_run.id
            )

        return analysis_run

    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/documents/{document_id}/analyses", response_model=List[AnalysisRunWithResults])
async def list_document_analyses(
    document_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[AnalysisRunWithResults]:
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

    return crud_analysis_execution.analysis_run.get_by_document(db, document_id)

@router.get("/analyses/{analysis_id}", response_model=AnalysisRunWithResults)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisRunWithResults:
    """
    Get detailed information about a specific analysis.
    """
    analysis_run = crud_analysis_execution.analysis_run.get(db, id=analysis_id)
    if not analysis_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    # Verify user has access to the document
    document = crud_document.document.get(db, id=analysis_run.document_id)
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return analysis_run

@router.post("/analyses/{analysis_id}/steps/{step_id}/execute", response_model=StepExecutionResultInfo)
async def execute_step(
    analysis_id: str,
    step_id: str,
    algorithm_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> StepExecutionResultInfo:
    """
    Execute a specific step in step-by-step mode.
    
    Args:
        analysis_id: ID of the analysis run
        step_id: ID of the step to execute
        algorithm_id: ID of the algorithm to use
        parameters: Optional algorithm parameters
    """
    # Verify analysis exists and is in step-by-step mode
    analysis_run = crud_analysis_config.analysis_run.get(db, id=analysis_id)
    if not analysis_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    if analysis_run.mode != AnalysisMode.STEP_BY_STEP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This analysis is not in step-by-step mode"
        )

    # Verify user has access to the document
    document = crud_document.document.get(db, id=analysis_run.document_id)
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Get step result
    step_result = next(
        (r for r in analysis_run.step_results if str(r.step_definition_id) == step_id),
        None
    )
    if not step_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Step not found in this analysis"
        )

    # Verify algorithm exists and is active
    algorithm = crud_analysis_config.algorithm_definition.get(db, id=algorithm_id)
    if not algorithm or not algorithm.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Algorithm not found or inactive"
        )

    try:
        # Update step configuration
        step_result_update = StepExecutionResultUpdate(
            algorithm_definition_id=algorithm_id,
            parameters=parameters or {},
            status=AnalysisStatus.PENDING
        )
        step_result = crud_analysis_config.step_execution_result.update(
            db,
            db_obj=step_result,
            obj_in=step_result_update
        )

        # Execute step in background
        if background_tasks:
            orchestrator = AnalysisOrchestrator()
            background_tasks.add_task(
                orchestrator.run_step,
                db,
                step_result.id
            )

        return step_result

    except Exception as e:
        logger.error(f"Error executing step: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/analyses/{analysis_id}/steps/{step_id}/corrections", response_model=StepExecutionResultInfo)
async def update_step_corrections(
    analysis_id: str,
    step_id: str,
    corrections: Dict[str, Any],
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> StepExecutionResultInfo:
    """
    Update user corrections for a step result.
    """
    # Verify analysis exists
    analysis_run = crud_analysis_config.analysis_run.get(db, id=analysis_id)
    if not analysis_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    # Verify user has access to the document
    document = crud_document.document.get(db, id=analysis_run.document_id)
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Get step result
    step_result = next(
        (r for r in analysis_run.step_results if str(r.step_definition_id) == step_id),
        None
    )
    if not step_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Step not found in this analysis"
        )

    # Update corrections
    return crud_analysis_config.step_execution_result.update_user_corrections(
        db=db,
        db_obj=step_result,
        corrections=corrections
    )

@router.get("/user/analyses", response_model=List[AnalysisRunWithResults])
async def list_user_analyses(
    status: Optional[AnalysisStatus] = Query(None, description="Filter by analysis status"),
    analysis_code: Optional[str] = Query(None, description="Filter by analysis code"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date (inclusive)"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date (inclusive)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[AnalysisRunWithResults]:
    """
    List all analyses for the current user with filtering options.
    """
    try:
        filters = {
            "user_id": str(current_user.id),
            "status": status,
            "analysis_code": analysis_code,
            "document_type": document_type,
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Remove None values from filters
        filters = {k: v for k, v in filters.items() if v is not None}
        
        # Get analyses using the analysis execution CRUD
        analyses = crud_analysis_execution.analysis_run.get_multi_by_filters(
            db=db,
            filters=filters,
            skip=skip,
            limit=limit
        )
        
        # Convert to AnalysisRunWithResults using from_orm
        # This will automatically handle nested relationships
        return [AnalysisRunWithResults.from_orm(analysis) for analysis in analyses]
    except Exception as e:
        logger.error(f"Error listing user analyses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing analyses: {str(e)}"
        ) 