from typing import Dict, Any, Optional
import logging
from datetime import datetime

from app.schemas.analysis.executions.analysis_run import (
    AnalysisRunConfig,
    StepConfig,
    NotificationConfig,
    AlgorithmSelection
)
from app.db.models.user import User
from app.db.models.document import Document
from app.enums.analysis import AnalysisProcessingType
from app.services.analysis.configs.registry import AnalysisRegistry

logger = logging.getLogger(__name__)

def prepare_analysis_config(
    *,
    user: User,
    document: Document,
    analysis_code: str,
    user_config: Optional[AnalysisRunConfig] = None,
    analysis_processing_type: AnalysisProcessingType = AnalysisProcessingType.SINGLE
) -> AnalysisRunConfig:
    """
    Prepare a complete analysis configuration based on user input and defaults.
    
    Args:
        db: Database session
        user: Current user
        document: Document to be analyzed
        analysis_code: Code of the analysis definition
        mode: Analysis execution mode
        user_config: Optional user-provided configuration
        
    Returns:
        Complete AnalysisRunConfig with all necessary defaults
    """
    # Start with user config or create new
    config = user_config or AnalysisRunConfig()
    
    # Prepare steps configuration
    config.steps = prepare_steps_config(
        analysis_code=analysis_code,
        user_steps_config=config.steps
    )
    
    # Prepare notification configuration
    config.notifications = prepare_notification_config(
        user=user,
        document=document,
        analysis_processing_type=analysis_processing_type,
        user_notification_config=config.notifications
    )
    
    # Prepare metadata
    config.metadata = prepare_metadata(
        user=user,
        document=document,
        analysis_processing_type=analysis_processing_type,
        analysis_code=analysis_code,
        user_metadata=config.metadata
    )
    
    return config

def prepare_steps_config(
    *,
    analysis_code: str,
    user_steps_config: Optional[Dict[str, StepConfig]] = None
) -> Dict[str, StepConfig]:
    """
    Prepare complete step configurations with defaults.
    
    Args:
        analysis_code: Code of the analysis definition
        user_steps_config: Optional user-provided step configurations
        
    Returns:
        Complete step configurations with defaults
    """
    steps_config = {}
    user_steps_config = user_steps_config or {}
    
    # Get analysis definition from registry
    analysis_def = AnalysisRegistry.get_analysis_definition(analysis_code)
    if not analysis_def:
        raise ValueError(f"Analysis definition {analysis_code} not found in registry")
    
    # Get all active steps from registry
    steps = AnalysisRegistry.list_steps(analysis_code)
    
    for step in steps:
        step_code = f"{analysis_code}.{step.code}"
        
        try:
            # Initialize step implementation to get defaults
            step_impl_class = import_class(step.implementation_path)
            step_impl = step_impl_class()
            
            # Get default algorithm configuration
            default_algo = step_impl.get_default_algorithm()
            default_algorithm_selection = None
            default_parameters = {}
            
            if default_algo:
                # Get default algorithm from registry
                default_algorithm = next(
                    (algo for algo in AnalysisRegistry.list_algorithms(step_code)
                     if algo.code == default_algo.algorithm_code
                     and algo.version == default_algo.algorithm_version),
                    None
                )
                
                if default_algorithm:
                    # Get default parameters
                    algo_impl_class = import_class(default_algorithm.implementation_path)
                    algo_impl = algo_impl_class()
                    default_parameters = algo_impl.get_default_parameters()
                    
                    
                    default_algorithm_selection = AlgorithmSelection(
                        algorithm_code=default_algorithm.code,
                        algorithm_version=default_algorithm.version,
                        parameters=default_parameters
                    )
            else:
                logger.warning(f"No default algorithm found for step {step_code}")
            
            # If user provided config for this step
            if step_code in user_steps_config:
                user_step_config = user_steps_config[step_code]
                
                # Handle algorithm configuration
                algorithm_selection = None
                if user_step_config.algorithm:
                    # Get algorithm from registry
                    algorithm = next(
                        (algo for algo in AnalysisRegistry.list_algorithms(step_code)
                         if algo.code == user_step_config.algorithm.code
                         and algo.version == user_step_config.algorithm.version),
                        None
                    )
                    
                    if algorithm:
                        # Get algorithm implementation for default parameters
                        algo_impl_class = import_class(algorithm.implementation_path)
                        algo_impl = algo_impl_class()
                        default_algo_params = algo_impl.get_default_parameters()
                        
                        # Merge user parameters with defaults
                        merged_parameters = {
                            **default_algo_params,
                            **(user_step_config.algorithm.parameters or {})
                        }
                        
                        algorithm_selection = AlgorithmSelection(
                            algorithm_code=algorithm.code,
                            algorithm_version=algorithm.version,
                            parameters=merged_parameters
                        )
                    else:
                        logger.warning(f"Algorithm {user_step_config.algorithm.code} not found in registry")
                        
                
                # Create step config with user values or defaults
                steps_config[step_code] = StepConfig(
                    algorithm=algorithm_selection or default_algorithm_selection,
                    enabled=user_step_config.enabled if user_step_config.enabled is not None else True,
                    timeout=user_step_config.timeout,  # None is a valid value for timeout
                    retry=user_step_config.retry if user_step_config.retry is not None else 3
                )
            else:
                # Create step config with all defaults
                steps_config[step_code] = StepConfig(
                    algorithm=default_algorithm_selection,
                    enabled=True,
                    timeout=None,
                    retry=3
                )

        except Exception as e:
            logger.error(f"Error preparing config for step {step_code}: {str(e)}")
            logger.error(f"Using default configuration for step {step_code}")
            # Use basic defaults if error occurs
            steps_config[step_code] = StepConfig(
                algorithm=None,
                enabled=True,
                timeout=None,
                retry=3
            )
            continue
    
    return steps_config

def prepare_notification_config(
    *,
    user: User,
    document: Document,
    analysis_processing_type: AnalysisProcessingType,
    user_notification_config: Optional[NotificationConfig] = None
) -> NotificationConfig:
    """
    Prepare notification configuration based on analysis mode.
    
    Args:
        user: Current user
        document: Document being analyzed
        mode: Analysis execution mode
        user_notification_config: Optional user-provided notification config
        
    Returns:
        Complete notification configuration
    """

    if user_notification_config:
        return user_notification_config
    
    # For single document analysis, use websocket notifications
    if analysis_processing_type == AnalysisProcessingType.SINGLE:
        return NotificationConfig(
            notify_on_completion=True,
            notify_on_failure=True,
            websocket_channel=f"user_{user.id}_analysis_{document.id}"
        )
    else:
        return NotificationConfig(
            notify_on_completion=True,
            notify_on_failure=True,
            websocket_channel=None
        )

def prepare_metadata(
    *,
    user: User,
    document: Document,
    analysis_processing_type: AnalysisProcessingType,
    analysis_code: str,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare metadata for the analysis run.
    
    Args:
        user: Current user
        document: Document being analyzed
        mode: Analysis execution mode
        analysis_code: Code of the analysis definition
        user_metadata: Optional user-provided metadata
        
    Returns:
        Complete metadata dictionary
    """
    metadata = user_metadata or {}
    
    # Add standard metadata
    metadata.update({
        "source": "api",
        "document_type": document.type,
        "created_at": datetime.utcnow().isoformat(),
        "user_id": str(user.id),
        "document_id": str(document.id),
        "analysis_code": analysis_code,
        "analysis_processing_type": analysis_processing_type.value
    })
    
    return metadata

def import_class(path: str) -> Any:
    """
    Dynamically import a class from a string path.
    
    Args:
        path: Full path to the class (e.g., 'app.services.analysis.implementations.my_step.MyStep')
    
    Returns:
        The class object
    """
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        logger.error(f"Error importing class {path}: {str(e)}")
        raise ImportError(f"Could not import class {path}") 