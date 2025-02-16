import logging
import importlib
import pkgutil
from sqlalchemy.orm import Session
from app.analysis.registry.registry import AnalysisRegistry
from app.crud import crud_analysis
from app.analysis.schemas.types import AnalysisTypeCreate, AnalysisTypeUpdate
from app.analysis.schemas.steps import AnalysisStepCreate, AnalysisStepUpdate
from app.analysis.schemas.algorithms import AlgorithmCreate

logger = logging.getLogger(__name__)

def discover_analysis_types() -> None:
    """Discover and register all analysis types"""
    try:
        # Import all analysis types
        import app.analysis.types as analysis_types
        
        for _, name, _ in pkgutil.iter_modules(analysis_types.__path__):
            try:
                # Import the analysis module
                importlib.import_module(f"app.analysis.types.{name}.analysis")
                logger.info(f"Successfully loaded analysis type: {name}")
            except Exception as e:
                logger.error(f"Error loading analysis type {name}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error discovering analysis types: {str(e)}")
        raise

def init_analysis_db(db: Session) -> None:
    """Initialize analysis database using registry"""
    try:
        # Discover all analysis types
        discover_analysis_types()
        
        # Initialize each registered analysis type
        for analysis_type in AnalysisRegistry.list_analysis_types():
            # Create or update analysis type
            db_analysis_type = crud_analysis.analysis_type.get_by_code_and_version(
                db, 
                code=analysis_type.identifier.code,
                version=analysis_type.identifier.version
            )
            
            if not db_analysis_type:
                logger.info(
                    f"Creating analysis type: {analysis_type.identifier.name} "
                    f"v{analysis_type.identifier.version}"
                )
                db_analysis_type = crud_analysis.analysis_type.create(
                    db,
                    obj_in=AnalysisTypeCreate(
                        code=analysis_type.identifier.code,
                        name=analysis_type.identifier.name,
                        version=analysis_type.identifier.version,
                        description=analysis_type.description,
                        supported_document_types=analysis_type.supported_document_types,
                        implementation_path=analysis_type.implementation_path,
                        is_active=True
                    )
                )
            else:
                # Update existing type if needed
                logger.info(
                    f"Updating analysis type: {analysis_type.identifier.name} "
                    f"v{analysis_type.identifier.version}"
                )
                crud_analysis.analysis_type.update(
                    db,
                    db_obj=db_analysis_type,
                    obj_in=AnalysisTypeUpdate(
                        name=analysis_type.identifier.name,
                        description=analysis_type.description,
                        supported_document_types=analysis_type.supported_document_types,
                        implementation_path=analysis_type.implementation_path,
                        is_active=True
                    )
                )
            
            # Create or update steps
            for step in analysis_type.steps:
                db_step = crud_analysis.analysis_step.get_by_code_and_version(
                    db,
                    code=step.identifier.code,
                    version=step.identifier.version,
                    analysis_type_id=str(db_analysis_type.id)
                )
                
                if not db_step:
                    logger.info(
                        f"Creating step: {step.identifier.name} "
                        f"v{step.identifier.version}"
                    )
                    db_step = crud_analysis.analysis_step.create(
                        db,
                        obj_in=AnalysisStepCreate(
                            code=step.identifier.code,
                            name=step.identifier.name,
                            version=step.identifier.version,
                            description=step.description,
                            order=step.order,
                            analysis_type_id=str(db_analysis_type.id),
                            base_parameters=step.base_parameters,
                            result_schema=step.result_schema,
                            implementation_path=step.implementation_path,
                            is_active=True
                        )
                    )
                else:
                    # Update existing step if needed
                    logger.info(
                        f"Updating step: {step.identifier.name} "
                        f"v{step.identifier.version}"
                    )
                    crud_analysis.analysis_step.update(
                        db,
                        db_obj=db_step,
                        obj_in=AnalysisStepUpdate(
                            name=step.identifier.name,
                            description=step.description,
                            order=step.order,
                            base_parameters=step.base_parameters,
                            result_schema=step.result_schema,
                            implementation_path=step.implementation_path,
                            is_active=True
                        )
                    )
                
                # Create or update algorithms
                for algo in step.algorithms:
                    db_algo = crud_analysis.algorithm.get_by_code_and_version(
                        db,
                        code=algo.identifier.code,
                        version=algo.identifier.version,
                        step_id=str(db_step.id)
                    )
                    
                    if not db_algo:
                        logger.info(
                            f"Creating algorithm: {algo.identifier.name} "
                            f"v{algo.identifier.version}"
                        )
                        crud_analysis.algorithm.create(
                            db,
                            obj_in=AlgorithmCreate(
                                code=algo.identifier.code,
                                name=algo.identifier.name,
                                version=algo.identifier.version,
                                description=algo.description,
                                step_id=str(db_step.id),
                                supported_document_types=algo.supported_document_types,
                                parameters=algo.parameters,
                                implementation_path=algo.implementation_path,
                                is_active=True
                            )
                        )
                    else:
                        # Update existing algorithm if needed
                        logger.info(
                            f"Updating algorithm: {algo.identifier.name} "
                            f"v{algo.identifier.version}"
                        )
                        crud_analysis.algorithm.update(
                            db,
                            db_obj=db_algo,
                            obj_in={
                                "name": algo.identifier.name,
                                "description": algo.description,
                                "supported_document_types": algo.supported_document_types,
                                "parameters": algo.parameters,
                                "implementation_path": algo.implementation_path,
                                "is_active": True
                            }
                        )
        
        # Deactivate any components not in registry
        deactivate_unused_components(db)
        
        logger.info("Analysis database initialization completed successfully")
    
    except Exception as e:
        logger.error(f"Error initializing analysis database: {str(e)}")
        raise

def deactivate_unused_components(db: Session) -> None:
    """Deactivate any components that are not in the registry"""
    try:
        # Get all registered components
        registered_types = {
            (t.identifier.code, t.identifier.version): True 
            for t in AnalysisRegistry.list_analysis_types()
        }
        registered_steps = {
            (s.identifier.code, s.identifier.version): True 
            for t in AnalysisRegistry.list_analysis_types() 
            for s in t.steps
        }
        registered_algorithms = {
            (a.identifier.code, a.identifier.version): True 
            for t in AnalysisRegistry.list_analysis_types() 
            for s in t.steps 
            for a in s.algorithms
        }
        
        # Deactivate analysis types not in registry
        for db_type in crud_analysis.analysis_type.get_active_types(db):
            if (db_type.code, db_type.version) not in registered_types:
                logger.info(
                    f"Deactivating analysis type: {db_type.name} "
                    f"v{db_type.version}"
                )
                crud_analysis.analysis_type.update(
                    db,
                    db_obj=db_type,
                    obj_in={"is_active": False}
                )
        
        # Deactivate steps not in registry
        for db_type in crud_analysis.analysis_type.get_active_types(db):
            for db_step in crud_analysis.analysis_step.get_by_analysis_type(db, str(db_type.id)):
                if (db_step.code, db_step.version) not in registered_steps:
                    logger.info(
                        f"Deactivating step: {db_step.name} "
                        f"v{db_step.version}"
                    )
                    crud_analysis.analysis_step.update(
                        db,
                        db_obj=db_step,
                        obj_in={"is_active": False}
                    )
        
        # Deactivate algorithms not in registry
        for db_type in crud_analysis.analysis_type.get_active_types(db):
            for db_step in crud_analysis.analysis_step.get_by_analysis_type(db, str(db_type.id)):
                for db_algo in crud_analysis.algorithm.get_by_step(db, str(db_step.id)):
                    if (db_algo.code, db_algo.version) not in registered_algorithms:
                        logger.info(
                            f"Deactivating algorithm: {db_algo.name} "
                            f"v{db_algo.version}"
                        )
                        crud_analysis.algorithm.update(
                            db,
                            db_obj=db_algo,
                            obj_in={"is_active": False}
                        )
    
    except Exception as e:
        logger.error(f"Error deactivating unused components: {str(e)}")
        raise 