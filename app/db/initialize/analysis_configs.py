import logging
import importlib
import pkgutil
from sqlalchemy.orm import Session
from app.services.analysis.configs.registry import AnalysisRegistry
from app.crud import crud_analysis_config
from app.schemas.analysis.configs.definitions import (
    AnalysisDefinitionCreate,
    AnalysisDefinitionUpdate
)
from app.schemas.analysis.configs.steps import (
    StepDefinitionCreate,
    StepDefinitionUpdate
)
from app.schemas.analysis.configs.algorithms import (
    AlgorithmDefinitionCreate,
    AlgorithmDefinitionUpdate
)

logger = logging.getLogger(__name__)

def discover_analysis_definitions() -> None:
    """Discover and register all analysis definitions"""
    try:
        # Import all analysis definitions
        import app.services.analysis.configs.definitions as analysis_defs
        
        for _, name, _ in pkgutil.iter_modules(analysis_defs.__path__):
            try:
                # Import the analysis module
                module = importlib.import_module(f"app.services.analysis.configs.definitions.{name}")
                # Call register_components if available
                if hasattr(module, 'register_components'):
                    module.register_components()
                logger.info(f"Successfully loaded analysis definition: {name}")
            except Exception as e:
                logger.error(f"Error loading analysis definition {name}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error discovering analysis definitions: {str(e)}")
        raise

def init_analysis_db(db: Session) -> None:
    """Initialize analysis database using registry"""
    try:
        # Discover all analysis definitions
        discover_analysis_definitions()
        
        # Initialize each registered analysis definition
        for definition in AnalysisRegistry.list_analysis_definitions():
            try:
                # Create or update analysis definition
                db_definition = crud_analysis_config.analysis_definition.get_by_code_and_version(
                    db, 
                    code=definition.code,
                    version=definition.version
                )
                
                if not db_definition:
                    logger.info(f"Creating analysis definition: {definition.name} v{definition.version}")
                    db_definition = crud_analysis_config.analysis_definition.create(
                        db,
                        obj_in=AnalysisDefinitionCreate(
                            code=definition.code,
                            name=definition.name,
                            version=definition.version,
                            description=definition.description,
                            supported_document_types=definition.supported_document_types,
                            implementation_path=definition.implementation_path,
                            is_active=True
                        )
                    )
                else:
                    logger.info(f"Updating analysis definition: {definition.name} v{definition.version}")
                    crud_analysis_config.analysis_definition.update(
                        db,
                        db_obj=db_definition,
                        obj_in=AnalysisDefinitionUpdate(
                            name=definition.name,
                            description=definition.description,
                            supported_document_types=definition.supported_document_types,
                            implementation_path=definition.implementation_path,
                            is_active=True
                        )
                    )
                
                # Create or update steps
                for step in AnalysisRegistry.list_steps(definition.code):
                    try:
                        db_step = crud_analysis_config.step_definition.get_by_code_and_version(
                            db,
                            code=step.code,
                            version=step.version,
                            analysis_definition_id=str(db_definition.id)
                        )
                        
                        if not db_step:
                            logger.info(f"Creating step: {step.name} v{step.version}")
                            # Validate result schema path
                            if not crud_analysis_config.step_definition.validate_result_schema(step.result_schema_path):
                                logger.warning(f"Invalid result schema path: {step.result_schema_path}")
                                continue
                            
                            try:
                                db_step = crud_analysis_config.step_definition.create(
                                    db,
                                    obj_in=StepDefinitionCreate(
                                        code=step.code,
                                        name=step.name,
                                        version=step.version,
                                        description=step.description,
                                        order=step.order,
                                        analysis_definition_id=str(db_definition.id),
                                        base_parameters=step.base_parameters,
                                        result_schema_path=step.result_schema_path,
                                        implementation_path=step.implementation_path,
                                        is_active=True
                                    )
                                )
                            except Exception as e:
                                db.rollback()  # Rollback on any error
                                if "UNIQUE constraint failed" in str(e):
                                    logger.warning(f"Step already exists: {step.name} v{step.version} for analysis {definition.name}")
                                    # Get the existing step and update it
                                    db_step = crud_analysis_config.step_definition.get_by_code_and_version(
                                        db,
                                        code=step.code,
                                        version=step.version,
                                        analysis_definition_id=str(db_definition.id)
                                    )
                                    if db_step:
                                        crud_analysis_config.step_definition.update(
                                            db,
                                            db_obj=db_step,
                                            obj_in=StepDefinitionUpdate(
                                                name=step.name,
                                                description=step.description,
                                                order=step.order,
                                                base_parameters=step.base_parameters,
                                                result_schema_path=step.result_schema_path,
                                                implementation_path=step.implementation_path,
                                                is_active=True
                                            )
                                        )
                                else:
                                    raise
                        
                        # Create or update algorithms
                        step_code = f"{definition.code}.{step.code}"
                        for algo in AnalysisRegistry.list_algorithms(step_code):
                            try:
                                db_algo = crud_analysis_config.algorithm_definition.get_by_code_and_version(
                                    db,
                                    code=algo.code,
                                    version=algo.version,
                                    step_id=str(db_step.id)
                                )
                                
                                if not db_algo:
                                    logger.info(f"Creating algorithm: {algo.name} v{algo.version}")
                                    try:
                                        crud_analysis_config.algorithm_definition.create(
                                            db,
                                            obj_in=AlgorithmDefinitionCreate(
                                                code=algo.code,
                                                name=algo.name,
                                                version=algo.version,
                                                description=algo.description,
                                                step_id=str(db_step.id),
                                                supported_document_types=algo.supported_document_types,
                                                parameters=algo.parameters,
                                                implementation_path=algo.implementation_path,
                                                is_active=True
                                            )
                                        )
                                    except Exception as e:
                                        db.rollback()  # Rollback on any error
                                        if "UNIQUE constraint failed" in str(e):
                                            logger.warning(f"Algorithm already exists: {algo.name} v{algo.version} for step {step.name}")
                                            # Get the existing algorithm and update it
                                            db_algo = crud_analysis_config.algorithm_definition.get_by_code_and_version(
                                                db,
                                                code=algo.code,
                                                version=algo.version,
                                                step_id=str(db_step.id)
                                            )
                                            if db_algo:
                                                crud_analysis_config.algorithm_definition.update(
                                                    db,
                                                    db_obj=db_algo,
                                                    obj_in=AlgorithmDefinitionUpdate(
                                                        name=algo.name,
                                                        description=algo.description,
                                                        supported_document_types=algo.supported_document_types,
                                                        parameters=algo.parameters,
                                                        implementation_path=algo.implementation_path,
                                                        is_active=True
                                                    )
                                                )
                                        else:
                                            raise
                                else:
                                    logger.info(f"Updating algorithm: {algo.name} v{algo.version}")
                                    crud_analysis_config.algorithm_definition.update(
                                        db,
                                        db_obj=db_algo,
                                        obj_in=AlgorithmDefinitionUpdate(
                                            name=algo.name,
                                            description=algo.description,
                                            supported_document_types=algo.supported_document_types,
                                            parameters=algo.parameters,
                                            implementation_path=algo.implementation_path,
                                            is_active=True
                                        )
                                    )
                            except Exception as e:
                                db.rollback()
                                logger.error(f"Error processing algorithm {algo.name}: {str(e)}")
                                raise
                    except Exception as e:
                        db.rollback()
                        logger.error(f"Error processing step {step.name}: {str(e)}")
                        raise
            except Exception as e:
                db.rollback()
                logger.error(f"Error processing analysis definition {definition.name}: {str(e)}")
                raise
        
        # Deactivate any components not in registry
        try:
            deactivate_unused_components(db)
        except Exception as e:
            db.rollback()
            logger.error(f"Error deactivating unused components: {str(e)}")
            raise
        
        logger.info("Analysis database initialization completed successfully")
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error initializing analysis database: {str(e)}")
        raise

def deactivate_unused_components(db: Session) -> None:
    """Deactivate any components that are not in the registry"""
    try:
        # Get all registered components
        registered_definitions = {
            (d.code, d.version): True 
            for d in AnalysisRegistry.list_analysis_definitions()
        }
        
        registered_steps = {
            (s.code, s.version): True 
            for d in AnalysisRegistry.list_analysis_definitions()
            for s in AnalysisRegistry.list_steps(d.code)
        }
        
        registered_algorithms = {
            (a.code, a.version): True 
            for d in AnalysisRegistry.list_analysis_definitions()
            for s in AnalysisRegistry.list_steps(d.code)
            for a in AnalysisRegistry.list_algorithms(f"{d.code}.{s.code}")
        }
        
        # Deactivate analysis definitions not in registry
        for db_def in crud_analysis_config.analysis_definition.get_active_definitions(db):
            if (db_def.code, db_def.version) not in registered_definitions:
                logger.info(f"Deactivating analysis definition: {db_def.name} v{db_def.version}")
                crud_analysis_config.analysis_definition.update(
                    db,
                    db_obj=db_def,
                    obj_in=AnalysisDefinitionUpdate(is_active=False)
                )
        
        # Deactivate steps not in registry
        for db_def in crud_analysis_config.analysis_definition.get_active_definitions(db):
            for db_step in crud_analysis_config.step_definition.get_by_analysis_definition(db, str(db_def.id)):
                if (db_step.code, db_step.version) not in registered_steps:
                    logger.info(f"Deactivating step: {db_step.name} v{db_step.version}")
                    crud_analysis_config.step_definition.update(
                        db,
                        db_obj=db_step,
                        obj_in=StepDefinitionUpdate(is_active=False)
                    )
        
        # Deactivate algorithms not in registry
        for db_def in crud_analysis_config.analysis_definition.get_active_definitions(db):
            for db_step in crud_analysis_config.step_definition.get_by_analysis_definition(db, str(db_def.id)):
                for db_algo in crud_analysis_config.algorithm_definition.get_by_step(db, str(db_step.id)):
                    if (db_algo.code, db_algo.version) not in registered_algorithms:
                        logger.info(f"Deactivating algorithm: {db_algo.name} v{db_algo.version}")
                        crud_analysis_config.algorithm_definition.update(
                            db,
                            db_obj=db_algo,
                            obj_in=AlgorithmDefinitionUpdate(is_active=False)
                        )
    
    except Exception as e:
        logger.error(f"Error deactivating unused components: {str(e)}")
        raise 