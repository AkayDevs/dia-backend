from typing import Type, Dict, Any
import importlib
import logging
from app.schemas.analysis.results.base import BaseResultSchema

logger = logging.getLogger(__name__)

class ResultSchemaLoader:
    """Utility class to load and manage result schemas"""
    
    _schemas: Dict[str, Type[BaseResultSchema]] = {}
    
    @classmethod
    def load_schema(cls, schema_path: str) -> Type[BaseResultSchema]:
        """
        Load a result schema class by its Python path.
        
        Args:
            schema_path: Full Python path to the schema class
                        (e.g., 'app.schemas.analysis.results.text_extraction.TextExtractionResult')
        
        Returns:
            Schema class
            
        Raises:
            ImportError: If schema cannot be imported
            AttributeError: If schema class doesn't exist in module
        """
        if schema_path in cls._schemas:
            return cls._schemas[schema_path]
            
        try:
            module_path, class_name = schema_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            schema_class = getattr(module, class_name)
            
            if not issubclass(schema_class, BaseResultSchema):
                raise TypeError(f"Schema class {schema_path} must inherit from BaseResultSchema")
            
            cls._schemas[schema_path] = schema_class
            return schema_class
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading schema {schema_path}: {str(e)}")
            raise
    
    @classmethod
    def validate_result(cls, schema_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a result against its schema.
        
        Args:
            schema_path: Full Python path to the schema class
            result: Result data to validate
            
        Returns:
            Validated result data
            
        Raises:
            ValidationError: If result doesn't match schema
        """
        schema_class = cls.load_schema(schema_path)
        validated = schema_class.validate_result(result)
        return validated.dict()
    
    @classmethod
    def get_schema_info(cls, schema_path: str) -> Dict[str, Any]:
        """Get schema metadata"""
        schema_class = cls.load_schema(schema_path)
        return schema_class.get_schema_info()
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached schemas"""
        cls._schemas.clear() 