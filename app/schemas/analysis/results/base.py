from typing import Dict, Any, Type, ClassVar
from pydantic import BaseModel, Field, create_model

class BaseResultSchema(BaseModel):
    """Base class for all step result schemas"""
    
    # Class variable to store schema metadata
    schema_info: ClassVar[Dict[str, Any]] = {
        "name": "",
        "description": "",
        "version": "1.0.0"
    }
    
    @classmethod
    def get_schema_info(cls) -> Dict[str, Any]:
        """Get schema metadata"""
        return cls.schema_info
    
    @classmethod
    def validate_result(cls, result: Dict[str, Any]) -> "BaseResultSchema":
        """Validate result against schema"""
        return cls(**result)
    
    @classmethod
    def create_dynamic_schema(
        cls,
        schema_name: str,
        field_definitions: Dict[str, tuple]
    ) -> Type["BaseResultSchema"]:
        """
        Create a dynamic result schema.
        
        Args:
            schema_name: Name of the schema
            field_definitions: Dictionary of field names and their types/defaults
                             Format: {field_name: (field_type, field_default)}
        """
        return create_model(
            schema_name,
            __base__=cls,
            **field_definitions
        ) 