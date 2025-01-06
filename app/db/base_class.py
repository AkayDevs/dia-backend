from typing import Any
from sqlalchemy.orm import DeclarativeBase, declared_attr

class Base(DeclarativeBase):
    """
    Custom base class for all SQLAlchemy models.
    Provides common functionality and naming conventions.
    """
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Generate __tablename__ automatically from class name.
        Converts CamelCase to snake_case (e.g., UserProfile becomes user_profile).
        """
        return cls.__name__.lower()
    
    # Implement any common model methods here
    def dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        } 