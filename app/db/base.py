"""
This module imports all models to ensure they are registered with SQLAlchemy.
This is required for Alembic to detect models and generate migrations.
"""

from app.db.base_class import Base  # noqa

# Import all models here
from app.db.models.token import BlacklistedToken  # noqa
from app.db.models.user import User  # noqa
from app.db.models.document import Document  # noqa
from app.db.models.analysis import (  # noqa
    AnalysisType,
    AnalysisStep,
    Algorithm,
    Analysis,
    AnalysisStepResult
)

# All models should be imported above this line
