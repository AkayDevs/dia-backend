from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AnalysisIdentifier(BaseModel):
    """Unique identifier for an analysis component"""
    name: str
    code: str
    version: str
    created_at: datetime = datetime.utcnow()
    updated_at: Optional[datetime] = None