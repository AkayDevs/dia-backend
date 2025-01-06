from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class Token(BaseModel):
    """Schema for access token response."""
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Schema for JWT token payload."""
    sub: str  # user id
    exp: datetime
    type: str = "access"  # token type (access, refresh, etc.) 