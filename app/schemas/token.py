from typing import Optional, Union
from pydantic import BaseModel, validator
from datetime import datetime


class Token(BaseModel):
    """Schema for access token response."""
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Schema for JWT token payload."""
    sub: str  # user id
    exp: Union[int, datetime]  # can be either timestamp or datetime
    type: str = "access"

    @validator("exp")
    def validate_exp(cls, v):
        if isinstance(v, int):
            return datetime.fromtimestamp(v)
        return v


class BlacklistedToken(BaseModel):
    token: str
    blacklisted_on: datetime
    expires_at: datetime

    class Config:
        from_attributes = True