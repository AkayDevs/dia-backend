from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from app.db.models.user import UserRole
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema with common attributes."""
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=100)
    avatar: Optional[str] = None


class UserCreate(UserBase):
    """Schema for user creation."""
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    role: Optional[UserRole] = None

    @validator("confirm_password")
    def passwords_match(cls, v, values, **kwargs):
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v


class UserUpdate(BaseModel):
    """Schema for user updates."""
    email: Optional[EmailStr] = None
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    avatar: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class UserInDBBase(UserBase):
    """Base schema for user in database."""
    id: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class User(UserInDBBase):
    """Schema for user response."""
    pass


class UserWithStats(User):
    """Schema for user with additional statistics."""
    total_documents: int = 0
    documents_analyzed: int = 0
    last_login: Optional[datetime] = None 