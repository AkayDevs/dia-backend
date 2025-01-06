from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator, ConfigDict
from datetime import datetime
import enum
from app.core.config import settings


class UserRole(str, enum.Enum):
    """Enum for user roles."""
    USER = "user"
    ADMIN = "admin"


class UserBase(BaseModel):
    """Base user schema with common attributes."""
    email: EmailStr = Field(..., description="User's email address")
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="User's full name"
    )
    avatar: Optional[str] = Field(None, description="URL to user's avatar")
    role: Optional[UserRole] = Field(
        default=UserRole.USER,
        description="User's role (defaults to regular user)"
    )


class UserCreate(UserBase):
    """Schema for user creation with password validation."""
    password: str = Field(
        ...,
        min_length=settings.SECURITY_PASSWORD_LENGTH_MIN,
        max_length=settings.SECURITY_PASSWORD_LENGTH_MAX,
        description="User's password"
    )
    confirm_password: str = Field(
        ...,
        min_length=settings.SECURITY_PASSWORD_LENGTH_MIN,
        max_length=settings.SECURITY_PASSWORD_LENGTH_MAX,
        description="Password confirmation"
    )

    @validator("password")
    def password_strength(cls, v):
        """Validate password strength."""
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one number")
        if not any(char in "!@#$%^&*(),.?\":{}|<>" for char in v):
            raise ValueError("Password must contain at least one special character")
        return v

    @validator("confirm_password")
    def passwords_match(cls, v, values, **kwargs):
        """Ensure passwords match."""
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v


class UserUpdate(BaseModel):
    """Schema for user updates with optional fields."""
    email: Optional[EmailStr] = Field(None, description="User's email address")
    name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=100,
        description="User's full name"
    )
    avatar: Optional[str] = Field(None, description="URL to user's avatar")
    password: Optional[str] = Field(
        None,
        min_length=settings.SECURITY_PASSWORD_LENGTH_MIN,
        max_length=settings.SECURITY_PASSWORD_LENGTH_MAX,
        description="New password"
    )
    is_active: Optional[bool] = Field(None, description="Account status")
    is_verified: Optional[bool] = Field(None, description="Email verification status")

    @validator("password")
    def validate_password(cls, v):
        """Validate password if provided."""
        if v is not None:
            if not any(char.isupper() for char in v):
                raise ValueError("Password must contain at least one uppercase letter")
            if not any(char.islower() for char in v):
                raise ValueError("Password must contain at least one lowercase letter")
            if not any(char.isdigit() for char in v):
                raise ValueError("Password must contain at least one number")
            if not any(char in "!@#$%^&*(),.?\":{}|<>" for char in v):
                raise ValueError("Password must contain at least one special character")
        return v


class UserInDBBase(UserBase):
    """Base schema for user in database with all fields."""
    id: str = Field(..., description="User's unique identifier")
    role: UserRole = Field(..., description="User's role")
    is_active: bool = Field(..., description="Whether the account is active")
    is_verified: bool = Field(..., description="Whether the email is verified")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    verification_token: Optional[str] = Field(None, description="Email verification token")
    password_reset_token: Optional[str] = Field(None, description="Password reset token")
    password_reset_expires: Optional[datetime] = Field(
        None,
        description="Password reset token expiration"
    )

    model_config = ConfigDict(from_attributes=True)


class User(UserInDBBase):
    """Schema for user response (excludes sensitive fields)."""
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "created_at": "2024-01-06T12:00:00Z",
                "updated_at": "2024-01-06T12:00:00Z"
            }
        }
    )

class UserWithStats(User):
    """Schema for user with additional statistics."""
    total_documents: int = Field(0, description="Total number of uploaded documents")
    documents_analyzed: int = Field(0, description="Number of analyzed documents")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    storage_used: int = Field(0, description="Total storage used in bytes")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "created_at": "2024-01-06T12:00:00Z",
                "updated_at": "2024-01-06T12:00:00Z",
                "total_documents": 10,
                "documents_analyzed": 8,
                "last_login": "2024-01-06T14:00:00Z",
                "storage_used": 1048576  # 1MB
            } 
        }
    )
