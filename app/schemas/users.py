from pydantic import BaseModel, EmailStr
from typing import Optional

class UserProfile(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True

class UserProfileUpdate(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = None

class NotificationSettings(BaseModel):
    id: int
    user_id: int
    email_notifications: bool = True
    analysis_complete: bool = True
    document_shared: bool = True
    security_alerts: bool = True

    class Config:
        from_attributes = True

class NotificationSettingsUpdate(BaseModel):
    email_notifications: Optional[bool] = None
    analysis_complete: Optional[bool] = None
    document_shared: Optional[bool] = None
    security_alerts: Optional[bool] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str 