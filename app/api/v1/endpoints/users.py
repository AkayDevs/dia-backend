from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.db import models
from app.core import security
from app.schemas import users as user_schemas

router = APIRouter()

@router.get("/me", response_model=user_schemas.UserProfile)
async def get_current_user_profile(
    current_user: models.User = Depends(security.get_current_user)
):
    """Get current user's profile"""
    return current_user

@router.put("/me", response_model=user_schemas.UserProfile)
async def update_user_profile(
    profile: user_schemas.UserProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Update user profile"""
    for key, value in profile.dict(exclude_unset=True).items():
        setattr(current_user, key, value)
    
    await db.commit()
    await db.refresh(current_user)
    return current_user

@router.get("/notifications", response_model=user_schemas.NotificationSettings)
async def get_notification_settings(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Get user's notification settings"""
    query = select(models.NotificationSettings).where(
        models.NotificationSettings.user_id == current_user.id
    )
    result = await db.execute(query)
    settings = result.scalar_one_or_none()
    
    if not settings:
        # Create default settings if none exist
        settings = models.NotificationSettings(
            user_id=current_user.id,
            email_notifications=True,
            analysis_complete=True,
            document_shared=True,
            security_alerts=True
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    
    return settings

@router.put("/notifications", response_model=user_schemas.NotificationSettings)
async def update_notification_settings(
    settings: user_schemas.NotificationSettingsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Update notification settings"""
    query = select(models.NotificationSettings).where(
        models.NotificationSettings.user_id == current_user.id
    )
    result = await db.execute(query)
    notification_settings = result.scalar_one_or_none()
    
    if not notification_settings:
        notification_settings = models.NotificationSettings(user_id=current_user.id)
        db.add(notification_settings)
    
    for key, value in settings.dict(exclude_unset=True).items():
        setattr(notification_settings, key, value)
    
    await db.commit()
    await db.refresh(notification_settings)
    return notification_settings

@router.put("/password")
async def change_password(
    password_change: user_schemas.PasswordChange,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Change user password"""
    if not security.verify_password(password_change.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    current_user.hashed_password = security.get_password_hash(password_change.new_password)
    await db.commit()
    return {"message": "Password updated successfully"}

@router.delete("/me")
async def delete_account(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Delete user account"""
    await db.delete(current_user)
    await db.commit()
    return {"message": "Account deleted successfully"} 