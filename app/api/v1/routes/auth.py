from datetime import timedelta, datetime
from typing import Any
from fastapi import APIRouter, Body, Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import logging

from app.core import security
from app.core.config import settings
from app.crud.crud_user import user as crud_user
from app.crud.crud_token import token as crud_token
from app.db import deps
from app.schemas.user import User, UserCreate
from app.schemas.token import Token

router = APIRouter()
logger = logging.getLogger("app.api.auth")

# Add a debug message to verify logger is working
logger.debug("Auth router initialized")


@router.post("/register", response_model=User)
def register(
    *,
    db: Session = Depends(deps.get_db),
    user_in: UserCreate,
) -> Any:
    """
    Register new user.
    """
    user = crud_user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Validate password confirmation
    if user_in.password != user_in.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match",
        )
    
    user = crud_user.create(db, obj_in=user_in)
    return user


@router.post("/login", response_model=Token)
async def login(
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login.
    """
    # Try to authenticate
    user = crud_user.authenticate(
        db, email=form_data.username, password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
        
    if not crud_user.is_active(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    # Create access token    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = security.create_access_token(
        user.id, expires_delta=access_token_expires
    )
    
    return {
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/password-recovery/{email}")
def recover_password(email: str, db: Session = Depends(deps.get_db)) -> Any:
    """
    Password Recovery.
    """
    user = crud_user.get_by_email(db, email=email)
    if user:
        password_reset_token = security.generate_password_reset_token(email)
        # TODO: Send password reset email
        user_in = {"password_reset_token": password_reset_token}
        crud_user.update(db, db_obj=user, obj_in=user_in)
    return {"msg": "Password recovery email sent"}


@router.post("/reset-password")
def reset_password(
    token: str = Body(...),
    new_password: str = Body(...),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Reset password.
    """
    email = security.verify_password_reset_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token",
        )
    user = crud_user.get_by_email(db, email=email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    if not crud_user.is_active(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    crud_user.set_password(db, user=user, password=new_password)
    return {"msg": "Password updated successfully"}


@router.post("/verify/{token}")
def verify_email(
    token: str,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Verify email address.
    """
    user = crud_user.get_by_field(db, "verification_token", token)
    if not user or crud_user.is_verified(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token",
        )
    crud_user.mark_verified(db, user=user)
    return {"msg": "Email verified successfully"}


@router.post("/logout")
async def logout(
    db: Session = Depends(deps.get_db),
    authorization: str = Header(...),
) -> Any:
    """
    Logout user by invalidating their JWT token.
    """
    try:
        # Extract token from Authorization header
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header",
            )
        token = authorization.split(" ")[1]
        
        # Decode token to get expiry
        payload = security.decode_token(token)
        expires_at = datetime.fromtimestamp(payload.get("exp"))
        
        # Blacklist the token
        crud_token.blacklist_token(db, token, expires_at)
        
        # Cleanup expired tokens (maintenance)
        crud_token.cleanup_expired_tokens(db)
        
        return {"msg": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error processing logout request",
        ) 