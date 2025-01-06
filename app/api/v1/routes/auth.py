from datetime import timedelta, datetime
from typing import Any
from fastapi import APIRouter, Body, Depends, HTTPException, status, Header, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core import security
from app.core.config import settings
from app.crud.crud_user import user as crud_user
from app.crud.crud_token import token as crud_token
from app.db import deps
from app.schemas.user import User, UserCreate
from app.schemas.token import Token, TokenPayload
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    db: Session = Depends(deps.get_db),
    user_in: UserCreate = Body(...),
) -> Any:
    """
    Register new user.
    
    Args:
        request: FastAPI request object
        db: Database session
        user_in: User creation data
        
    Returns:
        Newly created user object
        
    Raises:
        HTTPException: If email is already registered or passwords don't match
    """
    logger.info(f"Registration attempt for email: {user_in.email}")
    
    # Check if email exists
    user = crud_user.get_by_email(db, email=user_in.email)
    if user:
        logger.warning(f"Registration failed: Email already exists: {user_in.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    try:
        # Create new user
        user = crud_user.create(db, obj_in=user_in)
        logger.info(f"User registered successfully: {user.id}")
        
        # TODO: Send verification email
        # verification_token = security.create_email_verification_token(user.email)
        # send_verification_email(user.email, verification_token)
        
        return user
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user account",
        )


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login.
    
    Args:
        request: FastAPI request object
        db: Database session
        form_data: OAuth2 form containing username (email) and password
        
    Returns:
        Access token and token type
        
    Raises:
        HTTPException: If authentication fails or user is inactive
    """
    logger.info(f"Login attempt for user: {form_data.username}")
    
    # Check rate limiting
    client_ip = request.client.host
    if not deps.rate_limiter.is_allowed(f"login:{client_ip}"):
        logger.warning(f"Login rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": "60"}
        )
    
    # Try to authenticate
    user = crud_user.authenticate(
        db, email=form_data.username, password=form_data.password
    )
    
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    if not crud_user.is_active(user):
        logger.warning(f"Login attempt for inactive user: {user.id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is inactive. Please contact support.",
        )
    
    try:
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            user.id, expires_delta=access_token_expires
        )
        
        # Create refresh token
        refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
        refresh_token = security.create_refresh_token(
            user.id, expires_delta=refresh_token_expires
        )
        
        # Update last login timestamp
        crud_user.update_last_login(db, user=user)
        
        logger.info(f"User logged in successfully: {user.id}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing login request",
        )


@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    request: Request,
    db: Session = Depends(deps.get_db),
    refresh_token: str = Body(..., embed=True),
) -> Any:
    """
    Refresh access token using refresh token.
    
    Args:
        request: FastAPI request object
        db: Database session
        refresh_token: Valid refresh token
        
    Returns:
        New access token and refresh token
        
    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    try:
        # Verify refresh token
        payload = security.decode_token(refresh_token)
        token_data = TokenPayload(**payload)
        
        # Get user
        user = crud_user.get(db, id=token_data.sub)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        # Create new tokens
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            user.id, expires_delta=access_token_expires
        )
        
        refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
        new_refresh_token = security.create_refresh_token(
            user.id, expires_delta=refresh_token_expires
        )
        
        logger.info(f"Tokens refreshed for user: {user.id}")
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
        }
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.post("/password-recovery/{email}")
async def recover_password(
    request: Request,
    email: str,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Password Recovery.
    
    Args:
        request: FastAPI request object
        email: User's email address
        db: Database session
        
    Returns:
        Success message
        
    Note:
        Always returns success message even if email doesn't exist (security best practice)
    """
    logger.info(f"Password recovery requested for email: {email}")
    
    # Check rate limiting
    client_ip = request.client.host
    if not deps.rate_limiter.is_allowed(f"password_recovery:{client_ip}"):
        logger.warning(f"Password recovery rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
            headers={"Retry-After": "60"}
        )
    
    try:
        user = crud_user.get_by_email(db, email=email)
        if user and user.is_active:
            token = security.create_password_reset_token(email)
            expires = datetime.utcnow() + timedelta(hours=settings.EMAIL_RESET_TOKEN_EXPIRE_HOURS)
            
            # Update user with reset token
            crud_user.update(db, db_obj=user, obj_in={
                "password_reset_token": token,
                "password_reset_expires": expires
            })
            
            # TODO: Send password reset email
            # send_password_reset_email(email, token)
            
            logger.info(f"Password reset token created for user: {user.id}")
    except Exception as e:
        logger.error(f"Password recovery error: {str(e)}")
    
    # Always return success (prevents email enumeration)
    return {"msg": "If this email is registered, you will receive password reset instructions."}


@router.post("/reset-password")
async def reset_password(
    request: Request,
    token: str = Body(...),
    new_password: str = Body(...),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Reset password using reset token.
    
    Args:
        request: FastAPI request object
        token: Password reset token
        new_password: New password
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Verify token and get email
        email = security.verify_password_reset_token(token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )
        
        # Get user and verify token expiration
        user = crud_user.get_by_email(db, email=email)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token",
            )
        
        if not user.password_reset_token or user.password_reset_token != token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token",
            )
        
        if user.password_reset_expires and user.password_reset_expires < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired",
            )
        
        # Update password and clear reset token
        crud_user.set_password(db, user=user, password=new_password)
        crud_user.update(db, db_obj=user, obj_in={
            "password_reset_token": None,
            "password_reset_expires": None
        })
        
        # Invalidate all existing tokens
        crud_token.blacklist_all_user_tokens(db, user.id)
        
        logger.info(f"Password reset successful for user: {user.id}")
        
        return {"msg": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing password reset",
        )


@router.post("/verify/{token}")
async def verify_email(
    request: Request,
    token: str,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Verify email address.
    
    Args:
        request: FastAPI request object
        token: Email verification token
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Verify token
        user = crud_user.get_by_field(db, "verification_token", token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token",
            )
        
        if crud_user.is_verified(user):
            return {"msg": "Email already verified"}
        
        # Mark email as verified
        crud_user.mark_verified(db, user=user)
        logger.info(f"Email verified for user: {user.id}")
        
        return {"msg": "Email verified successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing email verification",
        )


@router.post("/logout")
async def logout(
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
    authorization: str = Header(...),
) -> Any:
    """
    Logout user by invalidating their JWT token.
    
    Args:
        request: FastAPI request object
        db: Database session
        current_user: Currently authenticated user
        authorization: Authorization header containing the token
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If token is invalid or error occurs during logout
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
        
        logger.info(f"User logged out successfully: {current_user.id}")
        
        return {"msg": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Logout error for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error processing logout request",
        ) 