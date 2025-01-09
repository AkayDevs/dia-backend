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
import uuid

router = APIRouter()

logger = logging.getLogger(__name__)

def get_request_info(request: Request) -> dict:
    """Extract common request information for logging."""
    return {
        "request_id": str(uuid.uuid4()),
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "path": request.url.path,
        "method": request.method
    }

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    db: Session = Depends(deps.get_db),
    user_in: UserCreate = Body(...),
) -> Any:
    """Register new user."""
    req_info = get_request_info(request)
    logger.info("Registration attempt initiated", extra={
        **req_info,
        "email": user_in.email,
        "event": "REGISTRATION_ATTEMPT"
    })
    
    # Check if email exists
    user = crud_user.get_by_email(db, email=user_in.email)
    if user:
        logger.warning("Registration failed - Email exists", extra={
            **req_info,
            "email": user_in.email,
            "event": "REGISTRATION_DUPLICATE_EMAIL"
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    try:
        user = crud_user.create(db, obj_in=user_in)
        logger.info("User registered successfully", extra={
            **req_info,
            "user_id": str(user.id),
            "email": user_in.email,
            "event": "REGISTRATION_SUCCESS"
        })
        return user
    except Exception as e:
        logger.error("Registration error", extra={
            **req_info,
            "email": user_in.email,
            "error": str(e),
            "event": "REGISTRATION_ERROR"
        })
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
    """OAuth2 compatible token login."""
    req_info = get_request_info(request)
    logger.info("Login attempt", extra={
        **req_info,
        "username": form_data.username,
        "event": "LOGIN_ATTEMPT"
    })
    
    # Check rate limiting
    if not deps.rate_limiter.is_allowed(f"login:{req_info['ip_address']}"):
        logger.warning("Login rate limit exceeded", extra={
            **req_info,
            "username": form_data.username,
            "event": "LOGIN_RATE_LIMIT_EXCEEDED"
        })
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
        logger.warning("Failed login attempt - Invalid credentials", extra={
            **req_info,
            "username": form_data.username,
            "event": "LOGIN_FAILED"
        })
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    if not crud_user.is_active(user):
        logger.warning("Login attempt for inactive account", extra={
            **req_info,
            "user_id": str(user.id),
            "username": form_data.username,
            "event": "LOGIN_INACTIVE_ACCOUNT"
        })
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
        
        logger.info("Login successful", extra={
            **req_info,
            "user_id": str(user.id),
            "username": form_data.username,
            "event": "LOGIN_SUCCESS",
            "access_token_expires": access_token_expires.total_seconds(),
            "refresh_token_expires": refresh_token_expires.total_seconds()
        })
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
    except Exception as e:
        logger.error("Login processing error", extra={
            **req_info,
            "user_id": str(user.id),
            "username": form_data.username,
            "error": str(e),
            "event": "LOGIN_ERROR"
        })
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
    """Refresh access token using refresh token."""
    req_info = get_request_info(request)
    logger.info("Token refresh attempt", extra={
        **req_info,
        "event": "TOKEN_REFRESH_ATTEMPT"
    })
    
    try:
        payload = security.decode_token(refresh_token)
        token_data = TokenPayload(**payload)
        user = crud_user.get(db, id=token_data.sub)
        
        if not user or not user.is_active:
            logger.warning("Invalid refresh token attempt", extra={
                **req_info,
                "token_sub": token_data.sub,
                "event": "TOKEN_REFRESH_INVALID"
            })
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            user.id, expires_delta=access_token_expires
        )
        
        refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
        new_refresh_token = security.create_refresh_token(
            user.id, expires_delta=refresh_token_expires
        )
        
        logger.info("Token refresh successful", extra={
            **req_info,
            "user_id": str(user.id),
            "event": "TOKEN_REFRESH_SUCCESS",
            "access_token_expires": access_token_expires.total_seconds(),
            "refresh_token_expires": refresh_token_expires.total_seconds()
        })
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
        }
    except Exception as e:
        logger.error("Token refresh error", extra={
            **req_info,
            "error": str(e),
            "event": "TOKEN_REFRESH_ERROR"
        })
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
    """Password Recovery."""
    req_info = get_request_info(request)
    logger.info("Password recovery initiated", extra={
        **req_info,
        "email": email,
        "event": "PASSWORD_RECOVERY_ATTEMPT"
    })
    
    if not deps.rate_limiter.is_allowed(f"password_recovery:{req_info['ip_address']}"):
        logger.warning("Password recovery rate limit exceeded", extra={
            **req_info,
            "email": email,
            "event": "PASSWORD_RECOVERY_RATE_LIMIT"
        })
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
            
            crud_user.update(db, db_obj=user, obj_in={
                "password_reset_token": token,
                "password_reset_expires": expires
            })
            
            logger.info("Password reset token created", extra={
                **req_info,
                "user_id": str(user.id),
                "email": email,
                "token_expires": expires.isoformat(),
                "event": "PASSWORD_RECOVERY_TOKEN_CREATED"
            })
    except Exception as e:
        logger.error("Password recovery error", extra={
            **req_info,
            "email": email,
            "error": str(e),
            "event": "PASSWORD_RECOVERY_ERROR"
        })
    
    return {"msg": "If this email is registered, you will receive password reset instructions."}

@router.post("/reset-password")
async def reset_password(
    request: Request,
    token: str = Body(...),
    new_password: str = Body(...),
    db: Session = Depends(deps.get_db),
) -> Any:
    """Reset password using reset token."""
    req_info = get_request_info(request)
    logger.info("Password reset attempt", extra={
        **req_info,
        "event": "PASSWORD_RESET_ATTEMPT"
    })
    
    try:
        email = security.verify_password_reset_token(token)
        if not email:
            logger.warning("Invalid password reset token", extra={
                **req_info,
                "event": "PASSWORD_RESET_INVALID_TOKEN"
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )
        
        user = crud_user.get_by_email(db, email=email)
        if not user or not user.is_active:
            logger.warning("Password reset attempt for invalid user", extra={
                **req_info,
                "email": email,
                "event": "PASSWORD_RESET_INVALID_USER"
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token",
            )
        
        if not user.password_reset_token or user.password_reset_token != token:
            logger.warning("Password reset token mismatch", extra={
                **req_info,
                "user_id": str(user.id),
                "event": "PASSWORD_RESET_TOKEN_MISMATCH"
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token",
            )
        
        if user.password_reset_expires and user.password_reset_expires < datetime.utcnow():
            logger.warning("Expired password reset token", extra={
                **req_info,
                "user_id": str(user.id),
                "event": "PASSWORD_RESET_TOKEN_EXPIRED"
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired",
            )
        
        crud_user.set_password(db, user=user, password=new_password)
        crud_user.update(db, db_obj=user, obj_in={
            "password_reset_token": None,
            "password_reset_expires": None
        })
        
        crud_token.blacklist_all_user_tokens(db, user.id)
        
        logger.info("Password reset successful", extra={
            **req_info,
            "user_id": str(user.id),
            "event": "PASSWORD_RESET_SUCCESS"
        })
        
        return {"msg": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password reset error", extra={
            **req_info,
            "error": str(e),
            "event": "PASSWORD_RESET_ERROR"
        })
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
    """Verify email address."""
    req_info = get_request_info(request)
    logger.info("Email verification attempt", extra={
        **req_info,
        "token": token[:8] + "...",  # Log only first 8 chars of token
        "event": "EMAIL_VERIFICATION_ATTEMPT"
    })
    
    try:
        user = crud_user.get_by_field(db, "verification_token", token)
        if not user:
            logger.warning("Invalid email verification token", extra={
                **req_info,
                "event": "EMAIL_VERIFICATION_INVALID_TOKEN"
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token",
            )
        
        if crud_user.is_verified(user):
            logger.info("Email already verified", extra={
                **req_info,
                "user_id": str(user.id),
                "event": "EMAIL_VERIFICATION_ALREADY_VERIFIED"
            })
            return {"msg": "Email already verified"}
        
        crud_user.mark_verified(db, user=user)
        logger.info("Email verification successful", extra={
            **req_info,
            "user_id": str(user.id),
            "event": "EMAIL_VERIFICATION_SUCCESS"
        })
        
        return {"msg": "Email verified successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Email verification error", extra={
            **req_info,
            "error": str(e),
            "event": "EMAIL_VERIFICATION_ERROR"
        })
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
    """Logout user by invalidating their JWT token."""
    req_info = get_request_info(request)
    logger.info("Logout attempt", extra={
        **req_info,
        "user_id": str(current_user.id),
        "event": "LOGOUT_ATTEMPT"
    })
    
    try:
        if not authorization.startswith("Bearer "):
            logger.warning("Invalid authorization header", extra={
                **req_info,
                "user_id": str(current_user.id),
                "event": "LOGOUT_INVALID_HEADER"
            })
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header",
            )
        
        token = authorization.split(" ")[1]
        payload = security.decode_token(token)
        expires_at = datetime.fromtimestamp(payload.get("exp"))
        
        crud_token.blacklist_token(db, token, expires_at)
        crud_token.cleanup_expired_tokens(db)
        
        logger.info("Logout successful", extra={
            **req_info,
            "user_id": str(current_user.id),
            "token_expires": expires_at.isoformat(),
            "event": "LOGOUT_SUCCESS"
        })
        
        return {"msg": "Successfully logged out"}
    except Exception as e:
        logger.error("Logout error", extra={
            **req_info,
            "user_id": str(current_user.id),
            "error": str(e),
            "event": "LOGOUT_ERROR"
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error processing logout request",
        ) 