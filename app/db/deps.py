from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from app.core.config import settings
from app.core.security import ALGORITHM
from app.crud.crud_user import user as crud_user
from app.crud.crud_token import token as crud_token
from app.db.session import SessionLocal
from app.schemas.token import TokenPayload
from app.db.models.user import User

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RateLimiter:
    """Simple in-memory rate limiter with configurable limits."""
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 100):
        """
        Initialize rate limiter with configurable limits.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
            burst_limit: Maximum burst size allowed
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests = {}

    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            key: Unique identifier for the request (e.g., IP + token)
            
        Returns:
            bool: True if request is allowed, False otherwise
        """
        now = datetime.now().timestamp()
        minute_ago = now - 60

        # Initialize if key doesn't exist
        if key not in self.requests:
            self.requests[key] = []

        # Clean old requests
        self.requests[key] = [ts for ts in self.requests[key] if ts > minute_ago]

        # Get requests in current window
        requests_in_window = len(self.requests[key])
        
        # Check both rate limit and burst limit
        if requests_in_window >= self.requests_per_minute or requests_in_window >= self.burst_limit:
            return False

        # Add current request
        self.requests[key].append(now)
        return True


# Initialize rate limiter with settings
try:
    rate_limiter = RateLimiter(
        requests_per_minute=getattr(settings, 'API_RATE_LIMIT_PER_MINUTE', 100),
        burst_limit=getattr(settings, 'API_RATE_LIMIT_BURST', 160)
    )
    logger.info(f"Rate limiter initialized with {rate_limiter.requests_per_minute} requests/minute")
except Exception as e:
    logger.warning(f"Failed to initialize rate limiter with custom settings: {e}. Using defaults.")
    rate_limiter = RateLimiter()


def get_db() -> Generator:
    """
    Database session dependency.
    Yields a SQLAlchemy session and ensures proper cleanup.
    """
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Get current user from JWT token.
    Validates the token and checks if it's not blacklisted.
    """
    try:
        # Rate limiting check
        client_ip = request.client.host
        if not rate_limiter.is_allowed(f"{client_ip}:{token}"):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": "60"}
            )

        # Check if token is blacklisted
        if crud_token.is_blacklisted(db, token):
            logger.warning(f"Attempt to use blacklisted token from IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked. Please log in again.",
            )

        # Decode and validate token
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        # Check token expiration
        if token_data.exp < datetime.now():
            logger.warning(f"Expired token used from IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please log in again.",
            )

    except (jwt.JWTError, ValidationError) as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials. Please log in again.",
        )

    # Get user from database
    user = crud_user.get(db, id=token_data.sub)
    if not user:
        logger.error(f"User not found for token sub: {token_data.sub}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found. The account may have been deleted.",
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.
    Ensures the user account is active.
    """
    if not crud_user.is_active(current_user):
        logger.warning(f"Inactive user attempted access: {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is inactive. Please contact support for assistance.",
        )
    return current_user


async def get_current_active_verified_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Get current active and verified user.
    Ensures the user's email is verified.
    """
    if not crud_user.is_verified(current_user):
        logger.warning(f"Unverified user attempted access: {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required. Please check your email for verification instructions.",
        )
    return current_user


async def get_current_active_superuser(
    current_user: User = Depends(get_current_active_verified_user),
) -> User:
    """
    Get current active superuser.
    Ensures the user has admin privileges.
    """
    if not crud_user.is_admin(current_user):
        logger.warning(f"Non-admin user attempted admin access: {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges. This action requires administrator access.",
        )
    return current_user 