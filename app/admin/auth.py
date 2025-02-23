from typing import Optional
from fastapi import Request, status
from starlette.responses import RedirectResponse
from sqladmin.authentication import AuthenticationBackend
from datetime import datetime, timedelta
import jwt as PyJWT
import logging
from sqlalchemy.orm import Session
import traceback

from app.db.session import get_db
from app.db.models.user import User, UserRole
from app.core.config import settings
from app.crud.crud_user import user as crud_user
from app.crud.crud_token import token as crud_token
from app.core.security import verify_password

logger = logging.getLogger(__name__)

class AdminAuth(AuthenticationBackend):
    """Enhanced authentication backend for admin interface with security features."""
    
    def __init__(self):
        super().__init__(secret_key=settings.SECRET_KEY)
        
    async def login(self, request: Request) -> bool:
        """Handle admin login with credentials."""
        db = None
        try:
            # Get form data
            form = await request.form()
            username = form.get("username")
            password = form.get("password")
            
            logger.debug(f"Admin login attempt for user: {username}")
            
            if not username or not password:
                logger.warning("Missing credentials in admin login attempt")
                return False
            
            # Get database session
            db = next(get_db())
            
            # Get user and verify credentials
            user = crud_user.get_by_email(db, email=username)
            if not user:
                logger.warning(f"User not found: {username}")
                return False
                
            if not verify_password(password, user.hashed_password):
                logger.warning(f"Invalid password for user: {username}")
                return False
            
            if not self._validate_admin_access(user):
                logger.warning(f"Non-admin user attempted login: {username}")
                return False
            
            # Create session data
            try:
                request.session.clear()  # Clear any existing session
                request.session["admin_authenticated"] = True
                request.session["admin_id"] = str(user.id)
                request.session["admin_email"] = user.email
                request.session["admin_role"] = user.role.value
                logger.info(f"Admin login successful: {user.email}")
                return True
            except Exception as session_error:
                logger.error(f"Session error during login: {str(session_error)}")
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"Admin login error: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        finally:
            if db:
                db.close()
                logger.debug("Database session closed")

    async def logout(self, request: Request) -> bool:
        """Handle admin logout."""
        try:
            # Log session data before clearing
            logger.debug(f"Logging out admin session: {request.session.get('admin_email')}")
            
            # Clear session
            request.session.clear()
            logger.info("Admin logout successful")
            return True
        except Exception as e:
            logger.error(f"Admin logout error: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def authenticate(self, request: Request) -> Optional[bool]:
        """Authenticate admin requests."""
        db = None
        try:
            # Check session authentication
            if not request.session.get("admin_authenticated"):
                logger.debug("No admin authentication in session")
                return self._redirect_to_login(request)
            
            # Get admin ID from session
            admin_id = request.session.get("admin_id")
            if not admin_id:
                logger.debug("No admin ID in session")
                return self._redirect_to_login(request)
            
            # Verify user still has admin access
            try:
                db = next(get_db())
                user = crud_user.get(db, id=admin_id)
                
                if not user:
                    logger.warning(f"Admin user not found: {admin_id}")
                    request.session.clear()
                    return self._redirect_to_login(request)
                
                if not self._validate_admin_access(user):
                    logger.warning(f"Invalid admin session detected: {admin_id}")
                    request.session.clear()
                    return self._redirect_to_login(request)
                
                return True
                
            except Exception as db_error:
                logger.error(f"Database error during authentication: {str(db_error)}")
                logger.error(traceback.format_exc())
                return self._redirect_to_login(request)
                
        except Exception as e:
            logger.error(f"Admin authentication error: {str(e)}")
            logger.error(traceback.format_exc())
            return self._redirect_to_login(request)
        finally:
            if db:
                db.close()
                logger.debug("Database session closed")

    def _validate_admin_access(self, user: Optional[User]) -> bool:
        """Validate if user has admin access."""
        try:
            is_valid = bool(
                user and 
                user.is_active and 
                user.is_verified and 
                user.role == UserRole.ADMIN
            )
            if not is_valid:
                logger.debug(f"Admin validation failed for user: {user.email if user else 'None'}")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating admin access: {str(e)}")
            return False

    def _create_admin_token(self, user: User) -> str:
        """Create a token for admin session."""
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "exp": expire,
            "type": "admin_access"
        }
        
        return PyJWT.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

    def _redirect_to_login(self, request: Request) -> RedirectResponse:
        """Return redirect response to login page."""
        try:
            return RedirectResponse(
                request.url_for("admin:login"),
                status_code=status.HTTP_302_FOUND
            )
        except Exception as e:
            logger.error(f"Error creating redirect response: {str(e)}")
            # Fallback to direct URL if url_for fails
            return RedirectResponse(
                "/admin/login",
                status_code=status.HTTP_302_FOUND
            ) 