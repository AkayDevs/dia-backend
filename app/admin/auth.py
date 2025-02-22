from typing import Optional
from fastapi import Request, status
from starlette.responses import RedirectResponse
from sqladmin.authentication import AuthenticationBackend
from datetime import datetime
import jwt as PyJWT
import logging

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
        """
        Handle admin login with both token and credentials support.
        
        Features:
        - Support for both token and username/password authentication
        - Token blacklist checking
        - Role-based access control
        - Account verification check
        - Secure session management
        - Comprehensive logging
        """
        try:
            form = await request.form()
            auth_header = request.headers.get("Authorization")
            
            # Try to get token from form or header
            token = None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
            elif "token" in form:
                token = form["token"]
                
            # If no token, try username/password
            if not token and "username" in form and "password" in form:
                db = next(get_db())
                user = crud_user.get_by_email(db, email=form["username"])
                
                if user and verify_password(form["password"], user.hashed_password):
                    if not self._validate_admin_access(user):
                        logger.warning(f"Invalid admin login attempt for user: {user.email}")
                        return False
                        
                    self._set_session(request, user)
                    logger.info(f"Admin login successful: {user.email}")
                    return True
                    
                logger.warning(f"Failed admin login attempt for user: {form['username']}")
                return False
                
            if not token:
                return False
                
            # Verify token
            try:
                payload = PyJWT.decode(
                    token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
                )
            except PyJWT.InvalidTokenError:
                logger.warning("Invalid token used for admin access")
                return False
            
            db = next(get_db())
            
            # Check if token is blacklisted
            if crud_token.is_blacklisted(db, token):
                logger.warning("Attempted use of blacklisted token for admin access")
                return False
                
            user = crud_user.get(db, id=payload.get("sub"))
            if not self._validate_admin_access(user):
                logger.warning(f"Invalid admin access attempt with token for user ID: {payload.get('sub')}")
                return False
                
            self._set_session(request, user)
            logger.info(f"Admin token login successful: {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Admin login error: {str(e)}")
            return False

    async def logout(self, request: Request) -> bool:
        """
        Handle admin logout with enhanced security.
        
        Features:
        - Token blacklisting
        - Session cleanup
        - Secure logout handling
        """
        try:
            # Blacklist current token if present
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                db = next(get_db())
                
                try:
                    # Get token expiry from payload
                    payload = PyJWT.decode(
                        token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
                    )
                    expires_at = datetime.fromtimestamp(payload.get("exp"))
                    
                    # Blacklist the token
                    crud_token.blacklist_token(db, token, expires_at)
                except PyJWT.InvalidTokenError:
                    logger.warning("Invalid token encountered during logout")
                
            # Clear session
            request.session.clear()
            logger.info("Admin logout successful")
            return True
            
        except Exception as e:
            logger.error(f"Admin logout error: {str(e)}")
            return False

    async def authenticate(self, request: Request) -> Optional[bool]:
        """
        Authenticate admin requests with enhanced security.
        
        Features:
        - Session validation
        - Token validation
        - Role-based access control
        - Account status verification
        """
        try:
            if request.session.get("admin_authenticated"):
                # Verify session data
                admin_id = request.session.get("admin_id")
                if not admin_id:
                    return self._redirect_to_login(request)
                
                db = next(get_db())
                user = crud_user.get(db, id=admin_id)
                if not self._validate_admin_access(user):
                    return self._redirect_to_login(request)
                    
                return True
                
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return self._redirect_to_login(request)
                
            token = auth_header.split(" ")[1]
            db = next(get_db())
            
            # Check if token is blacklisted
            if crud_token.is_blacklisted(db, token):
                logger.warning("Attempted use of blacklisted token for admin authentication")
                return self._redirect_to_login(request)
                
            try:
                payload = PyJWT.decode(
                    token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
                )
            except PyJWT.InvalidTokenError:
                logger.warning("Invalid token used for admin authentication")
                return self._redirect_to_login(request)
                
            user = crud_user.get(db, id=payload.get("sub"))
            if not self._validate_admin_access(user):
                logger.warning(f"Invalid admin authentication attempt for user ID: {payload.get('sub')}")
                return self._redirect_to_login(request)
                
            self._set_session(request, user)
            return True
            
        except Exception as e:
            logger.error(f"Admin authentication error: {str(e)}")
            return self._redirect_to_login(request)

    def _validate_admin_access(self, user: Optional[User]) -> bool:
        """
        Validate if user has admin access.
        
        Checks:
        - User exists
        - User is active
        - User is verified
        - User has admin role
        """
        return bool(
            user and 
            user.is_active and 
            user.is_verified and 
            user.role == UserRole.ADMIN
        )

    def _set_session(self, request: Request, user: User) -> None:
        """Set secure session data."""
        request.session.update({
            "admin_authenticated": True,
            "admin_id": str(user.id),
            "admin_email": user.email,
            "admin_role": user.role.value
        })

    def _redirect_to_login(self, request: Request) -> RedirectResponse:
        """Return redirect response to login page."""
        return RedirectResponse(
            request.url_for("admin:login"),
            status_code=status.HTTP_302_FOUND
        ) 