from sqladmin import ModelView, Admin
from sqladmin.authentication import AuthenticationBackend
from app.db.models.document import Document
from app.db.models.analysis_result import AnalysisResult
from app.db.models.token import BlacklistedToken
from app.db.models.user import User, UserRole
from app.core.config import settings
from fastapi import Request, status
from typing import Optional
from starlette.responses import RedirectResponse
from app.db.session import get_db
import jwt as PyJWT
from datetime import datetime
import logging
from app.schemas.analysis import AnalysisStatus
from markupsafe import Markup

logger = logging.getLogger(__name__)

class AdminAuth(AuthenticationBackend):
    """Authentication backend for admin interface with enhanced security."""
    
    def __init__(self):
        super().__init__(secret_key=settings.SECRET_KEY)
        
    async def login(self, request: Request) -> bool:
        """Handle admin login with both token and credentials support."""
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
                from app.crud.crud_user import user as crud_user
                from app.crud.crud_token import token as crud_token
                
                db = next(get_db())
                user = crud_user.authenticate(
                    db, 
                    email=form["username"], 
                    password=form["password"]
                )
                
                if user and user.is_active and user.role == UserRole.ADMIN:
                    # Check if user is verified
                    if not user.is_verified:
                        logger.warning(f"Unverified admin login attempt: {user.email}")
                        return False
                        
                    request.session.update({
                        "admin_authenticated": True,
                        "admin_id": str(user.id),
                        "admin_email": user.email
                    })
                    logger.info(f"Admin login successful: {user.email}")
                    return True
                    
                logger.warning(f"Failed admin login attempt for user: {form['username']}")
                return False
                
            if not token:
                return False
                
            # Verify token
            payload = PyJWT.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            
            db = next(get_db())
            # Check if token is blacklisted
            from app.crud.crud_token import token as crud_token
            if crud_token.is_blacklisted(db, token):
                logger.warning("Attempted use of blacklisted token for admin access")
                return False
                
            user = db.query(User).filter(User.id == payload.get("sub")).first()
            
            if not user or not user.is_active or user.role != UserRole.ADMIN:
                logger.warning(f"Invalid admin access attempt with token for user ID: {payload.get('sub')}")
                return False
                
            if not user.is_verified:
                logger.warning(f"Unverified admin access attempt: {user.email}")
                return False
                
            request.session.update({
                "admin_authenticated": True,
                "admin_id": str(user.id),
                "admin_email": user.email
            })
            logger.info(f"Admin token login successful: {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Admin login error: {str(e)}")
            return False

    async def logout(self, request: Request) -> bool:
        """Handle admin logout with token blacklisting."""
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                db = next(get_db())
                from app.crud.crud_token import token as crud_token
                
                # Get token expiry from payload
                payload = PyJWT.decode(
                    token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
                )
                expires_at = datetime.fromtimestamp(payload.get("exp"))
                
                # Blacklist the token
                crud_token.blacklist_token(db, token, expires_at)
                
            request.session.clear()
            logger.info("Admin logout successful")
            return True
            
        except Exception as e:
            logger.error(f"Admin logout error: {str(e)}")
            return False

    async def authenticate(self, request: Request) -> Optional[bool]:
        """Authenticate admin requests with enhanced security."""
        try:
            if request.session.get("admin_authenticated"):
                return True
                
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return RedirectResponse(
                    request.url_for("admin:login"),
                    status_code=status.HTTP_302_FOUND
                )
                
            token = auth_header.split(" ")[1]
            db = next(get_db())
            
            # Check if token is blacklisted
            from app.crud.crud_token import token as crud_token
            if crud_token.is_blacklisted(db, token):
                logger.warning("Attempted use of blacklisted token for admin authentication")
                return RedirectResponse(
                    request.url_for("admin:login"),
                    status_code=status.HTTP_302_FOUND
                )
                
            payload = PyJWT.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            user = db.query(User).filter(User.id == payload.get("sub")).first()
            
            if not user or not user.is_active or user.role != UserRole.ADMIN or not user.is_verified:
                logger.warning(f"Invalid admin authentication attempt for user ID: {payload.get('sub')}")
                return RedirectResponse(
                    request.url_for("admin:login"),
                    status_code=status.HTTP_302_FOUND
                )
                
            request.session.update({
                "admin_authenticated": True,
                "admin_id": str(user.id),
                "admin_email": user.email
            })
            return True
            
        except Exception as e:
            logger.error(f"Admin authentication error: {str(e)}")
            return RedirectResponse(
                request.url_for("admin:login"),
                status_code=status.HTTP_302_FOUND
            )


class UserAdmin(ModelView, model=User):
    """Admin interface for User model."""
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-users"
    column_list = [
        User.id, User.email, User.name, User.role, 
        User.is_active, User.is_verified, User.created_at, 
        User.updated_at
    ]
    column_searchable_list = [User.email, User.name]
    column_sortable_list = [
        User.email, User.name, User.role, 
        User.is_active, User.created_at, User.updated_at
    ]
    column_formatters = {
        User.is_active: lambda m, a: Markup("✓") if m.is_active else Markup("✗"),
        User.is_verified: lambda m, a: Markup("✓") if m.is_verified else Markup("✗"),
        User.role: lambda m, a: Markup(f"<span class='badge badge-{'primary' if m.role == UserRole.ADMIN else 'secondary'}'>{m.role.value}</span>")
    }
    
    can_create = False
    can_delete = False
    can_edit = True
    
    form_columns = [
        User.email,
        User.name,
        User.role,
        User.is_active,
        User.is_verified,
    ]
    
    column_descriptions = {
        User.email: "User's email address",
        User.name: "User's full name",
        User.role: "User's role (admin or user)",
        User.is_active: "Whether the user account is active",
        User.is_verified: "Whether the user's email is verified",
        User.created_at: "Account creation date",
        User.updated_at: "Last update date"
    }
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS


class DocumentAdmin(ModelView, model=Document):
    """Admin interface for Document model."""
    name = "Document"
    name_plural = "Documents"
    icon = "fa-solid fa-file"
    
    # Enhanced column list with all relevant fields
    column_list = [
        Document.id,
        Document.name,
        Document.type,
        Document.size,
        Document.uploaded_at,
        Document.updated_at,
        Document.user_id,
        Document.file_hash,
        Document.url
    ]
    
    # Enhanced searchable list
    column_searchable_list = [
        Document.name,
        Document.id,
        Document.user_id,
        Document.file_hash
    ]
    
    # Enhanced sortable list
    column_sortable_list = [
        Document.uploaded_at,
        Document.updated_at,
        Document.name,
        Document.size,
        Document.type
    ]
    
    # Improved formatters with better visual representation
    column_formatters = {
        Document.type: lambda m, a: Markup(f"<span class='badge badge-info'>{m.type.value}</span>"),
        Document.size: lambda m, a: Markup(f"{m.size / (1024 * 1024):.2f} MB") if m.size >= 1024 * 1024 else Markup(f"{m.size / 1024:.2f} KB"),
        Document.url: lambda m, a: Markup(f"<a href='{m.url}' target='_blank'>{m.url.split('/')[-1]}</a>"),
        Document.file_hash: lambda m, a: Markup(f"<span class='text-muted'>{m.file_hash[:8]}...{m.file_hash[-8:]}</span>")
    }
    
    # Enhanced column descriptions
    column_descriptions = {
        Document.id: "Unique identifier for the document",
        Document.name: "Original filename of the document",
        Document.type: "Type of document (PDF, DOCX, XLSX, IMAGE)",
        Document.size: "File size in bytes",
        Document.uploaded_at: "When the document was uploaded",
        Document.updated_at: "Last modification date",
        Document.user_id: "ID of the user who owns this document",
        Document.file_hash: "SHA-256 hash of the file content for deduplication",
        Document.url: "Storage location of the document"
    }
    
    # Details view with all fields and relationships
    column_details_list = [
        Document.id,
        Document.name,
        Document.type,
        Document.size,
        Document.uploaded_at,
        Document.updated_at,
        Document.user_id,
        Document.file_hash,
        Document.url,
        'tags',
        'analysis_results'
    ]
    
    # Keep security settings
    can_create = False
    can_delete = False
    can_edit = False
    
    # Pagination settings from global config
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS


class AnalysisResultAdmin(ModelView, model=AnalysisResult):
    """Admin interface for AnalysisResult model."""
    name = "Analysis Result"
    name_plural = "Analysis Results"
    icon = "fa-solid fa-chart-simple"
    column_list = [
        AnalysisResult.id,
        AnalysisResult.document_id,
        AnalysisResult.type,
        AnalysisResult.status,
        AnalysisResult.progress,
        AnalysisResult.created_at,
        AnalysisResult.completed_at
    ]
    column_searchable_list = [
        AnalysisResult.document_id,
        AnalysisResult.status,
        AnalysisResult.type
    ]
    column_sortable_list = [
        AnalysisResult.created_at,
        AnalysisResult.completed_at,
        AnalysisResult.type,
        AnalysisResult.status,
        AnalysisResult.progress
    ]
    column_formatters = {
        AnalysisResult.type: lambda m, a: Markup(f"<span class='badge badge-info'>{m.type.value}</span>"),
        AnalysisResult.status: lambda m, a: Markup(f"<span class='badge badge-{_get_status_badge(m.status)}'>{m.status.value}</span>"),
        AnalysisResult.progress: lambda m, a: Markup(f"<div class='progress'><div class='progress-bar' role='progressbar' style='width: {m.progress}%' aria-valuenow='{m.progress}' aria-valuemin='0' aria-valuemax='100'>{m.progress}%</div></div>")
    }
    
    column_descriptions = {
        AnalysisResult.document_id: "Associated document ID",
        AnalysisResult.type: "Type of analysis performed",
        AnalysisResult.status: "Current status of the analysis",
        AnalysisResult.progress: "Analysis progress (0-100%)",
        AnalysisResult.status_message: "Current status message",
        AnalysisResult.error: "Error message if analysis failed",
        AnalysisResult.created_at: "When analysis was started",
        AnalysisResult.completed_at: "When analysis was completed",
        AnalysisResult.result: "Analysis results (JSON)",
        AnalysisResult.parameters: "Analysis parameters (JSON)"
    }
    
    column_details_list = [
        AnalysisResult.id,
        AnalysisResult.document_id,
        AnalysisResult.type,
        AnalysisResult.status,
        AnalysisResult.progress,
        AnalysisResult.status_message,
        AnalysisResult.error,
        AnalysisResult.parameters,
        AnalysisResult.result,
        AnalysisResult.created_at,
        AnalysisResult.completed_at
    ]
    
    can_create = False
    can_delete = False
    can_edit = False
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

def _get_status_badge(status: AnalysisStatus) -> str:
    """Get appropriate Bootstrap badge class for status."""
    return {
        AnalysisStatus.PENDING: "warning",
        AnalysisStatus.PROCESSING: "info",
        AnalysisStatus.COMPLETED: "success",
        AnalysisStatus.FAILED: "danger"
    }.get(status, "secondary")


class BlacklistedTokenAdmin(ModelView, model=BlacklistedToken):
    """Admin interface for BlacklistedToken model."""
    name = "Blacklisted Token"
    name_plural = "Blacklisted Tokens"
    icon = "fa-solid fa-ban"
    column_list = [
        BlacklistedToken.token,
        BlacklistedToken.blacklisted_on,
        BlacklistedToken.expires_at
    ]
    column_sortable_list = [
        BlacklistedToken.blacklisted_on,
        BlacklistedToken.expires_at
    ]
    
    column_descriptions = {
        BlacklistedToken.token: "JWT token",
        BlacklistedToken.blacklisted_on: "When the token was blacklisted",
        BlacklistedToken.expires_at: "Token expiration date"
    }
    
    can_create = False
    can_delete = True  # Allow manual cleanup
    can_edit = False
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS


def setup_admin(app, engine) -> Admin:
    """Setup admin interface with all views and authentication."""
    try:
        authentication_backend = AdminAuth()
        admin = Admin(
            app,
            engine,
            authentication_backend=authentication_backend,
            title=f"{settings.PROJECT_NAME} Admin",
            base_url=settings.ADMIN_BASE_URL,
            logo_url=None  # You can add a logo URL here if needed
        )
        
        # Add views
        admin.add_view(UserAdmin)
        admin.add_view(DocumentAdmin)
        admin.add_view(AnalysisResultAdmin)
        admin.add_view(BlacklistedTokenAdmin)
        
        logger.info("Admin interface setup completed successfully")
        return admin
        
    except Exception as e:
        logger.error(f"Error setting up admin interface: {str(e)}")
        raise 