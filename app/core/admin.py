from sqladmin import ModelView, Admin
from sqladmin.authentication import AuthenticationBackend
from app.db.models.document import Document, Tag
from app.db.models.analysis import (
    AnalysisType, AnalysisStep, Algorithm, Analysis,
    AnalysisStepResult, AnalysisTypeEnum, AnalysisStepEnum
)
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
from fastapi import FastAPI

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

class TagAdmin(ModelView, model=Tag):
    """Admin interface for Tag model."""
    name = "Tag"
    name_plural = "Tags"
    icon = "fa-solid fa-tag"
    column_list = [
        Tag.id,
        Tag.name,
        Tag.created_at
    ]
    column_searchable_list = [Tag.name]
    column_sortable_list = [
        Tag.name,
        Tag.created_at
    ]
    
    column_descriptions = {
        Tag.name: "Tag name",
        Tag.created_at: "When the tag was created"
    }
    
    can_create = True
    can_delete = True
    can_edit = True
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

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
        User.is_active: lambda m, a: "✓" if m.is_active else "✗",
        User.is_verified: lambda m, a: "✓" if m.is_verified else "✗",
        User.role: lambda m, a: f"<span class='badge badge-{'primary' if m.role == UserRole.ADMIN else 'secondary'}'>{m.role.value}</span>"
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
    column_list = [
        Document.id,
        Document.name,
        Document.type,
        Document.uploaded_at,
        Document.updated_at,
        Document.size,
        Document.user_id,
        'tags',
        Document.is_archived,
        Document.archived_at,
        Document.retention_until
    ]
    column_searchable_list = [
        Document.name,
        Document.user_id,
        Document.previous_version_id
    ]
    column_sortable_list = [
        Document.uploaded_at,
        Document.updated_at,
        Document.name,
        Document.size,
        Document.is_archived,
        Document.archived_at,
        Document.retention_until
    ]
    column_formatters = {
        Document.type: lambda m, a: f"<span class='badge badge-info'>{m.type.value}</span>",
        Document.size: lambda m, a: f"{m.size / 1024:.2f} KB",
        'tags': lambda m, a: ", ".join([f"<span class='badge badge-secondary'>{tag.name}</span>" for tag in m.tags]),
        Document.is_archived: lambda m, a: "✓" if m.is_archived else "✗",
        Document.previous_version_id: lambda m, a: f"<a href='/admin/document/{m.previous_version_id}'>{m.previous_version_id}</a>" if m.previous_version_id else "-"
    }
    
    column_descriptions = {
        Document.name: "Document name",
        Document.type: "Document type",
        Document.uploaded_at: "Upload date",
        Document.updated_at: "Last update date",
        Document.size: "File size in bytes",
        Document.user_id: "Owner user ID",
        'tags': "Document tags",
        Document.previous_version_id: "ID of the previous version of this document",
        Document.is_archived: "Whether the document is archived",
        Document.archived_at: "When the document was archived",
        Document.retention_until: "Date until which the document must be retained"
    }
    
    can_create = False
    can_delete = False
    can_edit = True  # Allow editing for tags and archive status
    form_columns = [
        Document.name,
        'tags',
        Document.is_archived,
        Document.retention_until
    ]
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

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

class AnalysisTypeAdmin(ModelView, model=AnalysisType):
    """Admin interface for AnalysisType model."""
    name = "Analysis Type"
    name_plural = "Analysis Types"
    icon = "fa-solid fa-cube"
    column_list = [
        AnalysisType.id,
        AnalysisType.name,
        AnalysisType.description,
        AnalysisType.supported_document_types,
        AnalysisType.created_at,
        AnalysisType.updated_at
    ]
    column_searchable_list = [AnalysisType.name, AnalysisType.description]
    column_sortable_list = [
        AnalysisType.name,
        AnalysisType.created_at,
        AnalysisType.updated_at
    ]
    column_formatters = {
        AnalysisType.name: lambda m, a: f"<span class='badge badge-primary'>{m.name.value}</span>",
        AnalysisType.supported_document_types: lambda m, a: ", ".join(m.supported_document_types)
    }
    
    can_create = True
    can_delete = True
    can_edit = True
    
    form_columns = [
        AnalysisType.name,
        AnalysisType.description,
        AnalysisType.supported_document_types
    ]
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

class AnalysisStepAdmin(ModelView, model=AnalysisStep):
    """Admin interface for AnalysisStep model."""
    name = "Analysis Step"
    name_plural = "Analysis Steps"
    icon = "fa-solid fa-list-ol"
    column_list = [
        AnalysisStep.id,
        AnalysisStep.name,
        AnalysisStep.description,
        AnalysisStep.order,
        AnalysisStep.analysis_type_id,
        AnalysisStep.created_at,
        AnalysisStep.updated_at
    ]
    column_searchable_list = [
        AnalysisStep.name,
        AnalysisStep.description,
        AnalysisStep.analysis_type_id
    ]
    column_sortable_list = [
        AnalysisStep.order,
        AnalysisStep.created_at,
        AnalysisStep.updated_at
    ]
    column_formatters = {
        AnalysisStep.name: lambda m, a: f"<span class='badge badge-info'>{m.name.value}</span>"
    }
    
    can_create = True
    can_delete = True
    can_edit = True
    
    form_columns = [
        AnalysisStep.name,
        AnalysisStep.description,
        AnalysisStep.order,
        AnalysisStep.analysis_type_id,
        AnalysisStep.base_parameters
    ]
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

class AlgorithmAdmin(ModelView, model=Algorithm):
    """Admin interface for Algorithm model."""
    name = "Algorithm"
    name_plural = "Algorithms"
    icon = "fa-solid fa-microchip"
    column_list = [
        Algorithm.id,
        Algorithm.name,
        Algorithm.description,
        Algorithm.version,
        Algorithm.step_id,
        Algorithm.is_active,
        Algorithm.created_at,
        Algorithm.updated_at
    ]
    column_searchable_list = [
        Algorithm.name,
        Algorithm.description,
        Algorithm.version,
        Algorithm.step_id
    ]
    column_sortable_list = [
        Algorithm.name,
        Algorithm.version,
        Algorithm.is_active,
        Algorithm.created_at,
        Algorithm.updated_at
    ]
    column_formatters = {
        Algorithm.is_active: lambda m, a: "✓" if m.is_active else "✗",
        Algorithm.supported_document_types: lambda m, a: ", ".join(m.supported_document_types)
    }
    
    can_create = True
    can_delete = True
    can_edit = True
    
    form_columns = [
        Algorithm.name,
        Algorithm.description,
        Algorithm.version,
        Algorithm.step_id,
        Algorithm.supported_document_types,
        Algorithm.is_active,
        Algorithm.parameters
    ]
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

class AnalysisAdmin(ModelView, model=Analysis):
    """Admin interface for Analysis model."""
    name = "Analysis"
    name_plural = "Analyses"
    icon = "fa-solid fa-microscope"
    column_list = [
        Analysis.id,
        Analysis.document_id,
        Analysis.analysis_type_id,
        Analysis.mode,
        Analysis.status,
        Analysis.created_at,
        Analysis.updated_at,
        Analysis.completed_at
    ]
    column_searchable_list = [
        Analysis.document_id,
        Analysis.analysis_type_id,
        Analysis.status
    ]
    column_sortable_list = [
        Analysis.created_at,
        Analysis.updated_at,
        Analysis.completed_at,
        Analysis.status
    ]
    column_formatters = {
        Analysis.mode: lambda m, a: f"<span class='badge badge-primary'>{m.mode}</span>",
        Analysis.status: lambda m, a: f"<span class='badge badge-{_get_analysis_status_badge(m.status)}'>{m.status}</span>"
    }
    
    can_create = False
    can_delete = False
    can_edit = False
    
    column_details_list = [
        Analysis.id,
        Analysis.document_id,
        Analysis.analysis_type_id,
        Analysis.mode,
        Analysis.status,
        Analysis.error_message,
        Analysis.created_at,
        Analysis.updated_at,
        Analysis.completed_at
    ]
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

class AnalysisStepResultAdmin(ModelView, model=AnalysisStepResult):
    """Admin interface for AnalysisStepResult model."""
    name = "Analysis Step Result"
    name_plural = "Analysis Step Results"
    icon = "fa-solid fa-clipboard-check"
    column_list = [
        AnalysisStepResult.id,
        AnalysisStepResult.analysis_id,
        AnalysisStepResult.step_id,
        AnalysisStepResult.algorithm_id,
        AnalysisStepResult.status,
        AnalysisStepResult.created_at,
        AnalysisStepResult.completed_at
    ]
    column_searchable_list = [
        AnalysisStepResult.analysis_id,
        AnalysisStepResult.step_id,
        AnalysisStepResult.algorithm_id,
        AnalysisStepResult.status
    ]
    column_sortable_list = [
        AnalysisStepResult.created_at,
        AnalysisStepResult.updated_at,
        AnalysisStepResult.completed_at,
        AnalysisStepResult.status
    ]
    column_formatters = {
        AnalysisStepResult.status: lambda m, a: f"<span class='badge badge-{_get_analysis_status_badge(m.status)}'>{m.status}</span>"
    }
    
    can_create = False
    can_delete = False
    can_edit = False
    
    column_details_list = [
        AnalysisStepResult.id,
        AnalysisStepResult.analysis_id,
        AnalysisStepResult.step_id,
        AnalysisStepResult.algorithm_id,
        AnalysisStepResult.status,
        AnalysisStepResult.parameters,
        AnalysisStepResult.result,
        AnalysisStepResult.user_corrections,
        AnalysisStepResult.error_message,
        AnalysisStepResult.created_at,
        AnalysisStepResult.updated_at,
        AnalysisStepResult.completed_at
    ]
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS

def _get_analysis_status_badge(status: str) -> str:
    """Get appropriate Bootstrap badge class for analysis status."""
    return {
        "pending": "secondary",
        "in_progress": "info",
        "completed": "success",
        "failed": "danger",
        "cancelled": "warning",
        "error": "danger"
    }.get(status.lower(), "secondary")

def setup_admin(app: FastAPI, engine) -> None:
    """Setup admin interface."""
    try:

        authentication_backend = AdminAuth()
        admin = Admin(
            app,
            engine,
            base_url="/admin",
            title="DIA Admin",
            authentication_backend=authentication_backend,
            logo_url="/static/logo.png",
        )

        admin.add_view(UserAdmin)
        admin.add_view(DocumentAdmin)
        admin.add_view(TagAdmin)
        admin.add_view(AnalysisTypeAdmin)
        admin.add_view(AnalysisStepAdmin)
        admin.add_view(AlgorithmAdmin) 
        admin.add_view(AnalysisAdmin)
        admin.add_view(AnalysisStepResultAdmin)
        admin.add_view(BlacklistedTokenAdmin)
    except Exception as e:
        logger.error(f"Error setting up admin: {str(e)}")
        raise e
