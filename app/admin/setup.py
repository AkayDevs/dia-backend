from fastapi import FastAPI
from sqladmin import Admin
import logging
import traceback
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings
from app.admin.auth import AdminAuth
from app.admin.views.user import UserAdmin
from app.admin.views.document import DocumentAdmin, TagAdmin
from app.admin.views.analysis import (
    AnalysisDefinitionAdmin,
    StepDefinitionAdmin,
    AlgorithmDefinitionAdmin,
    AnalysisRunAdmin,
    StepExecutionResultAdmin
)

logger = logging.getLogger(__name__)

def setup_admin(app: FastAPI, engine) -> None:
    """
    Setup admin interface with all views and authentication.
    
    Features:
    - Secure authentication backend
    - Role-based access control
    - Audit logging
    - Customized views for all models
    - Export functionality
    - Search and filter capabilities
    """
    try:
        # Configure session middleware first
        app.add_middleware(
            SessionMiddleware,
            secret_key=settings.SECRET_KEY,
            session_cookie=settings.SESSION_COOKIE_NAME,
            max_age=settings.SESSION_COOKIE_EXPIRE,
            same_site=settings.SESSION_COOKIE_SAMESITE,
            https_only=settings.SESSION_COOKIE_SECURE
        )
        
        # Initialize authentication backend
        authentication_backend = AdminAuth()
        
        # Create admin instance
        admin = Admin(
            app,
            engine,
            authentication_backend=authentication_backend,
            base_url=settings.ADMIN_BASE_URL,
            title=settings.ADMIN_TITLE,
            logo_url=settings.ADMIN_LOGO_URL,
            templates_dir="app/admin/templates",  # Custom templates directory
        )
        
        # Register views
        admin.add_view(UserAdmin)
        admin.add_view(DocumentAdmin)
        admin.add_view(TagAdmin)
        admin.add_view(AnalysisDefinitionAdmin)
        admin.add_view(StepDefinitionAdmin)
        admin.add_view(AlgorithmDefinitionAdmin)
        admin.add_view(AnalysisRunAdmin)
        admin.add_view(StepExecutionResultAdmin)
        
        logger.info("Admin interface setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up admin interface: {str(e)}")
        logger.error(traceback.format_exc())
        raise 