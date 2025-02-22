from fastapi import FastAPI
from sqladmin import Admin
import logging

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
        # Initialize authentication backend
        authentication_backend = AdminAuth()
        
        # Create admin instance
        admin = Admin(
            app,
            engine,
            base_url="/admin",
            title=settings.ADMIN_TITLE,
            authentication_backend=authentication_backend,
            logo_url=settings.ADMIN_LOGO_URL,
        )
        
        # Register views in logical order
        
        # User management
        admin.add_view(UserAdmin)
        
        # Document management
        admin.add_view(DocumentAdmin)
        admin.add_view(TagAdmin)
        
        # Analysis configuration
        admin.add_view(AnalysisDefinitionAdmin)
        admin.add_view(StepDefinitionAdmin)
        admin.add_view(AlgorithmDefinitionAdmin)
        
        # Analysis execution
        admin.add_view(AnalysisRunAdmin)
        admin.add_view(StepExecutionResultAdmin)
        
        logger.info("Admin interface setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up admin interface: {str(e)}")
        raise 