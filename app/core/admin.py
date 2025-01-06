from sqladmin import ModelView, Admin, BaseView, expose
from sqladmin.authentication import AuthenticationBackend
from app.db.models.document import Document, AnalysisResult
from app.db.models.user import User, UserRole
from app.core.auth import get_current_user, get_current_admin
from app.core.config import settings
from fastapi import Depends, Request, HTTPException
from typing import Optional
from starlette.responses import RedirectResponse
from app.db.session import get_db
import jwt as PyJWT


class AdminAuth(AuthenticationBackend):
    def __init__(self):
        super().__init__(secret_key=settings.SECRET_KEY)
        
    async def login(self, request: Request) -> bool:
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
            db = next(get_db())
            user = crud_user.authenticate(
                db, 
                email=form["username"], 
                password=form["password"]
            )
            if user and user.is_active and user.role == UserRole.ADMIN:
                request.session.update({"admin_authenticated": True})
                return True
            return False
            
        if not token:
            return False
            
        try:
            # Verify token and get user
            payload = PyJWT.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            db = next(get_db())
            user = db.query(User).filter(User.id == payload.get("sub")).first()
            
            if not user or not user.is_active or user.role != UserRole.ADMIN:
                return False
                
            # Set session data
            request.session.update({"admin_authenticated": True})
            return True
        except (PyJWT.JWTError, Exception):
            return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> Optional[bool]:
        if request.session.get("admin_authenticated"):
            return True
            
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return RedirectResponse(
                request.url_for("admin:login"),
                status_code=302
            )
            
        try:
            token = auth_header.split(" ")[1]
            payload = PyJWT.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            db = next(get_db())
            user = db.query(User).filter(User.id == payload.get("sub")).first()
            
            if not user or not user.is_active or user.role != UserRole.ADMIN:
                return RedirectResponse(
                    request.url_for("admin:login"),
                    status_code=302
                )
                
            request.session.update({"admin_authenticated": True})
            return True
        except (PyJWT.JWTError, Exception):
            return RedirectResponse(
                request.url_for("admin:login"),
                status_code=302
            )


class UserAdmin(ModelView, model=User):
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-users"
    column_list = [User.id, User.email, User.name, User.role, User.is_active, User.is_verified, User.created_at]
    column_searchable_list = [User.email, User.name]
    column_sortable_list = [User.email, User.name, User.role, User.is_active, User.created_at]
    column_formatters = {
        User.is_active: lambda m, a: "✓" if m.is_active else "✗",
        User.is_verified: lambda m, a: "✓" if m.is_verified else "✗",
        User.role: lambda m, a: f"<span class='badge badge-{'primary' if m.role == UserRole.ADMIN else 'secondary'}'>{m.role.value}</span>"
    }
    
    # Allow editing but not creation or deletion
    can_create = False
    can_delete = False
    can_edit = True
    
    # Specify only the fields that can be edited
    form_columns = [
        User.email,
        User.name,
        User.role,
        User.is_active,
        User.is_verified,
    ]
    
    # Add field descriptions
    column_descriptions = {
        User.email: "User's email address",
        User.name: "User's full name",
        User.role: "User's role (admin or user)",
        User.is_active: "Whether the user account is active",
        User.is_verified: "Whether the user's email is verified",
    }
    
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS


class DocumentAdmin(ModelView, model=Document):
    name = "Document"
    name_plural = "Documents"
    icon = "fa-solid fa-file"
    column_list = [Document.id, Document.name, Document.type, Document.status, Document.uploaded_at, Document.user_id]
    column_searchable_list = [Document.name]
    column_sortable_list = [Document.uploaded_at, Document.name, Document.status]
    column_formatters = {
        Document.status: lambda m, a: f"<span class='badge badge-{m.status.lower()}'>{m.status}</span>",
        Document.type: lambda m, a: f"<span class='badge badge-info'>{m.type}</span>"
    }
    can_create = False
    can_delete = False
    can_edit = False
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS


class AnalysisResultAdmin(ModelView, model=AnalysisResult):
    name = "Analysis Result"
    name_plural = "Analysis Results"
    icon = "fa-solid fa-chart-simple"
    column_list = [AnalysisResult.id, AnalysisResult.document_id, AnalysisResult.type, AnalysisResult.created_at]
    column_sortable_list = [AnalysisResult.created_at, AnalysisResult.type]
    column_formatters = {
        AnalysisResult.type: lambda m, a: f"<span class='badge badge-info'>{m.type}</span>"
    }
    can_create = False
    can_delete = False
    can_edit = False
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS


def setup_admin(app, engine):
    authentication_backend = AdminAuth()
    admin = Admin(
        app,
        engine,
        authentication_backend=authentication_backend,
        title=f"{settings.PROJECT_NAME} Admin",
        base_url=f"{settings.API_V1_STR}/admin",
        logo_url=None  # You can add a logo URL here if needed
    )
    
    # Add views
    admin.add_view(UserAdmin)
    admin.add_view(DocumentAdmin)
    admin.add_view(AnalysisResultAdmin)
    
    return admin 