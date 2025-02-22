from typing import Any
from app.admin.base import BaseModelView
from app.db.models.user import User, UserRole
from app.core.security import get_password_hash

class UserAdmin(BaseModelView, model=User):
    """Enhanced admin interface for User model."""
    
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-users"
    
    # List view configuration
    column_list = [
        User.id,
        User.email,
        User.name,
        User.role,
        User.is_active,
        User.is_verified,
        User.created_at,
        User.updated_at
    ]
    
    column_searchable_list = [
        User.email,
        User.name
    ]
    
    column_sortable_list = [
        User.email,
        User.name,
        User.role,
        User.is_active,
        User.created_at,
        User.updated_at
    ]
    
    column_formatters = {
        User.is_active: lambda m, a: "✓" if m.is_active else "✗",
        User.is_verified: lambda m, a: "✓" if m.is_verified else "✗",
        User.role: lambda m, a: f"<span class='badge badge-{'primary' if m.role == UserRole.ADMIN else 'secondary'}'>{m.role.value}</span>"
    }
    
    # Form configuration
    form_columns = [
        User.email,
        User.name,
        User.role,
        User.is_active,
        User.is_verified,
        "password"  # For password changes
    ]
    
    form_excluded_columns = [
        User.hashed_password,
        User.documents
    ]
    
    # Column descriptions for better UX
    column_descriptions = {
        User.email: "User's email address (must be unique)",
        User.name: "User's full name",
        User.role: "User's role (admin or regular user)",
        User.is_active: "Whether the user account is active",
        User.is_verified: "Whether the user's email is verified",
        User.created_at: "Account creation date",
        User.updated_at: "Last update date",
        "password": "Set a new password (leave empty to keep current)"
    }
    
    # Security settings
    can_create = True  # Allow creating new users
    can_delete = False  # Prevent user deletion for audit purposes
    can_edit = True    # Allow editing user details
    can_export = True  # Allow exporting user data
    
    def on_model_change(self, form: Any, model: User, is_created: bool) -> None:
        """Handle user model changes with password hashing."""
        if hasattr(form, "password") and form.password.data:
            model.hashed_password = get_password_hash(form.password.data)
        
        # Ensure at least one admin exists
        if not is_created and model.role != UserRole.ADMIN:
            admin_count = self.session.query(User).filter(
                User.role == UserRole.ADMIN,
                User.is_active == True,
                User.id != model.id
            ).count()
            
            if admin_count == 0:
                raise ValueError("Cannot change the last admin user's role")
        
        super().on_model_change(form, model, is_created)
    
    def after_model_change(self, form: Any, model: User, is_created: bool) -> None:
        """Log user changes for audit purposes."""
        action = "created" if is_created else "updated"
        self.log_action(f"User {model.email} was {action}")
        super().after_model_change(form, model, is_created)
    
    def log_action(self, message: str) -> None:
        """Log admin actions for audit trail."""
        logger = self.logger
        logger.info(f"[UserAdmin] {message}") 