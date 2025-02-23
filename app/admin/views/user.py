from typing import Any
from app.admin.base import BaseModelView
from app.db.models.user import User, UserRole
from app.core.security import get_password_hash
import logging

logger = logging.getLogger(__name__)

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
    
    column_labels = {
        User.id: "ID",
        User.email: "Email",
        User.name: "Full Name",
        User.role: "Role",
        User.is_active: "Active",
        User.is_verified: "Verified",
        User.created_at: "Created",
        User.updated_at: "Updated"
    }
    
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
        User.role: lambda m, a: f"<span class='badge badge-{'primary' if m.role == UserRole.ADMIN else 'secondary'}'>{m.role.value}</span>",
        User.created_at: lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M"),
        User.updated_at: lambda m, a: m.updated_at.strftime("%Y-%m-%d %H:%M")
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
    
    form_widget_args = {
        "email": {
            "placeholder": "user@example.com"
        },
        "name": {
            "placeholder": "Full Name"
        },
        "password": {
            "placeholder": "Leave empty to keep current password"
        }
    }
    
    # Make certain fields read-only in edit mode
    form_edit_rules = [
        "email",  # Email can't be changed once set
        "name",
        "role",
        "is_active",
        "is_verified",
        "password"
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
    can_edit = True  # Allow editing user details
    can_export = True  # Allow exporting user data
    can_view_details = True  # Enable detailed view
    
    # List configuration
    page_size = 25  # Show 25 users per page
    can_set_page_size = True  # Allow admin to change items per page
    
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
        logger.info(f"[UserAdmin] {message}") 