from typing import Any, Dict, List, Optional
from sqladmin import ModelView
from app.core.config import settings

class BaseModelView(ModelView):
    """Base admin model view with common configurations."""
    
    # Pagination settings
    page_size = settings.ADMIN_PAGE_SIZE
    page_size_options = settings.ADMIN_PAGE_SIZE_OPTIONS
    
    # Common configurations
    can_create = False  # Default to False for safety
    can_delete = False  # Default to False for safety
    can_edit = False   # Default to False for safety
    can_view_details = True
    can_export = True
    
    # Audit fields that should typically be readonly
    readonly_fields = [
        "id",
        "created_at",
        "updated_at"
    ]
    
    # Common formatters
    column_formatters = {
        "created_at": lambda m, a: m.created_at.strftime("%Y-%m-%d %H:%M:%S") if m.created_at else "",
        "updated_at": lambda m, a: m.updated_at.strftime("%Y-%m-%d %H:%M:%S") if m.updated_at else "",
    }
    
    def is_accessible(self) -> bool:
        """Check if current user has access to this admin view."""
        # This will be handled by the authentication backend
        return True
    
    def is_visible(self) -> bool:
        """Check if this model should be visible in the menu."""
        return True
    
    def get_list_query(self):
        """Customize the list query if needed."""
        return self.session.query(self.model)
    
    def on_model_change(self, form: Any, model: Any, is_created: bool) -> None:
        """Handle model changes before save."""
        super().on_model_change(form, model, is_created)
    
    def after_model_change(self, form: Any, model: Any, is_created: bool) -> None:
        """Handle model changes after save."""
        super().after_model_change(form, model, is_created)
    
    def on_model_delete(self, model: Any) -> None:
        """Handle model deletion before delete."""
        super().on_model_delete(model)
    
    def after_model_delete(self, model: Any) -> None:
        """Handle model deletion after delete."""
        super().after_model_delete(model) 