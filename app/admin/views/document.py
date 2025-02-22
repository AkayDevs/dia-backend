from typing import Any
from app.admin.base import BaseModelView
from app.db.models.document import Document, Tag
from app.core.config import settings

class DocumentAdmin(BaseModelView, model=Document):
    """Enhanced admin interface for Document model."""
    
    name = "Document"
    name_plural = "Documents"
    icon = "fa-solid fa-file"
    
    # List view configuration
    column_list = [
        Document.id,
        Document.name,
        Document.type,
        Document.user_id,
        'tags',
        Document.size,
        Document.is_archived,
        Document.uploaded_at,
        Document.updated_at,
        Document.archived_at,
        Document.retention_until
    ]
    
    column_searchable_list = [
        Document.name,
        Document.user_id,
        Document.previous_version_id
    ]
    
    column_sortable_list = [
        Document.name,
        Document.uploaded_at,
        Document.updated_at,
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
    
    # Form configuration
    form_columns = [
        Document.name,
        Document.type,
        'tags',
        Document.is_archived,
        Document.retention_until
    ]
    
    form_excluded_columns = [
        Document.url,
        Document.hash,
        Document.analyses
    ]
    
    # Column descriptions for better UX
    column_descriptions = {
        Document.name: "Document name",
        Document.type: "Document type",
        Document.uploaded_at: "Upload date",
        Document.updated_at: "Last update date",
        Document.size: "File size in bytes",
        Document.user_id: "Owner user ID",
        'tags': "Document tags",
        Document.previous_version_id: "ID of the previous version",
        Document.is_archived: "Whether the document is archived",
        Document.archived_at: "When the document was archived",
        Document.retention_until: "Date until which the document must be retained"
    }
    
    # Security settings
    can_create = False  # Documents should be created through the API
    can_delete = False  # Documents should be archived, not deleted
    can_edit = True    # Allow editing metadata
    can_export = True  # Allow exporting document metadata
    
    def on_model_change(self, form: Any, model: Document, is_created: bool) -> None:
        """Handle document model changes."""
        # Update archive status
        if model.is_archived and not model.archived_at:
            from datetime import datetime
            model.archived_at = datetime.utcnow()
        elif not model.is_archived:
            model.archived_at = None
        
        super().on_model_change(form, model, is_created)
    
    def after_model_change(self, form: Any, model: Document, is_created: bool) -> None:
        """Log document changes for audit purposes."""
        action = "created" if is_created else "updated"
        self.log_action(f"Document {model.name} was {action}")
        super().after_model_change(form, model, is_created)
    
    def log_action(self, message: str) -> None:
        """Log admin actions for audit trail."""
        logger = self.logger
        logger.info(f"[DocumentAdmin] {message}")

class TagAdmin(BaseModelView, model=Tag):
    """Enhanced admin interface for Tag model."""
    
    name = "Tag"
    name_plural = "Tags"
    icon = "fa-solid fa-tag"
    
    # List view configuration
    column_list = [
        Tag.id,
        Tag.name,
        Tag.created_at,
        Tag.updated_at
    ]
    
    column_searchable_list = [Tag.name]
    
    column_sortable_list = [
        Tag.name,
        Tag.created_at,
        Tag.updated_at
    ]
    
    # Form configuration
    form_columns = [Tag.name]
    
    # Column descriptions
    column_descriptions = {
        Tag.name: "Tag name (must be unique)",
        Tag.created_at: "When the tag was created",
        Tag.updated_at: "Last update date"
    }
    
    # Security settings
    can_create = True
    can_delete = True
    can_edit = True
    can_export = True
    
    def on_model_change(self, form: Any, model: Tag, is_created: bool) -> None:
        """Handle tag model changes."""
        # Ensure tag name is unique
        if self.session.query(Tag).filter(
            Tag.name == model.name,
            Tag.id != model.id
        ).first():
            raise ValueError(f"Tag name '{model.name}' already exists")
        
        super().on_model_change(form, model, is_created)
    
    def after_model_change(self, form: Any, model: Tag, is_created: bool) -> None:
        """Log tag changes for audit purposes."""
        action = "created" if is_created else "updated"
        self.log_action(f"Tag {model.name} was {action}")
        super().after_model_change(form, model, is_created)
    
    def on_model_delete(self, model: Tag) -> None:
        """Log tag deletion."""
        self.log_action(f"Tag {model.name} was deleted")
        super().on_model_delete(model)
    
    def log_action(self, message: str) -> None:
        """Log admin actions for audit trail."""
        logger = self.logger
        logger.info(f"[TagAdmin] {message}") 