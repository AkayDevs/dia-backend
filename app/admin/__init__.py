"""
Admin module for the DIA backend application.

This module provides a comprehensive admin interface with:
- Secure authentication and authorization
- Role-based access control
- Audit logging
- Model management views
- Export functionality
- Search and filtering capabilities

The admin interface is organized into several sections:
1. User Management
2. Document Management
3. Analysis Configuration
4. Analysis Execution Monitoring

Each section provides appropriate views and functionality for managing
the corresponding aspects of the application.
"""

from .setup import setup_admin
from .auth import AdminAuth
from .base import BaseModelView

__all__ = [
    "setup_admin",
    "AdminAuth",
    "BaseModelView"
] 