from typing import Dict
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.engine import Inspector
import logging
import uuid
from app.core.config import settings
from app.core.security import get_password_hash
from app.db.models.user import User, UserRole

logger = logging.getLogger(__name__)


def get_table_info(inspector: Inspector, table_name: str) -> Dict:
    """Get detailed information about a table."""
    columns = inspector.get_columns(table_name)
    indexes = inspector.get_indexes(table_name)
    foreign_keys = inspector.get_foreign_keys(table_name)
    
    return {
        "columns": columns,
        "indexes": indexes,
        "foreign_keys": foreign_keys
    }


def verify_database_schema(db: Session) -> bool:
    """Verify that all tables and indexes are properly created."""
    try:
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'users', 'documents', 'analysis_definitions', 'step_definitions',
            'algorithm_definitions', 'analysis_runs', 'step_execution_results',
            'blacklisted_tokens', 'tags', 'document_tags'
        ]
        missing_tables = set(expected_tables) - set(tables)
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
            
        # Verify documents table
        doc_info = get_table_info(inspector, 'documents')
        doc_indexes = {idx['name'] for idx in doc_info['indexes']}
        expected_doc_indexes = {
            'ix_documents_id',
            'ix_documents_name',
            'ix_documents_user_id_uploaded_at',
            'ix_documents_type'
        }
        
        if not expected_doc_indexes.issubset(doc_indexes):
            logger.error(f"Missing document indexes: {expected_doc_indexes - doc_indexes}")
            return False
            
        # Verify analysis runs table
        analysis_info = get_table_info(inspector, 'analysis_runs')
        if not analysis_info:
            logger.error("Could not get analysis runs table info")
            return False

        # Verify step execution results table
        step_results_info = get_table_info(inspector, 'step_execution_results')
        if not step_results_info:
            logger.error("Could not get step execution results table info")
            return False

        # Verify users table
        users_info = get_table_info(inspector, 'users')
        users_indexes = {idx['name'] for idx in users_info['indexes']}
        expected_users_indexes = {
            'ix_users_id',
            'ix_users_email',
            'ix_users_verification',
            'ix_users_password_reset',
            'ix_users_role_active'
        }
        
        if not expected_users_indexes.issubset(users_indexes):
            logger.error(f"Missing users indexes: {expected_users_indexes - users_indexes}")
            return False

        # All checks passed
        return True
        
    except Exception as e:
        logger.error(f"Error verifying database schema: {e}")
        return False


def ensure_admin_exists(db: Session) -> bool:
    """Ensure that the admin user exists in the database."""
    try:
        user = db.query(User).filter(User.email == settings.FIRST_SUPERUSER).first()
        
        if not user:
            logger.info(f"Creating admin user with email: {settings.FIRST_SUPERUSER}")
            user = User(
                id=str(uuid.uuid4()),
                email=settings.FIRST_SUPERUSER,
                name="Initial Admin",
                hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                avatar=None,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("Admin user created successfully")
            return True
            
        logger.info(f"Admin user already exists with email: {settings.FIRST_SUPERUSER}")
        return False
        
    except Exception as e:
        logger.error(f"Error ensuring admin exists: {e}")
        db.rollback()
        raise 