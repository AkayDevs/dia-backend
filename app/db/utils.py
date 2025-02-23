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
            'users', 'documents', 'analysis_types', 'analysis_steps',
            'algorithms', 'analyses', 'analysis_step_results',
            'table_detections', 'table_structures', 'table_data'
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
            
        # Verify analysis tables
        analysis_info = get_table_info(inspector, 'analyses')
        analysis_indexes = {idx['name'] for idx in analysis_info['indexes']}
        expected_analysis_indexes = {
            'ix_analyses_document_id',
            'ix_analyses_analysis_type_id'
        }
        
        if not expected_analysis_indexes.issubset(analysis_indexes):
            logger.error(f"Missing analysis indexes: {expected_analysis_indexes - analysis_indexes}")
            return False

        # Verify table detection related tables
        table_detection_info = get_table_info(inspector, 'table_detections')
        table_detection_indexes = {idx['name'] for idx in table_detection_info['indexes']}
        expected_table_detection_indexes = {
            'ix_table_detections_document_id',
            'ix_table_detections_page_number'
        }
        
        if not expected_table_detection_indexes.issubset(table_detection_indexes):
            logger.error(f"Missing table detection indexes: {expected_table_detection_indexes - table_detection_indexes}")
            return False

        # Verify table structure related tables
        table_structure_info = get_table_info(inspector, 'table_structures')
        table_structure_indexes = {idx['name'] for idx in table_structure_info['indexes']}
        expected_table_structure_indexes = {
            'ix_table_structures_table_detection_id',
            'ix_table_structures_page_number'
        }
        
        if not expected_table_structure_indexes.issubset(table_structure_indexes):
            logger.error(f"Missing table structure indexes: {expected_table_structure_indexes - table_structure_indexes}")
            return False

        # Verify table data related tables
        table_data_info = get_table_info(inspector, 'table_data')
        table_data_indexes = {idx['name'] for idx in table_data_info['indexes']}
        expected_table_data_indexes = {
            'ix_table_data_table_structure_id',
            'ix_table_data_page_number'
        }
        
        if not expected_table_data_indexes.issubset(table_data_indexes):
            logger.error(f"Missing table data indexes: {expected_table_data_indexes - table_data_indexes}")
            return False
            
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