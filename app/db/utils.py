from typing import Dict, List
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
        
        expected_tables = ['users', 'documents', 'analysis_results']
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
            'ix_documents_status_type',
            'ix_documents_user_id_uploaded_at'
        }
        
        if not expected_doc_indexes.issubset(doc_indexes):
            logger.error(f"Missing document indexes: {expected_doc_indexes - doc_indexes}")
            return False
            
        # Verify analysis_results table
        analysis_info = get_table_info(inspector, 'analysis_results')
        analysis_indexes = {idx['name'] for idx in analysis_info['indexes']}
        expected_analysis_indexes = {
            'ix_analysis_results_id',
            'ix_analysis_results_type',
            'ix_analysis_results_document_id_type',
            'ix_analysis_results_created_at'
        }
        
        if not expected_analysis_indexes.issubset(analysis_indexes):
            logger.error(f"Missing analysis indexes: {expected_analysis_indexes - analysis_indexes}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying database schema: {e}")
        return False


def check_database_health(db: Session) -> Dict[str, bool]:
    """Check database health and connectivity."""
    try:
        # Test basic query
        db.execute(text("SELECT 1"))
        
        # Verify schema
        schema_valid = verify_database_schema(db)
        
        # Test write permission by creating a temporary record
        db.execute(text("CREATE TEMPORARY TABLE _health_check (id int)"))
        db.execute(text("DROP TABLE _health_check"))
        
        return {
            "connection": True,
            "schema_valid": schema_valid,
            "write_permission": True
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "connection": False,
            "schema_valid": False,
            "write_permission": False
        }


def get_table_stats(db: Session) -> Dict[str, int]:
    """Get basic statistics about the tables."""
    try:
        stats = {}
        for table in ['users', 'documents', 'analysis_results']:
            count = db.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()
            stats[table] = count
        return stats
    except Exception as e:
        logger.error(f"Error getting table stats: {e}")
        return {} 


def check_admin_user(db: Session) -> None:
    """Check and print admin user details."""
    user = db.query(User).filter(User.email == settings.FIRST_SUPERUSER).first()
    if user:
        logger.info(f"""
Admin user found:
- Email: {user.email}
- Role: {user.role}
- Active: {user.is_active}
- Verified: {user.is_verified}
""")
    else:
        logger.info("No admin user found in database")
    return user


def check_database_state(db: Session) -> None:
    """Check and print the current state of the database."""
    try:
        # Check if tables exist
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        logger.info(f"Tables in database: {tables}")
            
    except Exception as e:
        logger.error(f"Error checking database state: {e}")


def ensure_admin_exists(db: Session) -> bool:
    """Ensure that the admin user exists in the database."""
    logger.info(f"Checking for admin user with email: {settings.FIRST_SUPERUSER}")
    user = check_admin_user(db)
    
    if not user:
        logger.info("Admin user not found. Creating new admin user...")
        try:
            # Create admin user directly using the User model
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
            
            logger.info("Checking database state after admin creation:")
            check_database_state(db)
            return True
        except Exception as e:
            logger.error(f"Error creating admin user: {e}")
            db.rollback()
            raise
    else:
        logger.info(f"Admin user already exists with email: {settings.FIRST_SUPERUSER}")
        check_database_state(db)
        return False 