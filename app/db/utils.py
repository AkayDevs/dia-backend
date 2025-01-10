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
    """Get detailed information about a table.
    
    Args:
        inspector: SQLAlchemy inspector instance
        table_name: Name of the table to inspect
        
    Returns:
        Dict containing columns, indexes, and foreign keys information
    """
    try:
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        primary_keys = inspector.get_pk_constraint(table_name)
        
        return {
            "columns": columns,
            "indexes": indexes,
            "foreign_keys": foreign_keys,
            "primary_keys": primary_keys,
            "exists": True
        }
    except Exception as e:
        logger.error(f"Error getting table info for {table_name}: {e}")
        return {
            "exists": False,
            "error": str(e)
        }


def verify_database_schema(db: Session) -> Dict[str, bool]:
    """Verify that all tables and indexes are properly created.
    
    Args:
        db: SQLAlchemy session
        
    Returns:
        Dict containing validation results for each component
    """
    try:
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        validation_results = {
            "tables_exist": True,
            "indexes_valid": True,
            "foreign_keys_valid": True,
            "constraints_valid": True
        }
        
        # Required tables and their expected indexes
        schema_requirements = {
            'users': {
                'indexes': {
                    'ix_users_email',
                    'ix_users_id',
                    'ix_users_password_reset',
                    'ix_users_role_active',
                    'ix_users_verification'
                }
            },
            'documents': {
                'indexes': {
                    'ix_documents_id',
                    'ix_documents_name',
                    'ix_documents_type',
                    'ix_documents_user_id_uploaded_at',
                    'ix_documents_file_hash',
                    'ix_documents_user_id_file_hash'
                }
            },
            'analysis_results': {
                'indexes': {
                    'ix_analysis_results_id',
                    'ix_analysis_results_type',
                    'ix_analysis_results_document_id_type',
                    'ix_analysis_results_created_at'
                }
            }
        }
        
        # Check tables existence
        missing_tables = set(schema_requirements.keys()) - set(tables)
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            validation_results["tables_exist"] = False
        
        # Verify each table's structure
        for table_name, requirements in schema_requirements.items():
            if table_name not in tables:
                continue
                
            table_info = get_table_info(inspector, table_name)
            if not table_info["exists"]:
                validation_results["tables_exist"] = False
                continue
            
            # Check indexes
            existing_indexes = {idx['name'] for idx in table_info['indexes']}
            missing_indexes = requirements['indexes'] - existing_indexes
            if missing_indexes:
                logger.error(f"Missing indexes for {table_name}: {missing_indexes}")
                validation_results["indexes_valid"] = False
            
            # Verify foreign key constraints
            for fk in table_info['foreign_keys']:
                try:
                    # Test if the referenced table exists
                    db.execute(text(f"SELECT 1 FROM {fk['referred_table']} LIMIT 1"))
                except Exception as e:
                    logger.error(f"Invalid foreign key in {table_name}: {e}")
                    validation_results["foreign_keys_valid"] = False
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error verifying database schema: {e}")
        return {
            "tables_exist": False,
            "indexes_valid": False,
            "foreign_keys_valid": False,
            "constraints_valid": False
        }


def check_database_health(db: Session) -> Dict[str, bool]:
    """Check database health and connectivity.
    
    Args:
        db: SQLAlchemy session
        
    Returns:
        Dict containing health check results
    """
    health_status = {
        "connection": False,
        "write_permission": False,
        "schema_valid": False,
        "triggers_valid": False
    }
    
    try:
        # Test connection
        db.execute(text("SELECT 1"))
        health_status["connection"] = True
        
        # Test write permission
        try:
            db.execute(text("CREATE TEMPORARY TABLE _health_check (id int)"))
            db.execute(text("DROP TABLE _health_check"))
            health_status["write_permission"] = True
        except Exception as e:
            logger.error(f"Write permission test failed: {e}")
        
        # Verify schema
        schema_validation = verify_database_schema(db)
        health_status["schema_valid"] = all(schema_validation.values())
        
        # Check triggers
        try:
            triggers = db.execute(text("SELECT name FROM sqlite_master WHERE type='trigger'")).fetchall()
            required_triggers = {'update_documents_updated_at'}
            existing_triggers = {trigger[0] for trigger in triggers}
            health_status["triggers_valid"] = required_triggers.issubset(existing_triggers)
        except Exception as e:
            logger.error(f"Trigger validation failed: {e}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return health_status


def check_database_state(db: Session) -> Dict[str, any]:
    """Check and return the current state of the database.
    
    Args:
        db: SQLAlchemy session
        
    Returns:
        Dict containing database state information
    """
    state = {
        "tables": [],
        "row_counts": {},
        "health_status": {},
        "schema_validation": {},
        "storage_info": {}
    }
    
    try:
        # Get tables
        inspector = inspect(db.bind)
        state["tables"] = inspector.get_table_names()
        logger.info(f"Tables in database: {state['tables']}")
        
        # Get row counts
        for table in state["tables"]:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                state["row_counts"][table] = count
            except Exception as e:
                logger.error(f"Error getting row count for {table}: {e}")
                state["row_counts"][table] = -1
        
        # Get health status
        state["health_status"] = check_database_health(db)
        
        # Get schema validation
        state["schema_validation"] = verify_database_schema(db)
        
        # Get storage information (SQLite specific)
        try:
            page_count = db.execute(text("PRAGMA page_count")).scalar()
            page_size = db.execute(text("PRAGMA page_size")).scalar()
            state["storage_info"] = {
                "page_count": page_count,
                "page_size": page_size,
                "total_size_bytes": page_count * page_size if page_count and page_size else None
            }
        except Exception as e:
            logger.error(f"Error getting storage information: {e}")
        
        return state
            
    except Exception as e:
        logger.error(f"Error checking database state: {e}")
        return state


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