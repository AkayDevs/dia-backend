import logging
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys
from typing import Dict, List

# Add the app directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.db.base import Base
from app.core.config import settings
from app.core.logging_conf import setup_logging, LOGS_DIR

# Initialize logging using project's configuration
setup_logging()

# Initialize logger
logger = logging.getLogger("app.db.migrations")

def get_url() -> str:
    """Get the SQLAlchemy URL from settings."""
    return settings.SQLITE_URL

def include_object(object, name: str, type_: str, reflected: bool, compare_to) -> bool:
    """Define which database objects to include in migrations.
    
    Args:
        object: The database object
        name: Name of the object
        type_: Type of the object (table, index, etc.)
        reflected: Whether the object was reflected
        compare_to: The object being compared to
        
    Returns:
        bool: Whether to include the object in the migration
    """
    # Skip SQLite internal tables
    if name.startswith('sqlite_'):
        return False
        
    # Skip alembic version table
    if type_ == "table" and name == "alembic_version":
        return False
        
    return True

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well.
    By skipping the Engine creation we don't even need a DBAPI to be available.
    """
    logger.info("Running migrations offline")
    context.configure(
        url=get_url(),
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,  # Compare column types
        compare_server_default=True,  # Compare default values
        render_as_batch=True,  # Enable batch mode for SQLite
        user_module_prefix=None
    )

    with context.begin_transaction():
        context.run_migrations()
    logger.info("Offline migrations completed")

def process_revision_directives(context, revision, directives) -> None:
    """Allow customizing revision generation.
    
    Args:
        context: Migration context
        revision: Revision being processed
        directives: Migration directives
    """
    # Skip empty migrations
    if config.cmd_opts and config.cmd_opts.autogenerate:
        script = directives[0]
        if script.upgrade_ops.is_empty():
            logger.info("No changes detected; skipping migration generation")
            directives[:] = []

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    logger.info("Starting online migrations")
    
    # Configure SQLAlchemy URL
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    
    # Configure the engine
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"check_same_thread": False}  # Required for SQLite
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=True,
            process_revision_directives=process_revision_directives,
            user_module_prefix=None
        )

        try:
            logger.info("Executing migration transaction")
            with context.begin_transaction():
                context.run_migrations()
            logger.info("Migration transaction completed successfully")
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}", exc_info=True)
            raise

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
