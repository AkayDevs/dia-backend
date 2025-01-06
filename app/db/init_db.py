import logging
from sqlalchemy.orm import Session
import uuid

from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.db.session import Base, engine
from app.db.models.user import User, UserRole

logger = logging.getLogger(__name__)


def init_db(db: Session) -> None:
    """Initialize database with required tables and initial data."""
    logger.info("Creating initial database setup...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Check if we should create initial admin user
    user = db.query(User).filter(User.email == settings.FIRST_SUPERUSER).first()
    if not user:
        logger.info(f"Creating initial admin user with email: {settings.FIRST_SUPERUSER}")
        
        # Create admin user with explicit password
        password = settings.FIRST_SUPERUSER_PASSWORD
        password_hash = get_password_hash(password)
        
        user = User(
            id=str(uuid.uuid4()),
            email=settings.FIRST_SUPERUSER,
            name="Initial Admin",
            hashed_password=password_hash,
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
        )
        db.add(user)
        
        try:
            db.commit()
            db.refresh(user)
            
            logger.info("Admin user created successfully")
            logger.debug(f"""Admin user details:
- Email: {user.email}
- Role: {user.role}
- Active: {user.is_active}
- Verified: {user.is_verified}""")
            
        except Exception as e:
            logger.error(f"Error creating admin user: {e}")
            db.rollback()
            raise
    else:
        logger.debug("Admin user already exists") 