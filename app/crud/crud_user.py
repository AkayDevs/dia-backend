import logging
from typing import Any, Dict, Optional, Union, List
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from app.core.security import get_password_hash, verify_password
from app.crud.base import CRUDBase
from app.db.models.user import User, UserRole
from app.schemas.user import UserCreate, UserUpdate

logger = logging.getLogger(__name__)


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """Get a user by email."""
        return db.query(User).filter(User.email == email).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        """Create a new user with password hashing."""
        db_obj = User(
            id=str(uuid.uuid4()),
            email=obj_in.email,
            hashed_password=get_password_hash(obj_in.password),
            name=obj_in.name,
            role=obj_in.role if hasattr(obj_in, 'role') else UserRole.USER,
            avatar=obj_in.avatar if hasattr(obj_in, 'avatar') else None,
            is_active=True,
            is_verified=False,  # Regular users need verification
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: User, obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        """Update user with password hashing if new password provided."""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
            
        update_data["updated_at"] = datetime.utcnow()
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, email: str, password: str) -> Optional[User]:
        """Authenticate user by email and password."""
        user = self.get_by_email(db, email=email)
        if not user:
            return None
            
        if not verify_password(password, user.hashed_password):
            return None
            
        return user

    def get_user_documents(
        self, db: Session, *, user_id: str, skip: int = 0, limit: int = 100
    ) -> List[Any]:
        """Get all documents for a user with pagination."""
        user = self.get(db, id=user_id)
        if not user:
            return []
        return user.documents[skip:skip + limit]

    def is_active(self, user: User) -> bool:
        """Check if user is active."""
        return user.is_active

    def is_verified(self, user: User) -> bool:
        """Check if user is verified."""
        return user.is_verified

    def is_admin(self, user: User) -> bool:
        """Check if user is admin."""
        return user.role == UserRole.ADMIN

    def mark_verified(self, db: Session, *, user: User) -> User:
        """Mark user as verified."""
        user.is_verified = True
        user.verification_token = None
        user.updated_at = datetime.utcnow()
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def set_password(self, db: Session, *, user: User, password: str) -> User:
        """Set new password for user."""
        user.hashed_password = get_password_hash(password)
        user.password_reset_token = None
        user.password_reset_expires = None
        user.updated_at = datetime.utcnow()
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


user = CRUDUser(User) 