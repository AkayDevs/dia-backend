from sqlalchemy import String, DateTime, Enum, Boolean, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import TYPE_CHECKING, List

from app.db.base_class import Base
from app.schemas.user import UserRole

if TYPE_CHECKING:
    from app.db.models.document import Document


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    avatar: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Additional security fields
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verification_token: Mapped[str | None] = mapped_column(String, nullable=True)
    password_reset_token: Mapped[str | None] = mapped_column(String, nullable=True)
    password_reset_expires: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships

    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    # Optimized indexes
    __table_args__ = (
        # Index for email verification
        Index('ix_users_verification', verification_token, is_verified),
        # Index for password reset
        Index('ix_users_password_reset', password_reset_token, password_reset_expires),
        # Index for role-based queries
        Index('ix_users_role_active', role, is_active),
    ) 