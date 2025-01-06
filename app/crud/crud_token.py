from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.models.token import BlacklistedToken
from app.schemas.token import BlacklistedToken as BlacklistedTokenSchema

class CRUDToken:
    def is_blacklisted(self, db: Session, token: str) -> bool:
        """Check if a token is blacklisted and not expired."""
        token_record = db.query(BlacklistedToken).filter(
            BlacklistedToken.token == token,
            BlacklistedToken.expires_at > datetime.utcnow()
        ).first()
        return token_record is not None

    def blacklist_token(
        self, 
        db: Session, 
        token: str, 
        expires_at: datetime
    ) -> BlacklistedToken:
        """Blacklist a token."""
        db_token = BlacklistedToken(
            token=token,
            expires_at=expires_at
        )
        db.add(db_token)
        db.commit()
        db.refresh(db_token)
        return db_token

    def cleanup_expired_tokens(self, db: Session) -> int:
        """Remove expired tokens from the blacklist."""
        result = db.query(BlacklistedToken).filter(
            BlacklistedToken.expires_at <= datetime.utcnow()
        ).delete()
        db.commit()
        return result

token = CRUDToken() 