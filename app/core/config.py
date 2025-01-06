from typing import Optional, List, Union
from pydantic_settings import BaseSettings
from pydantic import EmailStr, validator, AnyHttpUrl
import secrets


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Intelligence Analysis"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # Database
    SQLITE_URL: str = "sqlite:///./dia.db"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Frontend development server
        "http://localhost:8000",  # Backend development server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "docx", "xlsx", "png", "jpg", "jpeg"}
    
    # ML Model Settings
    MODEL_DIR: str = "app/ml/models"
    
    # Admin Settings
    ADMIN_BASE_URL: str = "/admin"
    ADMIN_PAGE_SIZE: int = 25
    ADMIN_PAGE_SIZE_OPTIONS: List[int] = [25, 50, 100]
    
    # Admin User
    FIRST_SUPERUSER: EmailStr = "admin@example.com"
    FIRST_SUPERUSER_PASSWORD: str = "admin123"

    # Email Settings
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # Rate Limiting
    RATE_LIMIT_PER_USER: int = 1000  # requests per day
    
    # Security Headers
    SECURITY_BCRYPT_ROUNDS: int = 12
    SECURITY_PASSWORD_SALT: str = secrets.token_urlsafe(32)
    SECURITY_PASSWORD_HASH: str = "bcrypt"
    
    # Session
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
