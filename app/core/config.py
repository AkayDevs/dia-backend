from typing import Optional, List, Union, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import EmailStr, validator, AnyHttpUrl, Field
import secrets
from datetime import timedelta


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Intelligence Analysis"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for Document Intelligence Analysis"
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    ALGORITHM: str = "HS256"
    
    # Database
    SQLITE_URL: str = "sqlite:///./dia.db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False  # Set to True for SQL query logging
    
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
    ALLOWED_MIME_TYPES: Dict[str, str] = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg"
    }
    
    # Analysis Settings
    MODEL_DIR: str = "app/ml/models"
    MAX_ANALYSIS_TIME: int = 300  # Maximum time in seconds for analysis
    CONCURRENT_ANALYSIS_LIMIT: int = 5  # Maximum concurrent analysis tasks
    ANALYSIS_QUEUE_TIMEOUT: int = 600  # Timeout for queued analysis tasks
    
    # Admin Settings
    ADMIN_BASE_URL: str = "/admin"
    ADMIN_PAGE_SIZE: int = 25
    ADMIN_PAGE_SIZE_OPTIONS: List[int] = [25, 50, 100]
    
    # Admin User
    FIRST_SUPERUSER: EmailStr = "admin@example.com"
    FIRST_SUPERUSER_PASSWORD: str = "admin123"
    FIRST_SUPERUSER_NAME: str = "Admin"

    # Email Settings
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None
    EMAIL_TEMPLATES_DIR: str = "app/email-templates"
    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 24

    # Rate Limiting
    RATE_LIMIT_PER_USER: int = 1000  # requests per day
    RATE_LIMIT_BURST: int = 100  # maximum burst size
    RATE_LIMIT_WINDOW: int = 3600  # time window in seconds
    
    # Security Headers
    SECURITY_BCRYPT_ROUNDS: int = 12
    SECURITY_PASSWORD_SALT: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    SECURITY_PASSWORD_HASH: str = "bcrypt"
    SECURITY_PASSWORD_LENGTH_MIN: int = 8
    SECURITY_PASSWORD_LENGTH_MAX: int = 50
    
    # Session
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    SESSION_COOKIE_NAME: str = "dia_session"
    SESSION_COOKIE_EXPIRE: int = 60 * 60 * 24 * 7  # 7 days
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = "logs/dia.log"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "1 month"

    # Cache
    CACHE_TYPE: str = "simple"  # Options: simple, redis
    CACHE_REDIS_URL: Optional[str] = None
    CACHE_DEFAULT_TIMEOUT: int = 300
    CACHE_KEY_PREFIX: str = "dia_cache:"

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == "ALLOWED_EXTENSIONS":
                return {ext.strip() for ext in raw_val.split(",")}
            return raw_val


settings = Settings()
