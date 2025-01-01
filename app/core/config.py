from typing import Optional, List
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "DIA Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    SQLITE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./dia.db")
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]  # Add more origins as needed
    
    class Config:
        case_sensitive = True

settings = Settings() 