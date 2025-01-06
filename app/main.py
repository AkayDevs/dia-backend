from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.v1.routes import auth, health, documents, users, analysis
from app.core.admin import setup_admin
from app.db.session import engine, SessionLocal
from app.db.init_db import init_db
from app.db.utils import ensure_admin_exists, check_database_state, check_database_health
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import os
import logging

# Get the root logger
logger = logging.getLogger()

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Let uvicorn handle the logging configuration
# This ensures logging respects the --log-level argument

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Add trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize admin interface
admin = setup_admin(app, engine)

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Mount uploads directory for serving files
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(health.router, prefix=f"{settings.API_V1_STR}", tags=["health"])
app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["documents"]
)
app.include_router(
    users.router,
    prefix=f"{settings.API_V1_STR}/users",
    tags=["users"]
)
app.include_router(
    analysis.router,
    prefix=f"{settings.API_V1_STR}/analysis",
    tags=["analysis"]
)


@app.on_event("startup")
async def on_startup():
    db = SessionLocal()
    try:
        # Initialize database
        init_db(db)
        
        # Ensure admin user exists first
        ensure_admin_exists(db)
        
        # Verify database health
        health_status = check_database_health(db)
        if not all(health_status.values()):
            raise Exception("Database health check failed")
            
        # Check final database state
        check_database_state(db)
            
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
