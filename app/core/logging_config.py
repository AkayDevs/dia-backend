import logging
import logging.handlers
import os
from pathlib import Path
from app.core.config import settings

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
ACCESS_LOG = LOGS_DIR / "access.log"
ERROR_LOG = LOGS_DIR / "error.log"
INFO_LOG = LOGS_DIR / "info.log"

# Custom formatter with more details
class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.request_id = getattr(record, 'request_id', '-')
        record.user_id = getattr(record, 'user_id', '-')
        return super().format(record)

def setup_logging(env_mode: str = "development"):
    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "()": CustomFormatter,
                "format": "%(asctime)s | %(levelname)s | [%(name)s] | req_id=%(request_id)s | user=%(user_id)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s | %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "INFO",
            },
            "info_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": INFO_LOG,
                "formatter": "verbose",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "INFO",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": ERROR_LOG,
                "formatter": "verbose",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "ERROR",
            },
            "access_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": ACCESS_LOG,
                "formatter": "verbose",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "INFO",
            },
        },
        "loggers": {
            "app": {  # Root logger for your application
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.api": {  # Logger for all API routes
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.api.v1.routes": {  # Specific logger for route modules
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.crud": {  # Logger for CRUD operations
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.db.models": {  # Logger for database models
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.schemas": {  # Logger for schemas
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.services": {  # Logger for services
                "handlers": ["console", "info_file", "error_file"],
                "level": "INFO",
                "propagate": False,
            },
            "app.access": {  # Logger for access logs
                "handlers": ["access_file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console", "info_file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    # Environment-specific configurations
    if env_mode == "development":
        # More verbose logging in development
        config["handlers"]["console"]["formatter"] = "verbose"
        config["handlers"]["console"]["level"] = "DEBUG"
        config["loggers"]["app"]["level"] = "DEBUG"
    else:
        # Production settings
        config["handlers"]["console"]["level"] = "WARNING"
        
    # Apply configuration
    logging.config.dictConfig(config)

# Context manager for request logging
class RequestLogContext:
    def __init__(self, request_id: str, user_id: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.old_context = {}

    def __enter__(self):
        # Store the current context
        frame = logging.currentframe()
        while frame:
            if (logger_name := frame.f_locals.get('self', None).__class__.__module__) != 'logging':
                break
            frame = frame.f_back
        
        logger = logging.getLogger(logger_name)
        self.old_context = {
            'request_id': getattr(logger, 'request_id', None),
            'user_id': getattr(logger, 'user_id', None)
        }
        
        # Set new context
        logger.request_id = self.request_id
        logger.user_id = self.user_id
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        frame = logging.currentframe()
        while frame:
            if (logger_name := frame.f_locals.get('self', None).__class__.__module__) != 'logging':
                break
            frame = frame.f_back
        
        logger = logging.getLogger(logger_name)
        # Restore the old context
        logger.request_id = self.old_context.get('request_id')
        logger.user_id = self.old_context.get('user_id') 