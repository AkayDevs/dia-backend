import logging
import os
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger
from app.core.config import settings

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add module/function information
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add process/thread information
        log_record['process'] = record.process
        log_record['process_name'] = record.processName
        log_record['thread'] = record.thread
        log_record['thread_name'] = record.threadName

class RequestFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'request_id'):
            record.request_id = f'[{record.request_id}]'
        else:
            record.request_id = ''
        return super().format(record)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "()": RequestFormatter,
            "format": "%(asctime)s %(request_id)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": CustomJsonFormatter,
            "format": "%(timestamp)s %(level)s %(name)s %(message)s"
        },
        "access": {
            "format": "%(asctime)s [%(levelname)s] %(client_addr)s - '%(request_line)s' %(status_code)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "error": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\nException: %(exc_info)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "NOTSET",
            "stream": "ext://sys.stdout"
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "formatter": "error",
            "level": "ERROR",
            "stream": "ext://sys.stderr"
        },
        "info_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": os.path.join(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "level": "INFO"
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": os.path.join(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "level": "ERROR"
        },
        "access_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": os.path.join(LOGS_DIR, "access.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "level": "INFO"
        }
    },
    "loggers": {
        # Root logger
        "": {
            "level": "INFO",
            "handlers": ["console", "info_file", "error_file"],
            "propagate": False
        },
        # FastAPI logger
        "fastapi": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": False
        },
        # Uvicorn logger
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": False
        },
        # Uvicorn access logger
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console", "access_file"],
            "propagate": False
        },
        # SQLAlchemy logger
        "sqlalchemy.engine": {
            "level": "WARNING",
            "handlers": ["console", "error_file"],
            "propagate": False
        },
        # Application loggers
        "app": {
            "level": "INFO",
            "handlers": ["console", "info_file", "error_file"],
            "propagate": False
        },
        "app.api": {
            "level": "INFO",
            "handlers": ["console", "info_file", "error_file"],
            "propagate": False
        },
        "app.core": {
            "level": "INFO",
            "handlers": ["console", "info_file", "error_file"],
            "propagate": False
        },
        "app.db": {
            "level": "INFO",
            "handlers": ["console", "info_file", "error_file"],
            "propagate": False
        }
    }
}

def setup_logging():
    """Initialize the logging configuration"""
    # Set logging level based on environment
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    # Update root logger level
    LOGGING_CONFIG["loggers"][""]["level"] = log_level
    
    # Update app logger level
    LOGGING_CONFIG["loggers"]["app"]["level"] = log_level
    
    # Apply configuration
    dictConfig(LOGGING_CONFIG)
    
    # Create a startup log entry
    logging.getLogger("app").info(
        "Logging system initialized",
        extra={
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "log_level": log_level
        }
    )