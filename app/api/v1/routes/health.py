from typing import Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import deps
from app.db.utils import verify_database_schema

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(deps.get_db)) -> Dict:
    """
    Check system health including database connectivity and schema validation.
    """
    try:
        # Test database connectivity
        db.execute(text("SELECT 1"))
        connection_ok = True
    except Exception as e:
        connection_ok = False
    
    # Verify database schema
    schema_ok = verify_database_schema(db)
    
    health_status = {
        "connection": connection_ok,
        "schema_valid": schema_ok
    }
    
    return {
        "status": "healthy" if all(health_status.values()) else "unhealthy",
        "database": {
            "health": health_status,
            "details": {
                "message": "All systems operational" if all(health_status.values()) 
                else "Issues detected with database health"
            }
        }
    } 