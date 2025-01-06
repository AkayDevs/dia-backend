from typing import Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import deps
from app.db.utils import check_database_health, get_table_stats

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(deps.get_db)) -> Dict:
    """
    Check system health including database connectivity and schema validation.
    """
    db_health = check_database_health(db)
    db_stats = get_table_stats(db)
    
    return {
        "status": "healthy" if all(db_health.values()) else "unhealthy",
        "database": {
            "health": db_health,
            "stats": db_stats
        }
    } 