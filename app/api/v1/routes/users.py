from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from app.core.auth import get_current_user, get_current_admin
from app.crud.crud_user import user as crud_user
from app.db import deps
from app.schemas.user import User, UserUpdate, UserWithStats
from app.db.models.user import User as UserModel

router = APIRouter()
logger = logging.getLogger("app.api.users")


@router.get("/me", response_model=User)
async def read_user_me(
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get current user.
    """
    logger.debug(f"Fetching current user details - ID: {current_user.id}")
    return current_user


@router.get("/me/stats", response_model=UserWithStats)
async def read_user_stats(
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get current user's statistics.
    """
    logger.debug(f"Fetching statistics for user - ID: {current_user.id}")
    
    # Get document statistics
    total_documents = len(current_user.documents)
    documents_analyzed = sum(1 for doc in current_user.documents if doc.status == "ANALYZED")
    
    # Create response with stats
    user_dict = User.model_validate(current_user).model_dump()
    return {
        **user_dict,
        "total_documents": total_documents,
        "documents_analyzed": documents_analyzed,
    }


@router.put("/me", response_model=User)
async def update_user_me(
    *,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_user),
    user_in: UserUpdate,
) -> Any:
    """
    Update current user.
    """
    logger.debug(f"Updating user - ID: {current_user.id}")
    
    # If email is being updated, check it's not taken
    if user_in.email and user_in.email != current_user.email:
        if crud_user.get_by_email(db, email=user_in.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
    
    user = crud_user.update(db, db_obj=current_user, obj_in=user_in)
    logger.info(f"User updated successfully - ID: {user.id}")
    return user


@router.get("/", response_model=List[User])
async def read_users(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_admin),
) -> Any:
    """
    Retrieve users. Admin only.
    """
    logger.debug(f"Fetching users - Skip: {skip}, Limit: {limit}")
    users = db.query(UserModel).offset(skip).limit(limit).all()
    return users


@router.get("/{user_id}", response_model=User)
async def read_user_by_id(
    user_id: str,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_admin),
) -> Any:
    """
    Get a specific user by id. Admin only.
    """
    logger.debug(f"Fetching user details - ID: {user_id}")
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.put("/{user_id}", response_model=User)
async def update_user(
    *,
    db: Session = Depends(deps.get_db),
    user_id: str,
    user_in: UserUpdate,
    current_user: UserModel = Depends(get_current_admin),
) -> Any:
    """
    Update a user. Admin only.
    """
    logger.debug(f"Updating user - ID: {user_id}")
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # If email is being updated, check it's not taken
    if user_in.email and user_in.email != user.email:
        if crud_user.get_by_email(db, email=user_in.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
    
    user = crud_user.update(db, db_obj=user, obj_in=user_in)
    logger.info(f"User updated successfully - ID: {user.id}")
    return user 