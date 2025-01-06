from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from app.core.auth import get_current_user, get_current_admin
from app.crud.crud_user import user as crud_user
from app.crud.crud_document import document as crud_document
from app.db import deps
from app.schemas.user import User, UserUpdate, UserWithStats
from app.db.models.user import User as UserModel
import logging
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger("app.api.users")


@router.get("/me", response_model=User)
async def read_user_me(
    request: Request,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get current user profile.
    
    Args:
        request: FastAPI request object
        current_user: Currently authenticated user
        
    Returns:
        Current user's profile data
        
    Raises:
        HTTPException: If user is not found or inactive
    """
    logger.info(f"Fetching profile for user: {current_user.id}")
    return current_user


@router.get("/me/stats", response_model=UserWithStats)
async def read_user_stats(
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get current user's statistics and activity data.
    
    Args:
        request: FastAPI request object
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        User profile with additional statistics
        
    Raises:
        HTTPException: If error occurs while fetching statistics
    """
    try:
        logger.info(f"Fetching statistics for user: {current_user.id}")
        
        # Get document statistics
        total_documents = crud_document.count_by_user(db, user_id=current_user.id)
        analyzed_documents = crud_document.count_analyzed_by_user(db, user_id=current_user.id)
        storage_used = crud_document.get_storage_used_by_user(db, user_id=current_user.id)
        
        # Create response with stats
        user_dict = User.model_validate(current_user).model_dump()
        return {
            **user_dict,
            "total_documents": total_documents,
            "documents_analyzed": analyzed_documents,
            "last_login": current_user.last_login,
            "storage_used": storage_used
        }
    except Exception as e:
        logger.error(f"Error fetching user stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user statistics"
        )


@router.put("/me", response_model=User)
async def update_user_me(
    request: Request,
    user_in: UserUpdate,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Update current user's profile.
    
    Args:
        request: FastAPI request object
        user_in: Updated user data
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Updated user profile
        
    Raises:
        HTTPException: If email is already taken or update fails
    """
    try:
        logger.info(f"Updating profile for user: {current_user.id}")
        
        # If email is being updated, check it's not taken
        if user_in.email and user_in.email != current_user.email:
            if crud_user.get_by_email(db, email=user_in.email):
                logger.warning(f"Email already taken: {user_in.email}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )
        
        # Validate password if provided
        if user_in.password:
            if len(user_in.password) < settings.SECURITY_PASSWORD_LENGTH_MIN:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Password must be at least {settings.SECURITY_PASSWORD_LENGTH_MIN} characters long"
                )
        
        user = crud_user.update(db, db_obj=current_user, obj_in=user_in)
        logger.info(f"User profile updated successfully: {user.id}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user profile"
        )


@router.get("/", response_model=List[User])
async def read_users(
    *,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, min_length=3, max_length=50),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    is_verified: Optional[bool] = Query(None),
) -> Any:
    """
    Retrieve users with filtering and pagination. Admin only.
    
    Args:
        request: FastAPI request object
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        search: Search term for email or name
        role: Filter by user role
        is_active: Filter by active status
        is_verified: Filter by verification status
        current_user: Currently authenticated admin user
        
    Returns:
        List of users matching the criteria
        
    Raises:
        HTTPException: If error occurs while fetching users
    """
    try:
        logger.info(f"Fetching users - Skip: {skip}, Limit: {limit}, Search: {search}")
        
        # Build query filters
        filters = {}
        if role:
            filters["role"] = role
        if is_active is not None:
            filters["is_active"] = is_active
        if is_verified is not None:
            filters["is_verified"] = is_verified
            
        users = crud_user.get_multi(
            db,
            skip=skip,
            limit=limit,
            search=search,
            filters=filters
        )
        
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching users"
        )


@router.get("/{user_id}", response_model=User)
async def read_user_by_id(
    request: Request,
    user_id: str,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_admin),
) -> Any:
    """
    Get a specific user by ID. Admin only.
    
    Args:
        request: FastAPI request object
        user_id: ID of the user to fetch
        db: Database session
        current_user: Currently authenticated admin user
        
    Returns:
        User profile data
        
    Raises:
        HTTPException: If user is not found
    """
    try:
        logger.info(f"Fetching user details - ID: {user_id}")
        
        user = crud_user.get(db, id=user_id)
        if not user:
            logger.warning(f"User not found - ID: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
            
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user details"
        )


@router.put("/{user_id}", response_model=User)
async def update_user(
    request: Request,
    user_id: str,
    user_in: UserUpdate,
    db: Session = Depends(deps.get_db),
    current_user: UserModel = Depends(get_current_admin),
) -> Any:
    """
    Update a user's profile. Admin only.
    
    Args:
        request: FastAPI request object
        user_id: ID of the user to update
        user_in: Updated user data
        db: Database session
        current_user: Currently authenticated admin user
        
    Returns:
        Updated user profile
        
    Raises:
        HTTPException: If user is not found or update fails
    """
    try:
        logger.info(f"Updating user - ID: {user_id}")
        
        # Get user
        user = crud_user.get(db, id=user_id)
        if not user:
            logger.warning(f"User not found - ID: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # If email is being updated, check it's not taken
        if user_in.email and user_in.email != user.email:
            if crud_user.get_by_email(db, email=user_in.email):
                logger.warning(f"Email already taken: {user_in.email}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )
        
        # Update user
        updated_user = crud_user.update(db, db_obj=user, obj_in=user_in)
        logger.info(f"User updated successfully - ID: {updated_user.id}")
        
        return updated_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user"
        ) 