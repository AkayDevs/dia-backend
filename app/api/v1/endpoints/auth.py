from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db import models
from app.core import security
from app.schemas import auth as auth_schemas

router = APIRouter()

@router.post("/login", response_model=auth_schemas.Token)
async def login(
    auth_data: auth_schemas.Login,
    db: AsyncSession = Depends(get_db)
):
    """Login user and return access token"""
    user = await security.authenticate_user(auth_data.email, auth_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    access_token = security.create_access_token(user.id)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=auth_schemas.User)
async def register(
    user_data: auth_schemas.UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register new user"""
    # Check if user exists
    user = await security.get_user_by_email(user_data.email, db)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    user = await security.create_user(user_data, db)
    return user 