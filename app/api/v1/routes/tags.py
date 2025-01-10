from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import deps
from app.crud.crud_tag import tag as crud_tag
from app.schemas.tag import Tag, TagCreate, TagUpdate, DocumentTags
from app.db.models.user import User

router = APIRouter()


@router.post("", response_model=Tag)
async def create_tag(
    *,
    tag_in: TagCreate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Tag:
    """
    Create a new tag.
    
    Args:
        tag_in: Tag data
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Created tag
        
    Raises:
        HTTPException: If tag with same name already exists
    """
    existing_tag = crud_tag.get_by_name(db, name=tag_in.name)
    if existing_tag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tag with this name already exists"
        )
    return crud_tag.create(db=db, obj_in=tag_in)


@router.get("", response_model=List[Tag])
async def list_tags(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Tag]:
    """
    List all tags.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        List of tags
    """
    return crud_tag.get_multi(db, skip=skip, limit=limit)


@router.get("/{tag_id}", response_model=Tag)
async def get_tag(
    tag_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Tag:
    """
    Get a specific tag.
    
    Args:
        tag_id: Tag ID
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Tag details
        
    Raises:
        HTTPException: If tag is not found
    """
    tag = crud_tag.get(db=db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return tag


@router.put("/{tag_id}", response_model=Tag)
async def update_tag(
    *,
    tag_id: str,
    tag_in: TagUpdate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Tag:
    """
    Update a tag.
    
    Args:
        tag_id: Tag ID
        tag_in: Updated tag data
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Updated tag
        
    Raises:
        HTTPException: If tag is not found or new name already exists
    """
    tag = crud_tag.get(db=db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
        
    if tag_in.name and tag_in.name != tag.name:
        existing_tag = crud_tag.get_by_name(db, name=tag_in.name)
        if existing_tag:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tag with this name already exists"
            )
            
    return crud_tag.update(db=db, db_obj=tag, obj_in=tag_in)


@router.delete("/{tag_id}")
async def delete_tag(
    tag_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> dict:
    """
    Delete a tag.
    
    Args:
        tag_id: Tag ID
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If tag is not found
    """
    tag = crud_tag.get(db=db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    crud_tag.remove(db=db, id=tag_id)
    return {"message": "Tag deleted successfully"} 