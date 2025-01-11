from typing import List, Optional
from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    UploadFile, 
    File, 
    status, 
    Query, 
    BackgroundTasks,
    Form,
    Body
)
from sqlalchemy.orm import Session
import uuid
import os
import shutil
from datetime import datetime
import aiofiles
import magic
import asyncio
import logging
from pathlib import Path

from app.core.config import settings
from app.db import deps
from app.crud.crud_document import document as crud_document
from app.crud.crud_tag import tag as crud_tag
from app.schemas.document import (
    Document,
    DocumentCreate,
    DocumentUpdate,
    DocumentWithAnalysis,
    DocumentType,
    Tag,
    TagCreate
)
from app.db.models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)

# MIME type mapping
MIME_TYPES = {
    'application/pdf': DocumentType.PDF,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentType.XLSX,
    'image/jpeg': DocumentType.IMAGE,
    'image/png': DocumentType.IMAGE,
}


async def validate_and_save_file(
    file: UploadFile,
    user_id: str,
    background_tasks: BackgroundTasks
) -> tuple[Document, str]:
    """
    Validate and save an uploaded file.
    Returns a tuple of (Document schema, file path).
    """
    try:
        # Validate file size
        content = await file.read()
        size = len(content)
        await file.seek(0)

        if size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
            )

        # Create user-specific upload directory
        user_upload_dir = Path(settings.UPLOAD_DIR) / str(user_id)
        user_upload_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = user_upload_dir / filename

        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)

        # Validate file type
        mime = magic.from_file(str(file_path), mime=True)
        if mime not in MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {mime}"
            )

        doc_type = MIME_TYPES[mime]
        file_url = f"/uploads/{user_id}/{filename}"

        document = DocumentCreate(
            name=file.filename,
            type=doc_type,
            size=size,
            url=file_url,
        )

        return document, str(file_path)

    except Exception as e:
        if 'file_path' in locals():
            background_tasks.add_task(lambda: Path(file_path).unlink(missing_ok=True))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# Tag Management Endpoints
@router.get("/tag-list", response_model=List[Tag])
async def list_tags(
    document_id: Optional[str] = Query(None, description="Get tags for specific document"),
    name_filter: Optional[str] = Query(None, description="Filter tags by name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> List[Tag]:
    """
    List tags with optional filtering.
    
    Args:
        document_id: Optional document ID to get tags for a specific document
        name_filter: Optional string to filter tags by name
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        List of tags. If document_id is provided, returns tags for that document only.
    """
    if document_id:
        # Get document and verify ownership
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        if str(document.user_id) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return document.tags
        
    # Return all tags with optional name filtering
    return crud_tag.get_multi(
        db,
        skip=skip,
        limit=limit,
        name_filter=name_filter
    )

@router.post("/tag-list", response_model=Tag)
async def create_tag(
    tag_in: TagCreate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> Tag:
    """
    Create a new tag.
    """
    # Check if tag already exists
    existing_tag = crud_tag.get_by_name(db, name=tag_in.name)
    if existing_tag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tag with name '{tag_in.name}' already exists"
        )
    return crud_tag.create(db, obj_in=tag_in)

@router.delete("/tag-list/{tag_id}")
async def delete_tag(
    tag_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
):
    """
    Delete a tag.
    """
    tag = crud_tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    crud_tag.remove(db, id=tag_id)
    return {"message": "Tag deleted successfully"}

@router.patch("/tag-list/{tag_id}", response_model=Tag)
async def update_tag(
    tag_id: int,
    tag_in: TagCreate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> Tag:
    """
    Update a tag's properties.
    
    Args:
        tag_id: ID of the tag to update
        tag_in: Updated tag data
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Updated tag
        
    Raises:
        HTTPException: If tag is not found or name already exists
    """
    # Check if tag exists
    tag = crud_tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    
    # If name is being changed, check if new name already exists
    if tag_in.name != tag.name:
        existing_tag = crud_tag.get_by_name(db, name=tag_in.name)
        if existing_tag:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tag with name '{tag_in.name}' already exists"
            )
    
    # Update tag
    return crud_tag.update(db, db_obj=tag, obj_in=tag_in) 

# Document Tag Management
@router.put("/{document_id}/tags", response_model=Document)
async def update_document_tags(
    document_id: str,
    tag_ids: List[int] = Body(..., description="List of tag IDs to assign to the document"),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user)
) -> Document:
    """
    Update tags for a specific document.
    """
    # Check document exists and belongs to user
    document = crud_document.get(db, id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    if str(document.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Validate tags exist
    for tag_id in tag_ids:
        if not crud_tag.get(db, id=tag_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tag with id {tag_id} not found"
            )
    
    # Update document tags
    return crud_document.update_tags(db, db_obj=document, tag_ids=tag_ids)

# Document Management Endpoints
@router.post("", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    tag_ids: List[int] = Form(default=None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Document:
    """
    Upload a single document with optional tags.
    """
    try:
        logger.info(f"Processing document upload: {file.filename}", extra={
            "user_id": str(current_user.id),
            "file_size": file.size,
            "content_type": file.content_type,
            "tag_ids": tag_ids
        })

        # Validate tags if provided
        if tag_ids:
            for tag_id in tag_ids:
                if not crud_tag.get(db, id=tag_id):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tag with id {tag_id} not found"
                    )

        doc_create, _ = await validate_and_save_file(
            file, str(current_user.id), background_tasks
        )
        
        # Add tag IDs to document creation
        doc_create.tag_ids = tag_ids
        
        document = crud_document.create_with_user(
            db=db,
            obj_in=doc_create,
            user_id=str(current_user.id)
        )
        
        logger.info(f"Successfully created document: {file.filename}", extra={
            "user_id": str(current_user.id),
            "document_id": str(document.id),
            "tag_ids": tag_ids
        })
        return document
    except Exception as e:
        logger.error(f"Failed to create document: {file.filename}", extra={
            "user_id": str(current_user.id),
            "error": str(e),
            "tag_ids": tag_ids
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[Document])
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Document]:
    """
    Upload multiple documents in a batch.
    
    Args:
        files: List of files to upload
        background_tasks: Background tasks runner
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        List of created documents
        
    Raises:
        HTTPException: If any file fails validation or upload
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    documents = []
    errors = []

    for file in files:
        try:
            doc_create, file_path = await validate_and_save_file(
                file, str(current_user.id), background_tasks
            )
            
            document = crud_document.create_with_user(
                db=db,
                obj_in=doc_create,
                user_id=str(current_user.id)
            )
            documents.append(document)
            
        except HTTPException as e:
            errors.append({"filename": file.filename, "error": str(e.detail)})
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    if errors and not documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "All uploads failed", "errors": errors}
        )

    if errors:
        logger.warning(f"Some files failed to upload: {errors}")

    return documents


@router.get("", response_model=List[Document])
async def list_documents(
    tag_id: Optional[int] = Query(None, description="Filter by tag ID"),
    doc_type: Optional[DocumentType] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Document]:
    """
    List user's documents with optional filtering.
    
    Args:
        tag_id: Filter by tag ID
        doc_type: Filter by document type
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        List of documents
    """
    return crud_document.get_multi_by_user(
        db=db,
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
        doc_type=doc_type,
        tag_id=tag_id
    )


@router.get("/{document_id}", response_model=DocumentWithAnalysis)
async def get_document(
    document_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> DocumentWithAnalysis:
    """
    Get a specific document with its analysis results.
    
    Args:
        document_id: Document ID
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Document with analysis results
        
    Raises:
        HTTPException: If document is not found
    """
    try:
        logger.debug(f"Fetching document: {document_id}", extra={
            "user_id": str(current_user.id)
        })
        
        document = crud_document.get_document_with_results(
            db=db,
            document_id=document_id,
            user_id=str(current_user.id),
        )
        if not document:
            logger.warning(f"Document not found: {document_id}", extra={
                "user_id": str(current_user.id)
            })
            raise HTTPException(status_code=404, detail="Document not found")
            
        logger.info(f"Successfully retrieved document: {document_id}", extra={
            "user_id": str(current_user.id)
        })
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {document_id}", extra={
            "user_id": str(current_user.id),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> dict:
    """
    Delete a document and its associated file.
    
    Args:
        document_id: Document ID
        background_tasks: Background tasks runner
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If document is not found
    """
    document = crud_document.get_document_with_results(
        db=db,
        document_id=document_id,
        user_id=str(current_user.id),
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Delete document from database
    crud_document.remove(db=db, id=document_id)

    # Schedule file cleanup
    file_path = Path(settings.UPLOAD_DIR) / document.url.replace("/uploads/", "")
    background_tasks.add_task(lambda: file_path.unlink(missing_ok=True))

    return {"message": "Document deleted successfully"}


@router.patch("/{document_id}", response_model=Document)
async def update_document(
    document_id: str,
    file: Optional[UploadFile] = File(None),
    update_data: Optional[DocumentUpdate] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Document:
    """
    Update a document's metadata and/or content.
    
    Args:
        document_id: Document ID to update
        file: Optional new file content
        update_data: Optional metadata updates
        background_tasks: Background tasks runner
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Updated document
        
    Raises:
        HTTPException: If document is not found or user doesn't have permission
    """
    try:
        # Check if document exists and belongs to user
        document = crud_document.get(db, id=document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        if str(document.user_id) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )

        logger.info(f"Updating document: {document_id}", extra={
            "user_id": str(current_user.id),
            "has_file": bool(file),
            "update_data": update_data.model_dump(exclude_unset=True) if update_data else None
        })
        
        # If there's a new file, handle content update
        if file:
            # Validate and save the new file
            doc_create, file_path = await validate_and_save_file(
                file, str(current_user.id), background_tasks
            )
            
            # Archive the old document
            document.is_archived = True
            document.archived_at = datetime.utcnow()
            db.add(document)
            
            # Create new document version
            new_document = Document(
                id=str(uuid.uuid4()),
                name=update_data.name if update_data and update_data.name else document.name,
                type=doc_create.type,
                size=doc_create.size,
                url=doc_create.url,
                user_id=str(current_user.id),
                previous_version_id=document.id,
                tags=document.tags  # Preserve tags from previous version
            )
            db.add(new_document)
            
            # Schedule cleanup of old file after retention period
            old_file_path = Path(settings.UPLOAD_DIR) / document.url.replace("/uploads/", "")
            background_tasks.add_task(
                lambda: asyncio.sleep(settings.ARCHIVE_RETENTION_DAYS * 24 * 60 * 60) and old_file_path.unlink(missing_ok=True)
            )
            
            db.commit()
            db.refresh(new_document)
            return new_document
            
        # If only metadata update, use existing update logic
        elif update_data:
            updated_document = crud_document.update(
                db=db,
                db_obj=document,
                obj_in=update_data
            )
            logger.info(f"Successfully updated document metadata: {document_id}", extra={
                "user_id": str(current_user.id)
            })
            return updated_document
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {document_id}", extra={
            "user_id": str(current_user.id),
            "error": str(e)
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document"
        ) 