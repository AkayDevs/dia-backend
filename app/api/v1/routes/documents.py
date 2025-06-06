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
    Form
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
import hashlib
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.db import deps
from app.crud.crud_document import document as crud_document
from app.crud.crud_tag import tag as crud_tag
from app.schemas.document import (
    Document,
    DocumentCreate,
    DocumentWithAnalysis,
    DocumentType,
    DocumentUpdate
)
from app.schemas.tag import Tag, DocumentTags
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

        # Calculate file hash
        file_hash = hashlib.sha256(content).hexdigest()

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
            file_hash=file_hash
        )

        return document, str(file_path)

    except Exception as e:
        if 'file_path' in locals():
            background_tasks.add_task(lambda: Path(file_path).unlink(missing_ok=True))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


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


@router.post("", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Document:
    """
    Upload a single document.
    
    Args:
        file: File to upload
        background_tasks: Background tasks runner
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Created document
        
    Raises:
        HTTPException: If file validation or upload fails
    """
    doc_create, file_path = await validate_and_save_file(
        file, str(current_user.id), background_tasks
    )
    
    try:
        return crud_document.create_with_user(
            db=db,
            obj_in=doc_create,
            user_id=str(current_user.id)
        )
    except IntegrityError:
        # Clean up the uploaded file
        background_tasks.add_task(lambda: Path(file_path).unlink(missing_ok=True))
        
        # Return the existing document
        existing_doc = crud_document.get_by_hash(
            db=db,
            user_id=str(current_user.id),
            file_hash=doc_create.file_hash
        )
        if existing_doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "File already exists",
                    "existing_document": existing_doc
                }
            )
        raise


@router.get("", response_model=List[Document])
async def list_documents(
    doc_type: Optional[DocumentType] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Document]:
    """
    List user's documents with optional filtering.
    
    Args:
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
        doc_type=doc_type
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
    return document


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


@router.post("/{document_id}/tags", response_model=List[Tag])
async def update_document_tags(
    *,
    document_id: str,
    tags_in: DocumentTags,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Tag]:
    """
    Update tags for a document.
    
    Args:
        document_id: Document ID
        tags_in: List of tag IDs to assign
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Updated list of document tags
        
    Raises:
        HTTPException: If document is not found or tags are invalid
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
    
    try:
        # If no tags provided, assign default tag
        if not tags_in.tag_ids:
            default_tag = crud_tag.get_default_tag(db)
            return crud_tag.update_document_tags(
                db=db,
                document_id=document_id,
                tag_ids=[default_tag.id]
            )
        
        return crud_tag.update_document_tags(
            db=db,
            document_id=document_id,
            tag_ids=tags_in.tag_ids
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{document_id}/tags", response_model=List[Tag])
async def get_document_tags(
    document_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Tag]:
    """
    Get all tags for a document.
    
    Args:
        document_id: Document ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        List of document tags
        
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
    
    return crud_tag.get_document_tags(
        db=db,
        document_id=document_id,
        skip=skip,
        limit=limit
    )


@router.patch("/{document_id}", response_model=Document)
async def update_document(
    *,
    document_id: str,
    update_data: DocumentUpdate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Document:
    """
    Update a document.
    
    Args:
        document_id: Document ID
        update_data: Updated document data
        db: Database session
        current_user: Currently authenticated user
        
    Returns:
        Updated document
        
    Raises:
        HTTPException: If document is not found or update fails
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
    
    try:
        return crud_document.update_with_tags(
            db=db,
            db_obj=document,
            obj_in=update_data
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) 