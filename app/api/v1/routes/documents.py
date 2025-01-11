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
from app.schemas.document import (
    Document,
    DocumentCreate,
    DocumentUpdate,
    DocumentWithAnalysis,
    DocumentType
)
from app.schemas.analysis import AnalysisStatus
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
    try:
        logger.info(f"Processing document upload: {file.filename}", extra={
            "user_id": str(current_user.id),
            "file_size": file.size,
            "content_type": file.content_type
        })

        doc_create, _ = await validate_and_save_file(
            file, str(current_user.id), background_tasks
        )
        
        document = crud_document.create_with_user(
            db=db,
            obj_in=doc_create,
            user_id=str(current_user.id)
        )
        
        logger.info(f"Successfully created document: {file.filename}", extra={
            "user_id": str(current_user.id),
            "document_id": str(document.id)
        })
        return document
    except Exception as e:
        logger.error(f"Failed to create document: {file.filename}", extra={
            "user_id": str(current_user.id),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))


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
    update_data: DocumentUpdate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Document:
    """
    Update a document's metadata.
    
    Args:
        document_id: Document ID to update
        update_data: Updated document data
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
            "update_data": update_data.model_dump(exclude_unset=True)
        })
        
        # Update the document
        updated_document = crud_document.update(
            db=db,
            db_obj=document,
            obj_in=update_data
        )
        
        logger.info(f"Successfully updated document: {document_id}", extra={
            "user_id": str(current_user.id)
        })
        return updated_document
        
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