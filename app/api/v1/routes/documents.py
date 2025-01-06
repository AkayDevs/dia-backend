from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
import os
import shutil
from datetime import datetime
import aiofiles
import magic  # for file type verification

from app.core.config import settings
from app.db import deps
from app.crud.crud_document import document as crud_document
from app.crud.crud_document import analysis_result as crud_analysis
from app.schemas.document import (
    Document,
    DocumentCreate,
    DocumentWithAnalysis,
    AnalysisParameters,
    AnalysisResult,
)
from app.db.models.document import DocumentType, AnalysisStatus
from app.db.models.user import User

router = APIRouter()

# MIME type mapping
MIME_TYPES = {
    'application/pdf': DocumentType.PDF,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentType.XLSX,
    'image/jpeg': DocumentType.IMAGE,
    'image/png': DocumentType.IMAGE,
}


def cleanup_file(file_path: str) -> None:
    """Clean up uploaded file in case of errors."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")


def validate_file_type(file_path: str, filename: str) -> DocumentType:
    """Validate file type using both extension and MIME type."""
    # Check extension
    ext = filename.lower().split('.')[-1]
    ext_mapping = {
        'pdf': DocumentType.PDF,
        'docx': DocumentType.DOCX,
        'xlsx': DocumentType.XLSX,
        'png': DocumentType.IMAGE,
        'jpg': DocumentType.IMAGE,
        'jpeg': DocumentType.IMAGE,
    }
    
    if ext not in ext_mapping:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}"
        )
    
    # Verify actual file content
    mime = magic.from_file(file_path, mime=True)
    if mime not in MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file content type: {mime}"
        )
    
    # Verify extension matches content
    if MIME_TYPES[mime] != ext_mapping[ext]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File extension doesn't match content type"
        )
    
    return ext_mapping[ext]


async def save_upload_file(file: UploadFile, user_id: str) -> tuple[str, str]:
    """Save uploaded file and return its URL and file path."""
    # Create user-specific upload directory
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(user_upload_dir, filename)
    
    try:
        # Save file using aiofiles for async I/O
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Return both URL and file path
        return f"/uploads/{user_id}/{filename}", file_path
    except Exception as e:
        cleanup_file(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )


@router.post("", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> Document:
    """
    Upload a new document.
    """
    # Validate file size
    try:
        content = await file.read()
        size = len(content)
        await file.seek(0)  # Reset file pointer
        
        if size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
            )
        
        # Save file
        file_url, file_path = await save_upload_file(file, str(current_user.id))
        
        try:
            # Validate file type
            doc_type = validate_file_type(file_path, file.filename)
            
            # Create document record
            doc_in = DocumentCreate(
                name=file.filename,
                type=doc_type,
                size=size,
                url=file_url,
            )
            
            document = crud_document.create_with_user(
                db=db,
                obj_in=doc_in,
                user_id=str(current_user.id)
            )
            
            return document
            
        except HTTPException:
            # Clean up file if validation fails
            background_tasks.add_task(cleanup_file, file_path)
            raise
        except Exception as e:
            # Clean up file if database operation fails
            background_tasks.add_task(cleanup_file, file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> dict:
    """
    Delete a document and its associated file.
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
    
    # Get file path from URL
    file_path = os.path.join(
        settings.UPLOAD_DIR,
        document.url.replace("/uploads/", "")
    )
    
    # Delete document from database
    crud_document.remove(db=db, id=document_id)
    
    # Schedule file cleanup
    background_tasks.add_task(cleanup_file, file_path)
    
    return {"message": "Document deleted successfully"}


@router.get("", response_model=List[Document])
def get_documents(
    status: Optional[AnalysisStatus] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> List[Document]:
    """
    Get all documents for the current user.
    """
    return crud_document.get_multi_by_user(
        db=db,
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
        status=status,
    )


@router.get("/{document_id}", response_model=DocumentWithAnalysis)
def get_document(
    document_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> DocumentWithAnalysis:
    """
    Get a specific document with its analysis results.
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


@router.post("/{document_id}/analyze")
async def analyze_document(
    document_id: str,
    analysis_params: AnalysisParameters,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> dict:
    """
    Start document analysis with specified parameters.
    """
    # Verify document ownership
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
    
    # Check if document is already being processed
    if document.status == AnalysisStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already being processed"
        )
    
    # Update document status
    crud_document.update_status(
        db=db,
        document_id=document_id,
        status=AnalysisStatus.PROCESSING
    )
    
    # TODO: Implement async analysis task
    # For now, just create a placeholder result
    result = crud_analysis.create_result(
        db=db,
        document_id=document_id,
        type=analysis_params.type,
        result={"status": "queued", "params": analysis_params.model_dump()}
    )
    
    return {
        "message": "Analysis started",
        "analysis_id": result.id
    }


@router.get("/{document_id}/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis_result(
    document_id: str,
    analysis_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_verified_user),
) -> AnalysisResult:
    """
    Get the result of a specific analysis.
    """
    # Verify document ownership
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
    
    # Get analysis result
    result = crud_analysis.get(db=db, id=analysis_id)
    if not result or result.document_id != document_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result not found"
        )
    
    return result 