from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import os
import magic

from app.db.session import get_db
from app.db import models
from app.core import security
from app.schemas import documents as doc_schemas
from app.core.config import settings
from app.services.document_service import DocumentService

router = APIRouter()
document_service = DocumentService()

@router.post("/upload", response_model=doc_schemas.Document, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Upload a new document"""
    try:
        document = await document_service.create_document(file, current_user.id, db)
        return document
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{document_id}", response_model=doc_schemas.Document)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Get a specific document"""
    document = await document_service.get_document(document_id, current_user.id, db)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document

@router.get("/", response_model=List[doc_schemas.Document])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """List all documents for current user"""
    documents = await document_service.get_user_documents(current_user.id, db)
    return documents

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Delete a document"""
    success = await document_service.delete_document(document_id, current_user.id, db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

@router.post("/batch-analyze", response_model=List[doc_schemas.AnalysisResult])
async def batch_analyze_documents(
    analysis_request: doc_schemas.BatchAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Start analysis for multiple documents"""
    results = []
    for doc_id in analysis_request.document_ids:
        document = await document_service.get_document(doc_id, current_user.id, db)
        if document:
            analysis = await document_service.create_analysis(
                document, 
                analysis_request.analysis_config, 
                db
            )
            results.append(analysis)
    return results

@router.get("/parameters", response_model=doc_schemas.AnalysisParameters)
async def get_analysis_parameters():
    """Get available analysis parameters and options"""
    return {
        "available_types": [
            "table_detection",
            "text_extraction",
            "text_summarization",
            "template_conversion"
        ],
        "parameters": {
            "table_detection": {
                "confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
                "max_tables": {"type": "integer", "min": 1, "max": 100, "default": 10}
            },
            "text_extraction": {
                "language": {"type": "string", "options": ["en", "es", "fr"], "default": "en"},
                "ocr_enabled": {"type": "boolean", "default": True}
            },
            "text_summarization": {
                "max_length": {"type": "integer", "min": 100, "max": 1000, "default": 300},
                "style": {"type": "string", "options": ["concise", "detailed"], "default": "concise"}
            }
        }
    }

@router.post("/export", response_model=doc_schemas.ExportResponse)
async def export_results(
    export_request: doc_schemas.ExportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Export analysis results in specified format"""
    export_data = await document_service.generate_export(
        export_request.document_ids,
        export_request.format,
        current_user,
        db
    )
    return export_data 