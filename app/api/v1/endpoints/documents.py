from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.db.session import get_db
from app.db import models
from app.core import security
from app.schemas import documents as doc_schemas

router = APIRouter()

@router.post("/upload", response_model=doc_schemas.Document)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Upload a new document for analysis"""
    document = await security.create_document(file, current_user, db)
    return document

@router.post("/batch-upload", response_model=List[doc_schemas.Document])
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Upload multiple documents at once"""
    documents = []
    for file in files:
        document = await security.create_document(file, current_user, db)
        documents.append(document)
    return documents

@router.get("/list", response_model=List[doc_schemas.Document])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """List all documents for current user"""
    query = select(models.Document).where(models.Document.owner_id == current_user.id)
    result = await db.execute(query)
    documents = result.scalars().all()
    return documents

@router.post("/{document_id}/analyze", response_model=doc_schemas.AnalysisResult)
async def analyze_document(
    document_id: int,
    analysis_config: doc_schemas.AnalysisConfig,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Start document analysis"""
    document = await security.get_document(document_id, current_user, db)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    analysis = await security.create_analysis(document, analysis_config, db)
    return analysis

@router.post("/batch-analyze", response_model=List[doc_schemas.AnalysisResult])
async def batch_analyze_documents(
    analysis_request: doc_schemas.BatchAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Start analysis for multiple documents"""
    results = []
    for doc_id in analysis_request.document_ids:
        document = await security.get_document(doc_id, current_user, db)
        if document:
            analysis = await security.create_analysis(document, analysis_request.analysis_config, db)
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

@router.get("/history", response_model=List[doc_schemas.AnalysisResult])
async def get_analysis_history(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Get analysis history for current user"""
    query = select(models.AnalysisResult).join(models.Document).where(
        models.Document.owner_id == current_user.id
    )
    result = await db.execute(query)
    history = result.scalars().all()
    return history

@router.post("/export", response_model=doc_schemas.ExportResponse)
async def export_results(
    export_request: doc_schemas.ExportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Export analysis results in specified format"""
    export_data = await security.generate_export(
        export_request.document_ids,
        export_request.format,
        current_user,
        db
    )
    return export_data

@router.get("/results/{document_id}", response_model=doc_schemas.AnalysisResult)
async def get_analysis_results(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user)
):
    """Get analysis results for a specific document"""
    document = await security.get_document(document_id, current_user, db)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    result = await security.get_latest_analysis(document, db)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis results found"
        )
    return result 