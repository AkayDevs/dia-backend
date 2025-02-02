from typing import List
import fitz  # PyMuPDF
from pathlib import Path
import os
from PIL import Image
import io
import logging
from fastapi import HTTPException, status

from app.core.config import settings
from app.enums.document import DocumentType
from app.schemas.document import DocumentPage, DocumentPages

logger = logging.getLogger(__name__)

async def extract_document_pages(document_path: str, document_type: DocumentType, user_id: str) -> DocumentPages:
    """
    Extract pages from a document and convert them to images.
    Returns a list of page information including dimensions and image URLs.
    """
    try:
        full_path = Path(settings.UPLOAD_DIR) / document_path.replace("/uploads/", "")
        if not full_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )

        # Create directory for page images if it doesn't exist
        pages_dir = Path(settings.UPLOAD_DIR) / str(user_id) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        pages = []
        
        if document_type == DocumentType.PDF:
            # Process PDF document
            doc = fitz.open(str(full_path))
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    
                    # Save page as PNG
                    image_filename = f"{full_path.stem}_page_{page_num + 1}.png"
                    image_path = pages_dir / image_filename
                    pix.save(str(image_path))
                    
                    pages.append(DocumentPage(
                        page_number=page_num + 1,
                        width=pix.width,
                        height=pix.height,
                        image_url=f"/uploads/{user_id}/pages/{image_filename}"
                    ))
            finally:
                doc.close()
                
        elif document_type == DocumentType.IMAGE:
            # Process single image document
            with Image.open(full_path) as img:
                width, height = img.size
                image_filename = f"{full_path.stem}_page_1.png"
                image_path = pages_dir / image_filename
                
                # Convert to PNG if not already
                if full_path.suffix.lower() != '.png':
                    img.save(str(image_path), 'PNG')
                else:
                    # Just copy the file if it's already PNG
                    import shutil
                    shutil.copy2(full_path, image_path)
                
                pages.append(DocumentPage(
                    page_number=1,
                    width=width,
                    height=height,
                    image_url=f"/uploads/{user_id}/pages/{image_filename}"
                ))
                
        elif document_type == DocumentType.DOCX:
            # For DOCX files, we'll need to convert to PDF first
            # This is a placeholder - you'll need to implement DOCX conversion
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DOCX page extraction not implemented yet"
            )
            
        elif document_type == DocumentType.XLSX:
            # For XLSX files, we'll need special handling
            # This is a placeholder - you'll need to implement XLSX conversion
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="XLSX page extraction not implemented yet"
            )
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported document type: {document_type}"
            )
            
        return DocumentPages(
            total_pages=len(pages),
            pages=pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting document pages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract document pages: {str(e)}"
        ) 