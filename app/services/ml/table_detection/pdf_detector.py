"""PDF document table detection implementation using Microsoft Table Transformer."""
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image
import io
from transformers import (
    DetrImageProcessor,
    TableTransformerForObjectDetection,
    TableTransformerModel
)
import pytesseract
from pdf2image import convert_from_path
import cv2

from app.services.ml.base import BaseTableDetector
from app.schemas.analysis import (
    TableDetectionParameters,
    TableDetectionResult,
    PageTableInfo,
    DetectedTable,
    TableCell,
    BoundingBox
)
from app.core.config import settings

logger = logging.getLogger(__name__)

class PDFTableDetector(BaseTableDetector):
    """Service for detecting tables in PDF documents using Microsoft Table Transformer."""
    
    def __init__(self):
        """Initialize the PDF table detection service with Table Transformer model."""
        logger.debug("Initializing PDF Table Detection Service")
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.confidence_threshold = 0.5
        
    @property
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        return ["pdf"]

    def _load_model(self) -> None:
        """Load the Table Transformer model and processor."""
        try:
            if self.model is None:
                model_name = "microsoft/table-transformer-detection"
                self.processor = DetrImageProcessor.from_pretrained(model_name)
                self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Table Transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Table Transformer model: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")

    def _convert_pdf_to_images(self, file_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Images using either poppler or PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of PIL Images, one per page
        """
        try:
            # First try using pdf2image (poppler)
            try:
                import platform
                if platform.system() == "Darwin":  # macOS
                    logger.info("Using pdf2image with poppler on macOS")
                    from pdf2image import convert_from_path
                    return convert_from_path(
                        file_path,
                        dpi=300,  # Higher DPI for better quality
                        fmt="png",
                        poppler_path="/opt/homebrew/bin"  # macOS Homebrew path
                    )
                else:
                    logger.info("Using pdf2image with system poppler")
                    from pdf2image import convert_from_path
                    return convert_from_path(
                        file_path,
                        dpi=300,
                        fmt="png"
                    )
            except Exception as e:
                logger.warning(f"pdf2image conversion failed, falling back to PyMuPDF: {str(e)}")
                
            # Fallback to PyMuPDF
            logger.info("Converting PDF using PyMuPDF")
            doc = fitz.open(file_path)
            images = []
            
            for page in doc:
                # Get the page as a PNG image with higher resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))  # 3x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            doc.close()
            logger.debug(f"Converted {len(images)} PDF pages to images using PyMuPDF")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            raise ValueError(f"PDF conversion failed: {str(e)}")

    def _detect_tables_in_image(
        self,
        image: Image.Image,
        min_confidence: float
    ) -> List[DetectedTable]:
        """Detect tables in a single image using Table Transformer.
        
        Args:
            image: PIL Image of the page
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of DetectedTable objects
        """
        try:
            # Prepare image for the model
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert outputs to XYXY format
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=min_confidence,
                target_sizes=target_sizes
            )[0]
            
            detected_tables = []
            
            # Process each detected table
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                confidence = score.item()
                if confidence < min_confidence:
                    continue
                
                # Convert box coordinates to relative values
                x1, y1, x2, y2 = box.tolist()
                width, height = image.size
                bbox = BoundingBox(
                    x1=float(x1 / width),
                    y1=float(y1 / height),
                    x2=float(x2 / width),
                    y2=float(y2 / height)
                )
                
                # Extract table content
                table_region = image.crop((x1, y1, x2, y2))
                cells = self._extract_table_content(table_region)
                
                detected_tables.append(DetectedTable(
                    bbox=bbox,
                    confidence_score=confidence,
                    rows=len(set(cell.row_index for cell in cells)),
                    columns=len(set(cell.col_index for cell in cells)),
                    cells=cells,
                    has_headers=any(cell.is_header for cell in cells),
                    header_rows=[0] if any(cell.is_header for cell in cells) else []
                ))
            
            return detected_tables
            
        except Exception as e:
            logger.error(f"Failed to detect tables in image: {str(e)}")
            return []

    def _extract_table_content(self, table_image: Image.Image) -> List[TableCell]:
        """Extract table content using OCR and structure analysis.
        
        Args:
            table_image: Cropped image containing a single table
            
        Returns:
            List of TableCell objects
        """
        try:
            # Enhance image for better OCR
            enhanced_image = self._enhance_image_for_ocr(table_image)
            
            # Perform OCR with table structure recognition
            ocr_data = pytesseract.image_to_data(
                enhanced_image,
                output_type=pytesseract.Output.DICT,
                config='--psm 6 --oem 3'
            )
            
            cells = []
            current_row = 0
            current_col = 0
            last_y = None
            row_heights = []
            
            # Process OCR results
            for i in range(len(ocr_data["text"])):
                if not ocr_data["text"][i].strip():
                    continue
                
                # Get coordinates
                x = ocr_data["left"][i]
                y = ocr_data["top"][i]
                w = ocr_data["width"][i]
                h = ocr_data["height"][i]
                conf = float(ocr_data["conf"][i]) / 100
                
                # Detect new row based on y-coordinate
                if last_y is not None and abs(y - last_y) > h/2:
                    current_row += 1
                    current_col = 0
                    row_heights.append(h)
                last_y = y
                
                # Create cell
                cell = TableCell(
                    content=ocr_data["text"][i].strip(),
                    row_index=current_row,
                    col_index=current_col,
                    row_span=1,
                    col_span=1,
                    is_header=current_row == 0 and bool(ocr_data["text"][i].strip()),
                    confidence=conf
                )
                cells.append(cell)
                current_col += 1
            
            return cells
            
        except Exception as e:
            logger.error(f"Failed to extract table content: {str(e)}")
            return []

    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Additional preprocessing for better OCR
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            # Increase contrast
            contrast = cv2.convertScaleAbs(blurred, alpha=1.2, beta=0)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image

    async def detect_tables(
        self,
        file_path: str,
        parameters: Dict[str, Any]
    ) -> TableDetectionResult:
        """
        Detect and extract tables from a PDF document using Table Transformer.
        
        Args:
            file_path: Path to the PDF file
            parameters: Detection parameters
            
        Returns:
            TableDetectionResult containing detected tables
            
        Raises:
            ValueError: If file is invalid or processing fails
        """
        try:
            # Load model if not loaded
            self._load_model()
            
            # Convert parameters
            detection_params = TableDetectionParameters(**parameters)
            min_confidence = detection_params.confidence_threshold
            
            # Convert PDF to images
            images = self._convert_pdf_to_images(file_path)
            
            # Process each page
            pages = []
            total_confidence = 0.0
            total_tables = 0
            
            for page_num, image in enumerate(images, 1):
                # Detect tables in the page
                tables = self._detect_tables_in_image(image, min_confidence)
                
                if tables:
                    # Create page info
                    page_info = PageTableInfo(
                        page_number=page_num,
                        page_dimensions={
                            "width": float(image.size[0]),
                            "height": float(image.size[1])
                        },
                        tables=tables
                    )
                    pages.append(page_info)
                    
                    # Update statistics
                    total_tables += len(tables)
                    total_confidence += sum(table.confidence_score for table in tables)
            
            # Calculate average confidence
            avg_confidence = total_confidence / total_tables if total_tables > 0 else 0.0
            
            # Create final result
            return TableDetectionResult(
                pages=pages,
                total_tables=total_tables,
                average_confidence=avg_confidence,
                processing_metadata={
                    "processor": "Microsoft Table Transformer",
                    "model_version": self.model.config.model_type,
                    "parameters_used": parameters
                }
            )
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}", exc_info=True)
            raise ValueError(f"Table detection failed: {str(e)}")

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            if not Path(file_path).exists():
                logger.error(f"PDF file not found: {file_path}")
                return False
            
            # Try to open with PyMuPDF
            doc = fitz.open(file_path)
            is_valid = doc.page_count > 0
            doc.close()
            
            return is_valid
            
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False 