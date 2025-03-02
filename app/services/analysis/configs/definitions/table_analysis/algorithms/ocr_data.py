from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from datetime import datetime
from app.services.analysis.configs.base import BaseAlgorithm
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase, AlgorithmParameter, AlgorithmParameterValue
from app.enums.document import DocumentType
from app.schemas.analysis.results.table_data import (
    TableDataResult,
    PageTableDataResult,
    TableData,
    CellContent
)
from app.schemas.analysis.results.table_shared import BoundingBox, Confidence, PageInfo

class OCRTableDataAlgorithm(BaseAlgorithm):
    """Advanced table data extraction using Tesseract OCR and data type detection"""
    
    def get_info(self) -> AlgorithmDefinitionBase:
        return AlgorithmDefinitionBase(
            code="ocr_data",
            name="OCR Table Data Extraction",
            version="1.0.0",
            description="Advanced table data extraction using Tesseract OCR with data type detection",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            parameters=[
                AlgorithmParameter(
                    name="ocr_lang",
                    description="Tesseract language code(s), comma-separated",
                    type="string",
                    required=False,
                    default="eng"
                ),
                AlgorithmParameter(
                    name="detect_data_types",
                    description="Whether to detect and normalize data types",
                    type="boolean",
                    required=False,
                    default=True
                ),
                AlgorithmParameter(
                    name="confidence_threshold",
                    description="Minimum confidence score for OCR results",
                    type="float",
                    required=False,
                    default=0.5,
                    constraints={
                        "min": 0.0,
                        "max": 1.0
                    }
                ),
                AlgorithmParameter(
                    name="preprocessing_method",
                    description="Image preprocessing method (none, basic, adaptive)",
                    type="string",
                    required=False,
                    default="adaptive"
                )
            ],
            implementation_path="app.services.analysis.configs.definitions.table_analysis.algorithms.ocr_data.OCRTableDataAlgorithm",
            is_active=True
        )

    def get_default_parameters(self) -> List[AlgorithmParameterValue]:
        """Get default parameters for the algorithm"""
        return [
            AlgorithmParameterValue(
                name="ocr_lang",
                value="eng"
            ),
            AlgorithmParameterValue(
                name="detect_data_types",
                value=True
            ),
            AlgorithmParameterValue(
                name="confidence_threshold",
                value=0.5
            ),
            AlgorithmParameterValue(
                name="preprocessing_method",
                value="adaptive"
            )
        ]
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        try:
            import cv2
            import numpy as np
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(f"Required dependency not found: {str(e)}")
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if "document_path" not in input_data:
            raise ValueError("Document path not provided in input data")
        if "structures" not in input_data:
            raise ValueError("Table structure results not provided in input data")

    def _preprocess_image(
        self,
        image: np.ndarray,
        method: str = "adaptive"
    ) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if method == "none":
            return gray
        elif method == "basic":
            # Basic global thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        else:  # adaptive
            # Advanced adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2    # C constant
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Dilation to make text more prominent
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)
            
            return dilated

    def _detect_data_type(self, text: str) -> tuple[str, Any]:
        """Detect data type and normalize value"""
        # Remove extra whitespace
        text = text.strip()
        if not text:
            return "empty", None

        # Check for numbers
        numeric_pattern = r'^-?\d*\.?\d+$'
        if re.match(numeric_pattern, text):
            try:
                value = float(text)
                return "number", value
            except ValueError:
                pass

        # Check for percentages
        percentage_pattern = r'^-?\d*\.?\d+\s*%$'
        if re.match(percentage_pattern, text):
            try:
                value = float(text.replace('%', '').strip()) / 100
                return "percentage", value
            except ValueError:
                pass

        # Check for dates (various formats)
        date_patterns = [
            ('%Y-%m-%d', r'^\d{4}-\d{2}-\d{2}$'),
            ('%d/%m/%Y', r'^\d{2}/\d{2}/\d{4}$'),
            ('%Y/%m/%d', r'^\d{4}/\d{2}/\d{2}$'),
            ('%d-%m-%Y', r'^\d{2}-\d{2}-\d{4}$'),
            ('%B %d, %Y', r'^[A-Za-z]+ \d{1,2}, \d{4}$')
        ]
        
        for date_format, pattern in date_patterns:
            if re.match(pattern, text):
                try:
                    value = datetime.strptime(text, date_format)
                    return "date", value.isoformat()
                except ValueError:
                    continue

        # Check for currency
        currency_pattern = r'^[$€£¥]?\s*-?\d*\.?\d+\s*[$€£¥]?$'
        if re.match(currency_pattern, text):
            try:
                # Extract numeric value
                value = float(re.findall(r'-?\d*\.?\d+', text)[0])
                return "currency", value
            except (ValueError, IndexError):
                pass

        # Default to text
        return "text", text

    def _extract_cell_content(
        self,
        image: np.ndarray,
        bbox: Dict[str, int],
        ocr_lang: str,
        detect_types: bool,
        conf_threshold: float,
        preprocess_method: str
    ) -> CellContent:
        """Extract content from a single cell"""
        # Extract cell region
        cell_img = image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        
        # Preprocess cell image
        processed_img = self._preprocess_image(cell_img, preprocess_method)
        
        # Convert to PIL Image for Tesseract
        pil_img = Image.fromarray(processed_img)
        
        # Perform OCR
        ocr_result = pytesseract.image_to_data(
            pil_img,
            lang=ocr_lang,
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # Assume uniform block of text
        )
        
        # Combine text and get confidence
        text_parts = []
        conf_scores = []
        
        for i, conf in enumerate(ocr_result['conf']):
            if conf > conf_threshold * 100:  # Tesseract confidence is 0-100
                text = ocr_result['text'][i].strip()
                if text:
                    text_parts.append(text)
                    conf_scores.append(conf)
        
        # Combine results
        text = ' '.join(text_parts)
        avg_conf = np.mean(conf_scores) / 100 if conf_scores else 0.0
        
        # Detect data type if enabled
        data_type = None
        normalized_value = None
        if detect_types and text:
            data_type, normalized_value = self._detect_data_type(text)
        
        return CellContent(
            text=text,
            confidence=Confidence(
                score=float(avg_conf),
                method="tesseract_ocr"
            ),
            data_type=data_type,
            normalized_value=normalized_value
        )

    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table data extraction"""
        try:
            # Get parameters
            ocr_lang = parameters.get("ocr_lang", "eng")
            detect_types = parameters.get("detect_data_types", True)
            conf_threshold = parameters.get("confidence_threshold", 0.5)
            preprocess_method = parameters.get("preprocessing_method", "adaptive")
            
            # Extract document pages using document service
            from app.services.documents.document import extract_document_pages
            from app.enums.document import DocumentType

            # Determine document type from file extension
            if document_path.lower().endswith('.pdf'):
                doc_type = DocumentType.PDF
            elif document_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                doc_type = DocumentType.IMAGE
            elif document_path.lower().endswith(('.docx', '.doc')):
                doc_type = DocumentType.DOCX
            else:
                raise ValueError(f"Unsupported document type: {document_path}")

            # Extract user_id and document_id from path
            path_parts = document_path.split('/')
            if len(path_parts) < 3:
                raise ValueError(f"Invalid document path format: {document_path}")
            user_id = path_parts[0]
            document_id = path_parts[1]

            # Get document pages
            document_pages = await extract_document_pages(
                document_path=f"/uploads/{document_path}",
                document_type=doc_type,
                user_id=user_id,
                document_id=document_id
            )
            
            # Get structure results
            structure_results = previous_results.get("table_analysis.table_structure", {})
            if not structure_results:
                raise ValueError("No table structure results found in previous steps")
            
            # Process each page
            final_results = []
            total_tables = 0
            
            for page_idx, page_result in enumerate(structure_results.get("results", [])):
                # Get corresponding document page
                doc_page = next((p for p in document_pages.pages if p.page_number == page_idx + 1), None)
                if not doc_page:
                    continue

                # Load image from the page's image_url
                image_path = doc_page.image_url.lstrip('/')
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image from {image_path}")

                page_tables = []
                page_info = page_result["page_info"]
                
                for table_structure in page_result["tables"]:
                    # Initialize 2D array for table data
                    num_rows = table_structure["num_rows"]
                    num_cols = table_structure["num_cols"]
                    table_data = [[None for _ in range(num_cols)] for _ in range(num_rows)]
                    
                    # Process each cell
                    for cell in table_structure["cells"]:
                        # Get cell position from bbox
                        row_idx = cell["row"]
                        col_idx = cell["col"]
                        
                        # Extract cell content
                        content = self._extract_cell_content(
                            image,
                            cell["bbox"],
                            ocr_lang,
                            detect_types,
                            conf_threshold,
                            preprocess_method
                        )
                        
                        # Handle merged cells
                        row_span = cell.get("row_span", 1)
                        col_span = cell.get("col_span", 1)
                        
                        # Fill all spanned cells with the same content
                        for r in range(row_idx, row_idx + row_span):
                            for c in range(col_idx, col_idx + col_span):
                                table_data[r][c] = content
                    
                    # Create table data object
                    table = TableData(
                        bbox=BoundingBox(**table_structure["bbox"]),
                        cells=table_data,
                        confidence=Confidence(
                            score=float(table_structure["confidence"]["score"]),
                            method="ocr_extraction"
                        )
                    )
                    
                    page_tables.append(table)
                    total_tables += 1
                
                # Create page result
                if page_tables:
                    page_result = PageTableDataResult(
                        page_info=PageInfo(**page_info),
                        tables=page_tables,
                        processing_info={
                            "ocr_language": ocr_lang,
                            "preprocessing_method": preprocess_method,
                            "data_types_detected": detect_types,
                            "confidence_threshold": conf_threshold
                        }
                    )
                    final_results.append(page_result)
            
            # Create final result
            final_result = TableDataResult(
                results=final_results,
                total_pages_processed=len(final_results),
                total_tables_processed=total_tables,
                metadata={
                    "algorithm": "ocr_data",
                    "version": "1.0.0",
                    "parameters": parameters,
                    "ocr_engine": "tesseract"
                }
            )
            
            return final_result.dict()
            
        except Exception as e:
            raise RuntimeError(f"Table data extraction failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass 