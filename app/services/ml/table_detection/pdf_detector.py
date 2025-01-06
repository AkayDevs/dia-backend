from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import cv2
from collections import defaultdict
import pytesseract
import pandas as pd

from app.services.ml.base import BaseTableDetector
from .image_detector import ImageTableDetector

logger = logging.getLogger(__name__)

class PDFTableDetector(BaseTableDetector):
    """Service for detecting tables in PDF documents using hybrid approach."""
    
    def __init__(self):
        """Initialize the PDF table detection service."""
        logger.debug("Initializing PDF Table Detection Service")
        self.image_detector = ImageTableDetector()
        
    @property
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        return ["pdf"]

    def _detect_lines(self, page: fitz.Page) -> Dict[str, List]:
        """
        Detect horizontal and vertical lines in a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Dictionary containing horizontal and vertical lines
        """
        # Get page drawings
        paths = page.get_drawings()
        
        horizontals = []
        verticals = []
        
        for path in paths:
            # Get line coordinates
            rect = path["rect"]  # x0, y0, x1, y1
            
            # Calculate line properties
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # Classify as horizontal or vertical
            if height < 2 and width > 10:  # Horizontal line
                horizontals.append({
                    "x0": rect[0],
                    "x1": rect[2],
                    "y": rect[1],
                    "width": width
                })
            elif width < 2 and height > 10:  # Vertical line
                verticals.append({
                    "x": rect[0],
                    "y0": rect[1],
                    "y1": rect[3],
                    "height": height
                })
                
        return {
            "horizontal": sorted(horizontals, key=lambda x: x["y"]),
            "vertical": sorted(verticals, key=lambda x: x["x"])
        }

    def _find_intersections(self, lines: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Find line intersections to identify potential table cells.
        
        Args:
            lines: Dictionary of horizontal and vertical lines
            
        Returns:
            List of potential table regions
        """
        tables = []
        horizontals = lines["horizontal"]
        verticals = lines["vertical"]
        
        # Group nearby horizontal lines
        h_groups = []
        current_group = []
        
        for line in horizontals:
            if not current_group or abs(line["y"] - current_group[-1]["y"]) < 5:
                current_group.append(line)
            else:
                if len(current_group) > 1:
                    h_groups.append(current_group)
                current_group = [line]
                
        if len(current_group) > 1:
            h_groups.append(current_group)
            
        # Find table regions
        for h_group in h_groups:
            if len(h_group) < 2:  # Need at least 2 horizontal lines
                continue
                
            # Find vertical lines that intersect with horizontal group
            y_min = min(line["y"] for line in h_group)
            y_max = max(line["y"] for line in h_group)
            
            intersecting_verticals = []
            for v_line in verticals:
                if v_line["y0"] <= y_min and v_line["y1"] >= y_max:
                    intersecting_verticals.append(v_line)
                    
            if len(intersecting_verticals) < 2:  # Need at least 2 vertical lines
                continue
                
            # Create table region
            x_min = min(line["x"] for line in intersecting_verticals)
            x_max = max(line["x"] for line in intersecting_verticals)
            
            tables.append({
                "coordinates": {
                    "x1": x_min,
                    "y1": y_min,
                    "x2": x_max,
                    "y2": y_max
                },
                "width": x_max - x_min,
                "height": y_max - y_min,
                "area": (x_max - x_min) * (y_max - y_min),
                "confidence": 0.9  # High confidence for rule-based detection
            })
            
        return tables

    def _merge_overlapping_tables(self, tables: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Merge overlapping table regions.
        
        Args:
            tables: List of detected tables
            iou_threshold: IoU threshold for merging
            
        Returns:
            List of merged table regions
        """
        if not tables:
            return []
            
        def calculate_iou(box1: Dict, box2: Dict) -> float:
            """Calculate Intersection over Union."""
            x1 = max(box1["coordinates"]["x1"], box2["coordinates"]["x1"])
            y1 = max(box1["coordinates"]["y1"], box2["coordinates"]["y1"])
            x2 = min(box1["coordinates"]["x2"], box2["coordinates"]["x2"])
            y2 = min(box1["coordinates"]["y2"], box2["coordinates"]["y2"])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = box1["area"]
            area2 = box2["area"]
            
            return intersection / (area1 + area2 - intersection)
        
        merged = []
        used = set()
        
        for i, table1 in enumerate(tables):
            if i in used:
                continue
                
            current_table = table1.copy()
            used.add(i)
            
            for j, table2 in enumerate(tables[i+1:], i+1):
                if j in used:
                    continue
                    
                if calculate_iou(current_table, table2) > iou_threshold:
                    # Merge tables
                    coords1 = current_table["coordinates"]
                    coords2 = table2["coordinates"]
                    
                    merged_coords = {
                        "x1": min(coords1["x1"], coords2["x1"]),
                        "y1": min(coords1["y1"], coords2["y1"]),
                        "x2": max(coords1["x2"], coords2["x2"]),
                        "y2": max(coords1["y2"], coords2["y2"])
                    }
                    
                    current_table["coordinates"] = merged_coords
                    current_table["width"] = merged_coords["x2"] - merged_coords["x1"]
                    current_table["height"] = merged_coords["y2"] - merged_coords["y1"]
                    current_table["area"] = current_table["width"] * current_table["height"]
                    current_table["confidence"] = max(
                        current_table.get("confidence", 0),
                        table2.get("confidence", 0)
                    )
                    
                    used.add(j)
                    
            merged.append(current_table)
            
        return merged

    def _check_pdf_suitability(self, doc: fitz.Document) -> bool:
        """
        Check if the PDF is suitable for ML-based detection.
        
        Args:
            doc: PyMuPDF document
            
        Returns:
            Boolean indicating if ML detection should be used
        """
        try:
            # Check first page as a sample
            page = doc[0]
            
            # Try to get image
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            if not pix.samples:
                logger.warning("PDF page cannot be converted to image")
                return False
            
            # Check if page has vector graphics
            paths = page.get_drawings()
            if paths:
                logger.info("PDF contains vector graphics, using hybrid detection")
                return True
            
            # If no vector graphics but can be converted to image, use ML
            logger.info("PDF can be processed with ML detection")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check PDF suitability: {str(e)}")
            logger.warning("Defaulting to rule-based detection only")
            return False

    def _extract_table_data(
        self,
        page: fitz.Page,
        table: Dict[str, Any],
        enhance_image: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured data from a detected table.
        
        Args:
            page: PDF page containing the table
            table: Detected table information with coordinates
            enhance_image: Whether to enhance image for better OCR
            
        Returns:
            Dictionary containing structured table data
        """
        try:
            # Get table region coordinates
            coords = table["coordinates"]
            rect = fitz.Rect(
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"]
            )
            
            # Extract text blocks in the table region
            text_blocks = []
            for block in page.get_text("dict", clip=rect)["blocks"]:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_blocks.append({
                                "text": span["text"],
                                "bbox": span["bbox"],
                                "font_size": span["size"],
                                "font": span.get("font", ""),
                                "is_bold": "bold" in span.get("font", "").lower()
                            })
            
            if not text_blocks:
                # If no vector text found, try OCR
                return self._extract_table_data_ocr(page, table, enhance_image)
            
            # Sort blocks by position
            text_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))  # Sort by y, then x
            
            # Analyze table structure
            structure = self._analyze_table_structure(text_blocks, table)
            
            # Extract and organize data
            data = self._organize_table_data(text_blocks, structure)
            
            return {
                "structure": structure,
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Failed to extract table data: {str(e)}")
            return {"error": str(e)}

    def _extract_table_data_ocr(
        self,
        page: fitz.Page,
        table: Dict[str, Any],
        enhance_image: bool = True
    ) -> Dict[str, Any]:
        """Extract table data using OCR when vector text is not available."""
        try:
            # Get table region
            coords = table["coordinates"]
            rect = fitz.Rect(
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"]
            )
            
            # Convert region to image
            zoom = 2  # Increase resolution for better OCR
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            if enhance_image:
                img = self._enhance_image_for_ocr(img)
            
            # Perform OCR with table structure recognition
            ocr_data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform block of text
            )
            
            # Convert OCR data to structured format
            text_blocks = []
            for i in range(len(ocr_data["text"])):
                if ocr_data["conf"][i] > 30:  # Filter low confidence results
                    text_blocks.append({
                        "text": ocr_data["text"][i],
                        "bbox": (
                            ocr_data["left"][i] / zoom + coords["x1"],
                            ocr_data["top"][i] / zoom + coords["y1"],
                            (ocr_data["left"][i] + ocr_data["width"][i]) / zoom + coords["x1"],
                            (ocr_data["top"][i] + ocr_data["height"][i]) / zoom + coords["y1"]
                        ),
                        "confidence": ocr_data["conf"][i]
                    })
            
            # Analyze structure and organize data
            structure = self._analyze_table_structure(text_blocks, table)
            data = self._organize_table_data(text_blocks, structure)
            
            return {
                "structure": structure,
                "data": data,
                "ocr_used": True
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return {"error": str(e)}

    def _enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Enhance image for better OCR results."""
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
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
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return img

    def _analyze_table_structure(
        self,
        text_blocks: List[Dict[str, Any]],
        table: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze table structure from text blocks."""
        # Get table boundaries
        coords = table["coordinates"]
        table_width = coords["x2"] - coords["x1"]
        table_height = coords["y2"] - coords["y1"]
        
        # Find rows by clustering y-coordinates
        y_coords = [block["bbox"][1] for block in text_blocks]
        rows = self._cluster_coordinates(y_coords, table_height * 0.02)  # 2% threshold
        
        # Find columns by clustering x-coordinates
        x_coords = [block["bbox"][0] for block in text_blocks]
        columns = self._cluster_coordinates(x_coords, table_width * 0.02)  # 2% threshold
        
        # Detect headers
        header_row = self._detect_header_row(text_blocks, rows[0] if rows else None)
        
        # Analyze column types
        column_types = self._analyze_column_types(text_blocks, columns)
        
        return {
            "dimensions": {
                "rows": len(rows),
                "cols": len(columns)
            },
            "has_header": header_row is not None,
            "header_row": header_row,
            "row_positions": rows,
            "column_positions": columns,
            "column_types": column_types
        }

    def _cluster_coordinates(
        self,
        coords: List[float],
        threshold: float
    ) -> List[float]:
        """Cluster coordinates to find rows/columns."""
        if not coords:
            return []
            
        # Sort coordinates
        sorted_coords = sorted(coords)
        clusters = [[sorted_coords[0]]]
        
        # Cluster coordinates
        for coord in sorted_coords[1:]:
            if coord - clusters[-1][-1] > threshold:
                clusters.append([])
            clusters[-1].append(coord)
        
        # Return average position for each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def _detect_header_row(
        self,
        text_blocks: List[Dict[str, Any]],
        first_row_y: Optional[float]
    ) -> Optional[int]:
        """Detect if first row is a header."""
        if not first_row_y:
            return None
            
        # Get blocks in first row
        first_row_blocks = [
            block for block in text_blocks
            if abs(block["bbox"][1] - first_row_y) < 5
        ]
        
        # Check for header indicators
        header_indicators = 0
        for block in first_row_blocks:
            if (
                block.get("is_bold") or
                block.get("font_size", 0) > 0  # Larger font
            ):
                header_indicators += 1
        
        return 0 if header_indicators > len(first_row_blocks) / 2 else None

    def _analyze_column_types(
        self,
        text_blocks: List[Dict[str, Any]],
        columns: List[float]
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze data types for each column."""
        column_values = defaultdict(list)
        
        # Group values by column
        for block in text_blocks:
            col_idx = self._get_column_index(block["bbox"][0], columns)
            if col_idx is not None:
                column_values[col_idx].append(block["text"])
        
        # Analyze each column
        column_types = {}
        for col_idx, values in column_values.items():
            column_types[col_idx] = self._determine_column_type(values)
        
        return column_types

    def _get_column_index(
        self,
        x: float,
        columns: List[float]
    ) -> Optional[int]:
        """Get column index for an x-coordinate."""
        for i, col_x in enumerate(columns):
            if abs(x - col_x) < 20:  # 20px threshold
                return i
        return None

    def _determine_column_type(
        self,
        values: List[str]
    ) -> Dict[str, Any]:
        """Determine the type of data in a column."""
        if not values:
            return {"type": "unknown", "confidence": 0.0}
        
        numeric_count = 0
        date_count = 0
        total = len(values)
        
        for value in values:
            # Try numeric
            try:
                float(value.replace(',', ''))
                numeric_count += 1
                continue
            except ValueError:
                pass
            
            # Try date
            try:
                pd.to_datetime(value)
                date_count += 1
            except (ValueError, TypeError):
                pass
        
        # Determine type
        if numeric_count / total > 0.8:
            return {
                "type": "numeric",
                "confidence": numeric_count / total,
                "format": "integer" if all('.' not in v for v in values) else "float"
            }
        elif date_count / total > 0.8:
            return {
                "type": "date",
                "confidence": date_count / total
            }
        
        return {
            "type": "text",
            "confidence": 1.0
        }

    def _organize_table_data(
        self,
        text_blocks: List[Dict[str, Any]],
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Organize extracted text into structured table data."""
        rows = structure["row_positions"]
        cols = structure["column_positions"]
        has_header = structure["has_header"]
        
        # Initialize data structure
        data = {
            "headers": [] if has_header else None,
            "data": []
        }
        
        # Create grid for data organization
        grid = [[[] for _ in cols] for _ in rows]
        
        # Place blocks in grid
        for block in text_blocks:
            row_idx = self._get_row_index(block["bbox"][1], rows)
            col_idx = self._get_column_index(block["bbox"][0], cols)
            
            if row_idx is not None and col_idx is not None:
                grid[row_idx][col_idx].append(block)
        
        # Extract headers if present
        if has_header and grid:
            data["headers"] = [
                {
                    "text": " ".join(block["text"] for block in cell),
                    "formatting": {
                        "is_bold": any(block.get("is_bold", False) for block in cell),
                        "font_size": max((block.get("font_size", 0) for block in cell), default=0)
                    }
                }
                for cell in grid[0]
            ]
        
        # Extract data rows
        start_row = 1 if has_header else 0
        for row in grid[start_row:]:
            row_data = []
            for cell in row:
                cell_text = " ".join(block["text"] for block in cell)
                row_data.append({
                    "text": cell_text,
                    "value": self._convert_value(
                        cell_text,
                        structure["column_types"].get(len(row_data))
                    )
                })
            data["data"].append(row_data)
        
        return data

    def _get_row_index(
        self,
        y: float,
        rows: List[float]
    ) -> Optional[int]:
        """Get row index for a y-coordinate."""
        for i, row_y in enumerate(rows):
            if abs(y - row_y) < 10:  # 10px threshold
                return i
        return None

    def _convert_value(
        self,
        text: str,
        column_type: Optional[Dict[str, Any]]
    ) -> Any:
        """Convert text to appropriate data type."""
        if not column_type or not text:
            return text
            
        try:
            if column_type["type"] == "numeric":
                return float(text.replace(',', ''))
            elif column_type["type"] == "date":
                return pd.to_datetime(text).isoformat()
            return text
        except (ValueError, TypeError):
            return text

    def detect_tables(
        self,
        file_path: str,
        confidence_threshold: float = 0.5,
        min_row_count: int = 2,
        use_ml_detection: bool = True,
        extract_data: bool = True,
        enhance_image: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect and extract tables from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            confidence_threshold: Minimum confidence score for ML detections
            min_row_count: Minimum number of rows to consider as table
            use_ml_detection: Whether to use ML-based detection
            extract_data: Whether to extract table data
            enhance_image: Whether to enhance images for OCR
            
        Returns:
            List of detected tables with their properties and data
        """
        logger.debug(f"Detecting tables in PDF: {file_path}")
        
        try:
            # Open PDF
            doc = fitz.open(file_path)
            
            # Check if ML detection should be used
            if use_ml_detection:
                use_ml_detection = self._check_pdf_suitability(doc)
            
            all_tables = []
            
            # Process each page
            for page_num, page in enumerate(doc):
                logger.debug(f"Processing page {page_num + 1}")
                page_tables = []
                
                try:
                    # Detect tables
                    tables = super().detect_tables(
                        page,
                        confidence_threshold,
                        min_row_count,
                        use_ml_detection
                    )
                    
                    # Extract data if requested
                    if extract_data:
                        for table in tables:
                            table_data = self._extract_table_data(
                                page,
                                table,
                                enhance_image
                            )
                            table.update(table_data)
                    
                    page_tables.extend(tables)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
                
                # Add page numbers
                for table in page_tables:
                    table["page_number"] = page_num + 1
                
                all_tables.extend(page_tables)
            
            return all_tables
            
        finally:
            if 'doc' in locals():
                doc.close()

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Boolean indicating if file is valid
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"PDF file not found: {file_path}")
                return False
            
            # Check if file is a valid PDF
            try:
                doc = fitz.open(file_path)
                # Check if PDF has pages
                if doc.page_count == 0:
                    logger.error(f"PDF has no pages: {file_path}")
                    return False
                doc.close()
                return True
            except Exception:
                logger.error(f"Invalid PDF file: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False 