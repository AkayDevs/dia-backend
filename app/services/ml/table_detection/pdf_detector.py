from typing import List, Dict, Any
import logging
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import cv2
from collections import defaultdict

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

    def detect_tables(
        self, 
        file_path: str, 
        confidence_threshold: float = 0.5,
        min_row_count: int = 2,
        use_ml_detection: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect tables in a PDF document using hybrid approach.
        
        Args:
            file_path: Path to the PDF file
            confidence_threshold: Minimum confidence score for ML detections
            min_row_count: Minimum number of rows to consider as table
            use_ml_detection: Whether to use ML-based detection in addition to rule-based
            
        Returns:
            List of detected tables with their properties
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
                    # 1. Rule-based detection using lines
                    lines = self._detect_lines(page)
                    rule_based_tables = self._find_intersections(lines)
                    page_tables.extend(rule_based_tables)
                    
                    # 2. ML-based detection (if enabled and suitable)
                    if use_ml_detection:
                        try:
                            # Convert page to image with higher resolution for better detection
                            zoom = 2  # Increase if needed for better detection
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page.get_pixmap(matrix=mat, alpha=False)
                            
                            # Convert to PIL Image
                            try:
                                img_data = pix.samples
                                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                                
                                # Save temporary image
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format='PNG')
                                img_bytes.seek(0)
                                
                                # Detect tables using ML
                                ml_tables = self.image_detector.detect_tables(
                                    img_bytes,
                                    confidence_threshold=confidence_threshold
                                )
                                page_tables.extend(ml_tables)
                                
                            except ValueError as e:
                                logger.warning(f"Failed to convert page {page_num + 1} to image: {str(e)}")
                                logger.warning("Falling back to rule-based detection only")
                                
                        except Exception as e:
                            logger.warning(f"ML detection failed for page {page_num + 1}: {str(e)}")
                            logger.warning("Continuing with rule-based detection only")
                    
                    # 3. Merge overlapping detections
                    merged_tables = self._merge_overlapping_tables(page_tables)
                    
                    # 4. Filter small tables
                    filtered_tables = [
                        table for table in merged_tables
                        if table["height"] > 20 and table["width"] > 50  # Minimum size
                    ]
                    
                    # Add page number and sort by position
                    for table in filtered_tables:
                        table["page_number"] = page_num + 1
                        
                    all_tables.extend(filtered_tables)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    logger.error("Skipping this page and continuing with next")
                    continue
            
            if not all_tables:
                logger.warning("No tables detected in the document")
            else:
                logger.info(f"Detected {len(all_tables)} tables in PDF")
            
            return all_tables
            
        except Exception as e:
            logger.error(f"PDF table detection failed: {str(e)}")
            raise
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