from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import cv2
import pytesseract
import pandas as pd
from collections import defaultdict

from app.services.ml.base import BaseTableDetector

logger = logging.getLogger(__name__)

class ImageTableDetector(BaseTableDetector):
    """Service for detecting tables in images using Microsoft Table Transformer."""
    
    def __init__(self):
        """Initialize the table detection model and processor."""
        logger.debug("Initializing Image Table Detection Service")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        try:
            self.processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            ).to(self.device)
            self.model.eval()
            logger.info("Image Table Detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Image Table Detection model: {str(e)}")
            raise
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        return ["png", "jpg", "jpeg", "tiff", "bmp"]

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model inference.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple of processed image tensor and original image size
        """
        try:
            # Convert grayscale to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Get original size
            original_size = image.size
            
            # Prepare image for the model
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs, original_size
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better table detection.
        
        Args:
            image: PIL Image to enhance
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply adaptive thresholding
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Noise removal
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Image enhancement failed: {str(e)}")
            return image  # Return original if enhancement fails

    def detect_tables(
        self, 
        file_path: str, 
        confidence_threshold: float = 0.5,
        enhance_image: bool = True,
        extract_data: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect and extract tables from an image.
        
        Args:
            file_path: Path to the image file
            confidence_threshold: Minimum confidence score for detections
            enhance_image: Whether to apply image enhancement
            extract_data: Whether to extract table data
            
        Returns:
            List of detected tables with their properties and data
        """
        logger.debug(f"Detecting tables in image: {file_path}")
        
        try:
            # Load and optionally enhance image
            image = Image.open(file_path)
            if enhance_image:
                image = self.enhance_image(image)
            
            # Preprocess image
            inputs, original_size = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=confidence_threshold,
                target_sizes=[original_size]
            )[0]
            
            # Format detections
            tables = []
            for score, label, box in zip(
                results["scores"], 
                results["labels"], 
                results["boxes"]
            ):
                # Convert box coordinates to integers
                box = [int(x) for x in box.tolist()]
                
                table = {
                    "confidence": float(score),
                    "label": self.model.config.id2label[int(label)],
                    "coordinates": {
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3]
                    },
                    "width": box[2] - box[0],
                    "height": box[3] - box[1],
                    "area": (box[2] - box[0]) * (box[3] - box[1])
                }
                
                # Extract table data if requested
                if extract_data:
                    table_data = self._extract_table_data(image, table)
                    table.update(table_data)
                
                tables.append(table)
            
            # Sort tables by position (top to bottom)
            tables.sort(key=lambda x: x["coordinates"]["y1"])
            
            logger.info(f"Detected {len(tables)} tables in image")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            raise

    def _extract_table_data(
        self,
        image: Image.Image,
        table: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured data from a detected table.
        
        Args:
            image: Source image
            table: Detected table information with coordinates
            
        Returns:
            Dictionary containing structured table data
        """
        try:
            # Crop table region
            coords = table["coordinates"]
            table_region = image.crop((
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"]
            ))
            
            # Enhance image for OCR
            enhanced = self._enhance_image_for_ocr(table_region)
            
            # Perform OCR with table structure recognition
            ocr_data = pytesseract.image_to_data(
                enhanced,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform block of text
            )
            
            # Convert OCR data to text blocks
            text_blocks = []
            for i in range(len(ocr_data["text"])):
                if ocr_data["conf"][i] > 30:  # Filter low confidence results
                    text_blocks.append({
                        "text": ocr_data["text"][i],
                        "bbox": (
                            ocr_data["left"][i] + coords["x1"],
                            ocr_data["top"][i] + coords["y1"],
                            ocr_data["left"][i] + ocr_data["width"][i] + coords["x1"],
                            ocr_data["top"][i] + ocr_data["height"][i] + coords["y1"]
                        ),
                        "confidence": ocr_data["conf"][i],
                        "font_size": ocr_data["height"][i]  # Use height as font size approximation
                    })
            
            # Analyze structure
            structure = self._analyze_table_structure(text_blocks, table)
            
            # Extract and organize data
            data = self._organize_table_data(text_blocks, structure)
            
            return {
                "structure": structure,
                "data": data,
                "ocr_used": True
            }
            
        except Exception as e:
            logger.error(f"Failed to extract table data: {str(e)}")
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
        
        # Check for header indicators (font size comparison)
        if not first_row_blocks:
            return None
            
        avg_font_size = sum(block.get("font_size", 0) for block in text_blocks) / len(text_blocks)
        header_font_size = sum(block.get("font_size", 0) for block in first_row_blocks) / len(first_row_blocks)
        
        return 0 if header_font_size > avg_font_size * 1.1 else None

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
                    "confidence": min((block.get("confidence", 0) for block in cell), default=0)
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
                    ),
                    "confidence": min((block.get("confidence", 0) for block in cell), default=0)
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

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is suitable for table detection.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Boolean indicating if file is valid
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"Image file not found: {file_path}")
                return False
            
            # Check if file is an image
            try:
                with Image.open(file_path) as img:
                    img.verify()
                    # Check image size
                    if img.size[0] < 100 or img.size[1] < 100:
                        logger.error(f"Image too small: {file_path}")
                        return False
            except Exception:
                logger.error(f"Invalid image file: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False 