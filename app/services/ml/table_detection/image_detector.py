from typing import List, Dict, Any, Tuple
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import numpy as np
from PIL import Image
import logging
from pathlib import Path

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
            logger.info("Image Table Detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Image Table Detection model: {str(e)}")
            raise
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        return ["png", "jpg", "jpeg", "tiff", "bmp"]

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of processed image tensor and original image size
        """
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # Prepare image for the model
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs, original_size
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def detect_tables(
        self, 
        file_path: str, 
        confidence_threshold: float = 0.5,
        min_row_count: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect tables in an image.
        
        Args:
            file_path: Path to the image file
            confidence_threshold: Minimum confidence score for detections
            min_row_count: Minimum number of rows to consider a valid table
            
        Returns:
            List of detected tables with their properties
        """
        logger.debug(f"Detecting tables in image: {file_path}")
        
        try:
            # Preprocess image
            inputs, original_size = self.preprocess_image(file_path)
            
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
                table = {
                    "confidence": float(score),
                    "bbox": [float(x) for x in box],
                    "label": self.model.config.id2label[int(label)],
                    "coordinates": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                tables.append(table)
            
            logger.info(f"Detected {len(tables)} tables in image")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            raise

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
            except Exception:
                logger.error(f"Invalid image file: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False 