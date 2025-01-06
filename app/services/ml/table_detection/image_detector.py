from typing import List, Dict, Any, Tuple
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import cv2

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
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect tables in an image.
        
        Args:
            file_path: Path to the image file
            confidence_threshold: Minimum confidence score for detections
            enhance_image: Whether to apply image enhancement
            
        Returns:
            List of detected tables with their properties
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
                tables.append(table)
            
            # Sort tables by position (top to bottom)
            tables.sort(key=lambda x: x["coordinates"]["y1"])
            
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