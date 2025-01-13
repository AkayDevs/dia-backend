from typing import Optional, List, Dict, Any, Tuple
import os
import torch
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2
from transformers import (
    TableTransformerForObjectDetection,
    TableTransformerForTokenClassification,
    TableTransformerConfig,
    DetrFeatureExtractor
)

class MSAModelManager:
    """Manages Microsoft Table Transformer models and common operations."""
    
    DETECTION_MODEL = "microsoft/table-transformer-detection"
    STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition"
    DETECTION_THRESHOLD = 0.7
    
    def __init__(self):
        self.detection_model = None
        self.structure_model = None
        self.feature_extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def initialize_detection_model(self, model_version: str = "latest") -> None:
        """Initialize table detection model."""
        if self.detection_model is None:
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(
                self.DETECTION_MODEL,
                revision=model_version if model_version != "latest" else None
            )
            self.detection_model.to(self.device)
            self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.DETECTION_MODEL)

    async def initialize_structure_model(self, model_version: str = "latest") -> None:
        """Initialize structure recognition model."""
        if self.structure_model is None:
            self.structure_model = TableTransformerForTokenClassification.from_pretrained(
                self.STRUCTURE_MODEL,
                revision=model_version if model_version != "latest" else None
            )
            self.structure_model.to(self.device)

    def load_document(self, document_path: str) -> fitz.Document:
        """Load document using PyMuPDF."""
        return fitz.open(document_path)

    def get_page_image(
        self,
        doc: fitz.Document,
        page_num: int,
        zoom: int = 2
    ) -> Tuple[np.ndarray, float, float]:
        """Get page image and scaling factors."""
        page = doc[page_num - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height,
            pix.width,
            pix.n
        )
        
        # Calculate scaling factors
        scale_x = pix.width / page.rect.width
        scale_y = pix.height / page.rect.height
        
        return img, scale_x, scale_y

    def preprocess_image(
        self,
        image: np.ndarray,
        enhance_contrast: bool = True,
        denoise: bool = True
    ) -> np.ndarray:
        """Preprocess image for better detection."""
        if enhance_contrast:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl,a,b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        if denoise:
            # Apply denoising
            image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                10,
                10,
                7,
                21
            )

        return image

    def extract_cell_image(
        self,
        page_image: np.ndarray,
        bbox: Dict[str, float]
    ) -> np.ndarray:
        """Extract cell image from page using bounding box."""
        x1, y1 = int(bbox["x"]), int(bbox["y"])
        x2 = int(x1 + bbox["width"])
        y2 = int(y1 + bbox["height"])
        
        return page_image[y1:y2, x1:x2]

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image using OpenCV."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # Calculate skew angle
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated

    @staticmethod
    def rescale_coordinates(
        coords: Dict[str, float],
        scale_x: float,
        scale_y: float
    ) -> Dict[str, float]:
        """Rescale coordinates back to original document space."""
        return {
            "x": coords["x"] / scale_x,
            "y": coords["y"] / scale_y,
            "width": coords["width"] / scale_x,
            "height": coords["height"] / scale_y
        }

# Create singleton instance
msa_model_manager = MSAModelManager() 