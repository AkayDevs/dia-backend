from typing import List, Dict, Any
from datetime import datetime
import logging
import torch
import numpy as np

from app.enums.analysis import AnalysisStatus
from app.schemas.table_analysis import (
    TableAnalysisDetectionResult,
    DetectedTable,
    BoundingBox,
    MSATableDetectionParameters
)
from app.services.analysis.ml.table_analysis.utils.msa_model import msa_model_manager

logger = logging.getLogger(__name__)

class MSATableDetector:
    """Microsoft Table Transformer (MSA) based table detector."""

    def __init__(self):
        self.model_version = None

    async def initialize(self, model_version: str = "latest") -> None:
        """Initialize the MSA model."""
        if self.model_version != model_version:
            await msa_model_manager.initialize_detection_model(model_version)
            self.model_version = model_version

    async def detect_tables(
        self,
        document_path: str,
        parameters: MSATableDetectionParameters
    ) -> TableAnalysisDetectionResult:
        """
        Detect tables in a document using MSA.
        
        Args:
            document_path: Path to the document
            parameters: Detection parameters
            
        Returns:
            TableAnalysisDetectionResult containing detected tables
        """
        try:
            # Initialize model with specified version
            await self.initialize(parameters.model_version)

            # Load document
            doc = msa_model_manager.load_document(document_path)
            
            # Detect tables
            detected_tables: List[DetectedTable] = []
            page_numbers = self._parse_page_range(parameters.page_range, doc.page_count)
            
            for page_num in page_numbers:
                # Get page image
                image, scale_x, scale_y = msa_model_manager.get_page_image(doc, page_num)
                
                # Preprocess image if enabled
                if parameters.refine_edges:
                    image = msa_model_manager.preprocess_image(image)
                
                # Detect tables on page
                tables_on_page = await self._detect_tables_on_page(
                    image,
                    scale_x,
                    scale_y,
                    page_num,
                    parameters
                )
                detected_tables.extend(tables_on_page)

            # Filter results based on parameters
            filtered_tables = self._filter_results(
                detected_tables,
                parameters
            )

            # Calculate accuracy based on confidence scores
            accuracy = sum(table.confidence for table in filtered_tables) / len(filtered_tables) if filtered_tables else 0.0

            # Create result
            result = TableAnalysisDetectionResult(
                status=AnalysisStatus.COMPLETED,
                detected_tables=filtered_tables,
                total_tables=len(filtered_tables),
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                accuracy=accuracy
            )

            return result

        except Exception as e:
            logger.error(f"Error in MSA table detection: {str(e)}")
            return TableAnalysisDetectionResult(
                status=AnalysisStatus.FAILED,
                error=str(e),
                detected_tables=[],
                total_tables=0,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )

    def _parse_page_range(self, page_range: str, total_pages: int) -> List[int]:
        """Parse page range string into list of page numbers."""
        if page_range == "all":
            return list(range(1, total_pages + 1))

        pages = set()
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                end = min(end, total_pages)
                pages.update(range(start, end + 1))
            else:
                page = int(part)
                if page <= total_pages:
                    pages.add(page)
        return sorted(list(pages))

    async def _detect_tables_on_page(
        self,
        image: np.ndarray,
        scale_x: float,
        scale_y: float,
        page_num: int,
        parameters: MSATableDetectionParameters
    ) -> List[DetectedTable]:
        """Detect tables on a specific page using MSA model."""
        # Prepare image for model
        inputs = msa_model_manager.feature_extractor(
            images=image,
            return_tensors="pt"
        ).to(msa_model_manager.device)
        
        # Run model inference
        with torch.no_grad():
            outputs = msa_model_manager.detection_model(**inputs)
        
        # Process predictions
        tables = []
        probas = outputs.logits.softmax(-1)[0, :, 1].cpu()
        keep = probas > parameters.confidence_threshold
        
        # Convert predictions to bounding boxes
        boxes = outputs.pred_boxes[0, keep].cpu()
        
        for idx, (box, score) in enumerate(zip(boxes, probas[keep])):
            # Convert normalized coordinates to actual coordinates
            x, y, w, h = box.tolist()
            x = x * image.shape[1]
            y = y * image.shape[0]
            w = w * image.shape[1]
            h = h * image.shape[0]
            
            # Rescale coordinates to original document space
            coords = msa_model_manager.rescale_coordinates(
                {"x": x, "y": y, "width": w, "height": h},
                scale_x,
                scale_y
            )
            
            table = DetectedTable(
                table_id=f"table_{page_num}_{idx + 1}",
                bounding_box=BoundingBox(**coords),
                confidence=float(score)
            )
            tables.append(table)
        
        return tables

    def _filter_results(
        self,
        tables: List[DetectedTable],
        parameters: MSATableDetectionParameters
    ) -> List[DetectedTable]:
        """Filter detected tables based on parameters."""
        filtered = []
        
        for table in tables:
            # Apply confidence threshold
            if table.confidence < parameters.confidence_threshold:
                continue

            # Apply size threshold
            table_area = table.bounding_box.width * table.bounding_box.height
            if parameters.min_table_size and table_area < parameters.min_table_size:
                continue

            filtered.append(table)

            # Apply max tables limit
            if parameters.max_tables and len(filtered) >= parameters.max_tables:
                break

        return filtered
