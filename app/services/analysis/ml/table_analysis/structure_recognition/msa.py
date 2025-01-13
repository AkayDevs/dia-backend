from typing import List, Dict, Any
from datetime import datetime
import logging
import torch
import numpy as np
from itertools import groupby
from operator import itemgetter

from app.enums.analysis import AnalysisStatus
from app.schemas.table_analysis import (
    TableAnalysisStructureRecognitionResult,
    TableAnalysisDetectionResult,
    RecognizedTableStructure,
    Cell,
    BoundingBox,
    MSATableStructureRecognitionParameters
)
from app.services.analysis.ml.table_analysis.utils.msa_model import msa_model_manager

logger = logging.getLogger(__name__)

class MSATableStructureRecognizer:
    """Microsoft Table Transformer (MSA) based table structure recognizer."""

    def __init__(self):
        self.model_version = None

    async def initialize(self, model_version: str = "latest") -> None:
        """Initialize the MSA model."""
        if self.model_version != model_version:
            await msa_model_manager.initialize_structure_model(model_version)
            self.model_version = model_version

    async def recognize_structure(
        self,
        document_path: str,
        parameters: MSATableStructureRecognitionParameters,
        detection_result: TableAnalysisDetectionResult
    ) -> TableAnalysisStructureRecognitionResult:
        """
        Recognize table structure using MSA.
        
        Args:
            document_path: Path to the document
            parameters: Structure recognition parameters
            detection_result: Result from table detection step
            
        Returns:
            TableAnalysisStructureRecognitionResult containing recognized structures
        """
        try:
            # Initialize model with specified version
            await self.initialize(parameters.model_version)

            # Load document
            doc = msa_model_manager.load_document(document_path)
            
            # Process each detected table
            recognized_structures: List[RecognizedTableStructure] = []
            total_confidence = 0.0
            
            for detected_table in detection_result.detected_tables:
                # Get page image
                page_num = int(detected_table.table_id.split("_")[1])
                image, scale_x, scale_y = msa_model_manager.get_page_image(doc, page_num)
                
                # Extract table region
                table_image = msa_model_manager.extract_cell_image(
                    image,
                    {
                        "x": detected_table.bounding_box.x * scale_x,
                        "y": detected_table.bounding_box.y * scale_y,
                        "width": detected_table.bounding_box.width * scale_x,
                        "height": detected_table.bounding_box.height * scale_y
                    }
                )
                
                # Recognize structure
                structure = await self._recognize_table_structure(
                    table_image,
                    detected_table,
                    scale_x,
                    scale_y,
                    parameters
                )
                
                # Refine boundaries if enabled
                if parameters.refine_boundaries:
                    structure = self._refine_boundaries(structure, parameters)
                
                recognized_structures.append(structure)
                total_confidence += structure.confidence

            # Calculate average confidence
            avg_confidence = total_confidence / len(recognized_structures) if recognized_structures else 0.0

            # Create result
            result = TableAnalysisStructureRecognitionResult(
                status=AnalysisStatus.COMPLETED,
                recognized_structure=recognized_structures,
                total_structures=len(recognized_structures),
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                accuracy=avg_confidence
            )

            return result

        except Exception as e:
            logger.error(f"Error in MSA structure recognition: {str(e)}")
            return TableAnalysisStructureRecognitionResult(
                status=AnalysisStatus.FAILED,
                error=str(e),
                recognized_structure=[],
                total_structures=0,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )

    async def _recognize_table_structure(
        self,
        table_image: np.ndarray,
        detected_table: Any,
        scale_x: float,
        scale_y: float,
        parameters: MSATableStructureRecognitionParameters
    ) -> RecognizedTableStructure:
        """Recognize structure of a single table using MSA model."""
        # Prepare image for model
        inputs = msa_model_manager.feature_extractor(
            images=table_image,
            return_tensors="pt"
        ).to(msa_model_manager.device)
        
        # Run model inference
        with torch.no_grad():
            outputs = msa_model_manager.structure_model(**inputs)
        
        # Process predictions
        logits = outputs.logits[0].cpu()
        probs = logits.softmax(-1)
        
        # Get cell predictions
        cell_probs = probs[:, :, 0]  # Cell class probabilities
        keep = cell_probs > parameters.confidence_threshold
        
        # Convert predictions to cells
        cells = []
        cell_positions = []
        
        for i, j in torch.nonzero(keep).tolist():
            # Get cell coordinates
            x1 = j * table_image.shape[1] / logits.shape[1]
            y1 = i * table_image.shape[0] / logits.shape[0]
            x2 = (j + 1) * table_image.shape[1] / logits.shape[1]
            y2 = (i + 1) * table_image.shape[0] / logits.shape[0]
            
            # Store cell position for structure analysis
            cell_positions.append({
                "row": i,
                "col": j,
                "coords": (x1, y1, x2, y2),
                "confidence": float(cell_probs[i, j])
            })
        
        # Analyze table structure
        rows, cols = self._analyze_table_structure(cell_positions)
        
        # Create cells with proper row/column spans
        for pos in cell_positions:
            # Calculate spans
            rowspan = 1
            colspan = 1
            
            # Check for merged cells
            for other in cell_positions:
                if other == pos:
                    continue
                
                # Check if cells overlap significantly
                overlap = self._calculate_overlap(
                    pos["coords"],
                    other["coords"]
                )
                
                if overlap > 0.5:  # Significant overlap
                    if other["row"] > pos["row"]:
                        rowspan = max(rowspan, other["row"] - pos["row"] + 1)
                    if other["col"] > pos["col"]:
                        colspan = max(colspan, other["col"] - pos["col"] + 1)
            
            # Convert coordinates back to original space
            x1, y1, x2, y2 = pos["coords"]
            coords = msa_model_manager.rescale_coordinates(
                {
                    "x": x1 + detected_table.bounding_box.x,
                    "y": y1 + detected_table.bounding_box.y,
                    "width": x2 - x1,
                    "height": y2 - y1
                },
                scale_x,
                scale_y
            )
            
            cell = Cell(
                row=pos["row"] + 1,  # Convert to 1-based indexing
                column=pos["col"] + 1,
                bounding_box=BoundingBox(**coords),
                rowspan=rowspan,
                colspan=colspan
            )
            cells.append(cell)
        
        # Calculate overall confidence
        confidence = sum(pos["confidence"] for pos in cell_positions) / len(cell_positions)
        
        structure = RecognizedTableStructure(
            table_id=detected_table.table_id,
            rows=rows,
            columns=cols,
            cells=cells,
            confidence=confidence
        )
        
        return structure

    def _analyze_table_structure(
        self,
        cell_positions: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """Analyze cell positions to determine table structure."""
        rows = max(pos["row"] for pos in cell_positions) + 1
        cols = max(pos["col"] for pos in cell_positions) + 1
        return rows, cols

    def _calculate_overlap(
        self,
        coords1: tuple[float, float, float, float],
        coords2: tuple[float, float, float, float]
    ) -> float:
        """Calculate overlap ratio between two cells."""
        x1, y1, x2, y2 = coords1
        x3, y3, x4, y4 = coords2
        
        # Calculate intersection
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        
        # Return overlap ratio
        return intersection / min(area1, area2)

    def _refine_boundaries(
        self,
        structure: RecognizedTableStructure,
        parameters: MSATableStructureRecognitionParameters
    ) -> RecognizedTableStructure:
        """Refine table boundaries if enabled in parameters."""
        if not parameters.refine_boundaries:
            return structure
            
        # Group cells by row
        cells_by_row = {}
        for cell in structure.cells:
            if cell.row not in cells_by_row:
                cells_by_row[cell.row] = []
            cells_by_row[cell.row].append(cell)
        
        # Adjust cell boundaries
        for row_cells in cells_by_row.values():
            # Sort cells by column
            row_cells.sort(key=lambda c: c.column)
            
            # Adjust horizontal boundaries
            for i in range(len(row_cells) - 1):
                current = row_cells[i]
                next_cell = row_cells[i + 1]
                
                # Calculate midpoint
                mid_x = (current.bounding_box.x + current.bounding_box.width +
                        next_cell.bounding_box.x) / 2
                
                # Adjust boundaries
                current.bounding_box.width = mid_x - current.bounding_box.x
                next_cell.bounding_box.width = (next_cell.bounding_box.x +
                                              next_cell.bounding_box.width - mid_x)
                next_cell.bounding_box.x = mid_x
        
        return structure
