from typing import List, Dict, Any
from datetime import datetime
import logging
import torch
import numpy as np
import pytesseract
from PIL import Image
import cv2

from app.enums.analysis import AnalysisStatus
from app.schemas.table_analysis import (
    TableAnalysisDataExtractionResult,
    TableAnalysisStructureRecognitionResult,
    ExtractedTableData,
    ExtractedCell,
    MSATableDataExtractionParameters
)
from app.services.analysis.ml.table_analysis.utils.msa_model import msa_model_manager

logger = logging.getLogger(__name__)

class MSATableDataExtractor:
    """Microsoft Table Transformer (MSA) based table data extractor."""

    def __init__(self):
        self.model_version = None
        # Configure Tesseract
        self.tesseract_config = r'--oem 3 --psm 6'

    async def initialize(self, model_version: str = "latest") -> None:
        """Initialize the MSA model."""
        if self.model_version != model_version:
            await msa_model_manager.initialize_structure_model(model_version)
            self.model_version = model_version

    async def extract_data(
        self,
        document_path: str,
        parameters: MSATableDataExtractionParameters,
        structure_result: TableAnalysisStructureRecognitionResult
    ) -> TableAnalysisDataExtractionResult:
        """
        Extract table data using MSA.
        
        Args:
            document_path: Path to the document
            parameters: Data extraction parameters
            structure_result: Result from structure recognition step
            
        Returns:
            TableAnalysisDataExtractionResult containing extracted data
        """
        try:
            # Initialize model with specified version
            await self.initialize(parameters.model_version)

            # Load document
            doc = msa_model_manager.load_document(document_path)
            
            # Process each recognized table structure
            extracted_tables: List[ExtractedTableData] = []
            total_confidence = 0.0
            
            for table_structure in structure_result.recognized_structure:
                # Get page image
                page_num = int(table_structure.table_id.split("_")[1])
                image, scale_x, scale_y = msa_model_manager.get_page_image(doc, page_num)
                
                # Extract table data
                extracted_table = await self._extract_table_data(
                    image,
                    table_structure,
                    scale_x,
                    scale_y,
                    parameters
                )
                extracted_tables.append(extracted_table)
                total_confidence += extracted_table.confidence

            # Calculate average confidence
            avg_confidence = total_confidence / len(extracted_tables) if extracted_tables else 0.0

            # Create result
            result = TableAnalysisDataExtractionResult(
                status=AnalysisStatus.COMPLETED,
                extracted_table=extracted_tables[0] if extracted_tables else None,  # TODO: Handle multiple tables
                average_confidence=avg_confidence,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                accuracy=avg_confidence
            )

            return result

        except Exception as e:
            logger.error(f"Error in MSA data extraction: {str(e)}")
            return TableAnalysisDataExtractionResult(
                status=AnalysisStatus.FAILED,
                error=str(e),
                extracted_table=None,
                average_confidence=0.0,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )

    async def _extract_table_data(
        self,
        page_image: np.ndarray,
        table_structure: Any,
        scale_x: float,
        scale_y: float,
        parameters: MSATableDataExtractionParameters
    ) -> ExtractedTableData:
        """Extract data from a single table."""
        extracted_cells: List[ExtractedCell] = []
        total_confidence = 0.0
        max_attempts = parameters.max_attempts or 1
        
        for cell in table_structure.cells:
            # Extract cell image
            cell_coords = {
                "x": cell.bounding_box.x * scale_x,
                "y": cell.bounding_box.y * scale_y,
                "width": cell.bounding_box.width * scale_x,
                "height": cell.bounding_box.height * scale_y
            }
            cell_image = msa_model_manager.extract_cell_image(page_image, cell_coords)
            
            # Preprocess cell image
            if parameters.preprocess_image:
                cell_image = self._preprocess_cell_image(cell_image, parameters)
            
            # Extract text with multiple attempts if needed
            cell_text = None
            best_confidence = 0.0
            
            for attempt in range(max_attempts):
                # Apply different preprocessing on each attempt
                if attempt > 0:
                    cell_image = self._apply_additional_preprocessing(cell_image, attempt)
                
                # Extract text using Tesseract
                try:
                    text = pytesseract.image_to_data(
                        Image.fromarray(cell_image),
                        config=self.tesseract_config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Get text and confidence
                    conf_values = [float(c) / 100.0 for c in text['conf'] if c != '-1']
                    if conf_values:
                        confidence = sum(conf_values) / len(conf_values)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            cell_text = ' '.join(
                                word for word in text['text']
                                if word.strip()
                            )
                
                except Exception as e:
                    logger.warning(f"Error in text extraction attempt {attempt + 1}: {str(e)}")
                    continue
                
                # Break if we got good confidence
                if best_confidence >= parameters.confidence_threshold:
                    break
            
            # Create extracted cell
            extracted_cell = ExtractedCell(
                row=cell.row,
                column=cell.column,
                data=cell_text if cell_text else "",
                confidence=best_confidence
            )
            
            extracted_cells.append(extracted_cell)
            total_confidence += best_confidence
        
        # Calculate average confidence for the table
        avg_confidence = total_confidence / len(extracted_cells) if extracted_cells else 0.0
        
        extracted_table = ExtractedTableData(
            table_id=table_structure.table_id,
            cells=extracted_cells,
            confidence=avg_confidence
        )
        
        return extracted_table

    def _preprocess_cell_image(
        self,
        cell_image: np.ndarray,
        parameters: MSATableDataExtractionParameters
    ) -> np.ndarray:
        """Preprocess cell image for better text extraction."""
        try:
            # Deskew image
            cell_image = msa_model_manager.deskew_image(cell_image)
            
            # Convert to grayscale
            if len(cell_image.shape) == 3:
                cell_image = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            cell_image = cv2.adaptiveThreshold(
                cell_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Remove noise
            cell_image = cv2.medianBlur(cell_image, 3)
            
            # Add padding
            cell_image = cv2.copyMakeBorder(
                cell_image,
                10, 10, 10, 10,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            
            return cell_image
            
        except Exception as e:
            logger.warning(f"Error in cell image preprocessing: {str(e)}")
            return cell_image

    def _apply_additional_preprocessing(
        self,
        cell_image: np.ndarray,
        attempt: int
    ) -> np.ndarray:
        """Apply additional preprocessing based on attempt number."""
        try:
            if attempt == 1:
                # Increase contrast
                cell_image = cv2.convertScaleAbs(
                    cell_image,
                    alpha=1.5,
                    beta=0
                )
            elif attempt == 2:
                # Apply different thresholding
                _, cell_image = cv2.threshold(
                    cell_image,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            elif attempt == 3:
                # Apply erosion
                kernel = np.ones((2,2), np.uint8)
                cell_image = cv2.erode(cell_image, kernel, iterations=1)
            
            return cell_image
            
        except Exception as e:
            logger.warning(f"Error in additional preprocessing: {str(e)}")
            return cell_image
