from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
import os
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
from PIL import Image
import logging
from app.services.analysis.configs.base import BaseAlgorithm
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase, AlgorithmParameter, AlgorithmParameterValue
from app.enums.document import DocumentType
from app.schemas.analysis.results.table_structure import (
    TableStructureResult,
    PageTableStructureResult,
    TableStructure,
    Cell
)
from app.schemas.analysis.results.table_shared import BoundingBox, Confidence, PageInfo

logger = logging.getLogger(__name__)

class MSATableStructureAlgorithm(BaseAlgorithm):
    """Microsoft Table Transformer (DETR) based table structure recognition"""
    
    _feature_extractor: DetrFeatureExtractor
    _model: TableTransformerForObjectDetection
    
    def get_info(self) -> AlgorithmDefinitionBase:
        return AlgorithmDefinitionBase(
            code="msa_structure",
            name="Microsoft Table Transformer Structure",
            version="1.0.0",
            description="Table structure recognition using Microsoft's Table Transformer (DETR) model",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOCX],
            parameters=[
                AlgorithmParameter(
                    name="confidence_threshold",
                    description="Minimum confidence score for cell detection",
                    type="float",
                    required=False,
                    default=0.5,
                    constraints={
                        "min": 0.1,
                        "max": 1.0
                    }
                ),
                AlgorithmParameter(
                    name="model_name",
                    description="Hugging Face model name for Table Transformer or 'fine-tuned' to use fine-tuned model",
                    type="string",
                    required=False,
                    default="microsoft/table-transformer-structure-recognition"
                ),
                AlgorithmParameter(
                    name="fine_tuned_model_path",
                    description="Path to the fine-tuned model directory (relative to models directory)",
                    type="string",
                    required=False,
                    default=None
                ),
                AlgorithmParameter(
                    name="use_fine_tuned",
                    description="Whether to use the fine-tuned model instead of the base model",
                    type="boolean",
                    required=False,
                    default=False
                )
            ],
            implementation_path="app.services.analysis.configs.definitions.table_analysis.algorithms.msa_structure.MSATableStructureAlgorithm",
            is_active=True
        )

    def get_default_parameters(self) -> List[AlgorithmParameterValue]:
        """Get default parameters for the algorithm"""
        return [
            AlgorithmParameterValue(
                name="confidence_threshold",
                value=0.5
            ),
            AlgorithmParameterValue(
                name="model_name",
                value="microsoft/table-transformer-structure-recognition"
            ),
            AlgorithmParameterValue(
                name="fine_tuned_model_path",
                value=None
            ),
            AlgorithmParameterValue(
                name="use_fine_tuned",
                value=False
            )
        ]
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        try:
            import torch
            from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(f"Required dependency not found: {str(e)}")
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if "document_path" not in input_data:
            raise ValueError("Document path not provided in input data")
        if "tables" not in input_data:
            raise ValueError("Table detection results not provided in input data")
    
    def _get_model_path(self, parameters: Dict[str, Any]) -> str:
        """Get the appropriate model path based on parameters"""
        use_fine_tuned = parameters.get("use_fine_tuned", False)
        default_model = "microsoft/table-transformer-structure-recognition"
        
        if use_fine_tuned:
            try:
                fine_tuned_path = parameters.get("fine_tuned_model_path")
                if not fine_tuned_path:
                    logger.warning("Fine-tuned model path not provided, falling back to pre-trained model")
                    return default_model
                
                # Construct absolute path to the fine-tuned model
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                models_dir = os.path.join(base_dir, "models")
                model_path = os.path.join(models_dir, fine_tuned_path)
                
                if not os.path.exists(model_path):
                    logger.warning(f"Fine-tuned model not found at {model_path}, falling back to pre-trained model")
                    return default_model
                
                return model_path
            except Exception as e:
                logger.error(f"Error loading fine-tuned model: {str(e)}, falling back to pre-trained model")
                return default_model
        
        return parameters.get("model_name", default_model)

    def _process_cell_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        confidence_threshold: float,
        image_size: Tuple[int, int]
    ) -> Tuple[List[Cell], int, int]:
        """Process model predictions to create cell objects"""
        height, width = image_size
        
        # Get model's label mapping
        id2label = self._model.config.id2label
        logger.debug(f"Model label mapping: {id2label}")

        results = self._feature_extractor.post_process_object_detection(
            outputs, 
            threshold=confidence_threshold,
            target_sizes=[(height, width)]
        )[0]
        
        # Initialize counters for table dimensions
        max_row = 0
        max_col = 0
        cells = []
        
        # Process predictions
        scores = results["scores"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        # First pass: Count rows and columns
        for score, label in zip(scores, labels):
            if score < confidence_threshold:
                continue
            
            label_name = id2label[label.item()].lower()
            if 'column' in label_name:
                max_col += 1
            if 'row' in label_name:
                max_row += 1
        
        # Ensure minimum dimensions
        max_row = max(max_row, 1)
        max_col = max(max_col, 1)
        
        # Second pass: Process all cells
        for score, box, label in zip(scores, boxes, labels):
            if score < confidence_threshold:
                continue
            
            label_name = id2label[label.item()].lower()
            logger.debug(f"Processing {label_name} with confidence {score:.3f}")
            
            # Create cell object
            cell = Cell(
                bbox=BoundingBox(
                    x1=int(box[0]),
                    y1=int(box[1]),
                    x2=int(box[2]),
                    y2=int(box[3])
                ),
                row_span=2 if 'spanning' in label_name else 1,
                col_span=2 if 'spanning' in label_name else 1,
                is_header='header' in label_name,
                confidence=Confidence(
                    score=float(score),
                    method=f"table_transformer_{label_name}"
                )
            )
            cells.append(cell)
            
            logger.debug(f"Added cell: {label_name}, bbox={box}, is_header={'header' in label_name}")
        
        return cells, max_row, max_col

    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table structure recognition"""
        try:
            # Get parameters
            confidence_threshold = parameters.get("confidence_threshold", 0.5)
            
            # Get appropriate model path
            model_path = self._get_model_path(parameters)
            
            # Initialize model and processor
            self._feature_extractor = DetrFeatureExtractor()
            self._model = TableTransformerForObjectDetection.from_pretrained(model_path)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            self._model.eval()

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
            
            # Get table detection results
            table_results = previous_results.get("table_analysis.table_detection", {})
            if not table_results:
                raise ValueError("No table detection results found in previous steps")
            
            # Process each page
            final_results = []
            total_tables = 0
            
            for page_idx, page_result in enumerate(table_results.get("results", [])):
                # Get corresponding document page
                doc_page = next((p for p in document_pages.pages if p.page_number == page_idx + 1), None)
                if not doc_page:
                    continue

                # Load image from the page's image_url
                image_path = doc_page.image_url.lstrip('/')
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                page_tables = []
                page_info = page_result["page_info"]
                
                for table_loc in page_result["tables"]:
                    # Extract table region
                    bbox = table_loc["bbox"]
                    table_img = image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
                    
                    # Prepare image for model
                    encoding = self._feature_extractor(table_img, return_tensors="pt")
                    encoding = {k: v.to(device) for k, v in encoding.items()}
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = self._model(**encoding)

                    # Process predictions
                    cells, num_rows, num_cols = self._process_cell_predictions(
                        outputs,
                        confidence_threshold,
                        table_img.size[::-1]  # Convert (width, height) to (height, width)
                    )
                    
                    # Create table structure
                    table_structure = TableStructure(
                        bbox=BoundingBox(**bbox),
                        cells=cells,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        confidence=Confidence(
                            score=float(table_loc["confidence"]["score"]),
                            method="table_transformer"
                        )
                    )
                    
                    page_tables.append(table_structure)
                    total_tables += 1
                
                # Create page result
                if page_tables:
                    page_result = PageTableStructureResult(
                        page_info=PageInfo(**page_info),
                        tables=page_tables,
                        processing_info={
                            "model_path": model_path,
                            "model_type": "fine-tuned" if parameters.get("use_fine_tuned") else "base",
                            "device": str(device),
                            "confidence_threshold": confidence_threshold
                        }
                    )
                    final_results.append(page_result)
            
            # Create final result
            final_result = TableStructureResult(
                results=final_results,
                total_pages_processed=len(final_results),
                total_tables_processed=total_tables,
                metadata={
                    "algorithm": "msa_structure",
                    "version": "1.0.0",
                    "parameters": parameters,
                    "execution_info": {
                        "device": str(device),
                        "model_type": "fine-tuned" if parameters.get("use_fine_tuned") else "base"
                    }
                }
            )
            
            return final_result.dict()
            
        except Exception as e:
            raise RuntimeError(f"Table structure detection failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass 