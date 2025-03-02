from typing import Dict, Any, List
import cv2
import numpy as np
import torch
import os
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
from PIL import Image
from app.services.analysis.configs.base import BaseAlgorithm
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase, AlgorithmParameter, AlgorithmParameterValue
from app.enums.document import DocumentType
from app.schemas.analysis.results.table_detection import TableDetectionResult, PageTableDetectionResult, TableLocation
from app.schemas.analysis.results.table_shared import BoundingBox, Confidence, PageInfo
import logging

logger = logging.getLogger(__name__)

class MSATableDetectionAlgorithm(BaseAlgorithm):
    """Microsoft Table Transformer (DETR) based table detection"""

    _feature_extractor: DetrFeatureExtractor
    _model: TableTransformerForObjectDetection

    def get_info(self) -> AlgorithmDefinitionBase:
        return AlgorithmDefinitionBase(
            code="msa_detection",
            name="Microsoft Table Transformer Detection",
            version="1.0.0",
            description="Table detection using Microsoft's Table Transformer (DETR) model",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOCX],
            parameters=[
                AlgorithmParameter(
                    name="max_tables",
                    description="Maximum number of tables to detect per page (default: 1000 to detect all tables)",
                    type="integer",
                    required=False,
                    default=1000,
                    constraints={
                        "min": 1,
                        "max": 10000
                    }
                ),
                AlgorithmParameter(
                    name="confidence_threshold",
                    description="Minimum confidence score for table detection",
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
                    default="microsoft/table-transformer-detection"
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
            implementation_path="app.services.analysis.configs.definitions.table_analysis.algorithms.msa_detection.MSATableDetectionAlgorithm",
            is_active=True
        )

    def get_default_parameters(self) -> List[AlgorithmParameterValue]:
        """Get default parameters for the algorithm"""
        return [
            AlgorithmParameterValue(
                name="max_tables",
                value=1000
            ),
            AlgorithmParameterValue(
                name="confidence_threshold",
                value=0.5
            ),
            AlgorithmParameterValue(
                name="model_name",
                value="microsoft/table-transformer-detection"
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
            from transformers import DetrImageProcessor, TableTransformerForObjectDetection
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(f"Required dependency not found: {str(e)}")
    
    async def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data format"""
        if "document_path" not in input_data:
            raise ValueError("Document path not provided in input data")
    
    def _get_model_path(self, parameters: Dict[str, Any]) -> str:
        """Get the appropriate model path based on parameters"""
        use_fine_tuned = parameters.get("use_fine_tuned", False)
        default_model = "microsoft/table-transformer-detection"
        
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
    
    async def execute(
        self,
        document_path: str,
        parameters: Dict[str, Any],
        previous_results: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """Execute table detection using Table Transformer"""
        try:
            # Get parameters with defaults
            max_tables = parameters.get("max_tables", 1000)  # Default to 1000 to effectively detect all tables
            confidence_threshold = parameters.get("confidence_threshold", 0.5)
            
            # Get appropriate model path
            model_path = self._get_model_path(parameters)
            
            # Initialize model and processor
            self._feature_extractor = DetrFeatureExtractor()
            self._model = TableTransformerForObjectDetection.from_pretrained(model_path)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            self._model.eval()  # Set model to evaluation mode

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

            all_tables = []
            for page in document_pages.pages:
                # Load image from the page's image_url
                image_path = page.image_url.lstrip('/')
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Log image processing
                logger.debug(f"Processing page {page.page_number}, image size: {image.size}")
                
                # Prepare image for model
                encoding = self._feature_extractor(image, return_tensors="pt")
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self._model(**encoding)
                
                # Convert outputs to COCO format
                results = self._feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(page.height, page.width)])[0]
                
                # Log detection results
                logger.info(f"Page {page.page_number} detection results: {len(results['scores'])} tables found")
                
                # Format results for this page
                tables = []
                scores = results["scores"].cpu().numpy()  # Convert to numpy for easier handling
                boxes = results["boxes"].cpu().numpy()
                
                # Process all detected tables up to max_tables
                for score, box in zip(scores[:max_tables], boxes[:max_tables]):
                    logger.debug(f"Table detected - Score: {score:.3f}, Box: {box}")
                    tables.append(
                        TableLocation(
                            bbox=BoundingBox(
                                x1=int(box[0]),
                                y1=int(box[1]),
                                x2=int(box[2]),
                                y2=int(box[3])
                            ),
                            confidence=Confidence(
                                score=float(score),
                                method="table_transformer"
                            ),
                            table_type="table"  # Table Transformer doesn't distinguish between table types
                        )
                    )
                
                # Create page result
                page_result = PageTableDetectionResult(
                    page_info=PageInfo(
                        page_number=page.page_number,
                        width=image.size[0],  # Use actual image dimensions
                        height=image.size[1]
                    ),
                    tables=tables,
                    processing_info={
                        "model_path": model_path,
                        "model_type": "fine-tuned" if parameters.get("use_fine_tuned") else "base",
                        "device": str(device),
                        "confidence_threshold": confidence_threshold
                    }
                )
                all_tables.append(page_result)
            
            # Create final result using schema
            final_result = TableDetectionResult(
                results=all_tables,
                total_pages_processed=len(document_pages.pages),
                total_tables_found=sum(len(page.tables) for page in all_tables),
                metadata={
                    "algorithm": "msa_detection",
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
            raise RuntimeError(f"Table detection failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources"""
        pass
