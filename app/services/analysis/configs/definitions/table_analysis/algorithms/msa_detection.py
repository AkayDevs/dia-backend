from typing import Dict, Any, List
import cv2
import numpy as np
import torch
import os
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
from app.services.analysis.configs.base import BaseAlgorithm
from app.schemas.analysis.configs.algorithms import AlgorithmDefinitionBase, AlgorithmParameter, AlgorithmParameterValue
from app.enums.document import DocumentType
from app.schemas.analysis.results.table_detection import TableDetectionResult, PageTableDetectionResult, TableLocation
from app.schemas.analysis.results.table_shared import BoundingBox, Confidence, PageInfo

class MSATableDetectionAlgorithm(BaseAlgorithm):
    """Microsoft Table Transformer (DETR) based table detection"""
    
    def get_info(self) -> AlgorithmDefinitionBase:
        return AlgorithmDefinitionBase(
            code="msa_detection",
            name="Microsoft Table Transformer Detection",
            version="1.0.0",
            description="Table detection using Microsoft's Table Transformer (DETR) model",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
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
        
        if use_fine_tuned:
            fine_tuned_path = parameters.get("fine_tuned_model_path")
            if not fine_tuned_path:
                raise ValueError("Fine-tuned model path not provided but use_fine_tuned is True")
            
            # Construct absolute path to the fine-tuned model
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, "models")
            model_path = os.path.join(models_dir, fine_tuned_path)
            
            if not os.path.exists(model_path):
                raise ValueError(f"Fine-tuned model not found at {model_path}")
            
            return model_path
        else:
            return parameters.get("model_name", "microsoft/table-transformer-detection")
    
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
            
            # Load image
            image = Image.open(document_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Initialize model and processor
            processor = DetrImageProcessor.from_pretrained(model_path)
            model = TableTransformerForObjectDetection.from_pretrained(model_path)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Prepare image for model
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to same device as model
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert outputs to COCO format
            results = processor.post_process_object_detection(
                outputs,
                threshold=confidence_threshold,
                target_sizes=[(image.size[1], image.size[0])]
            )[0]
            
            # Format results using proper schema types
            tables = []
            scores = results["scores"]
            boxes = results["boxes"]
            
            # Process all detected tables up to max_tables
            for score, box in zip(scores[:max_tables], boxes[:max_tables]):
                x1, y1, x2, y2 = box.tolist()
                tables.append(
                    TableLocation(
                        bbox=BoundingBox(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2)
                        ),
                        confidence=Confidence(
                            score=float(score),
                            method="table_transformer"
                        ),
                        table_type="table"  # Table Transformer doesn't distinguish between table types
                    )
                )
            
            # Create page result using schema
            page_result = PageTableDetectionResult(
                page_info=PageInfo(
                    page_number=1,  # Single page processing
                    width=image.size[0],
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
            
            # Create final result using schema
            final_result = TableDetectionResult(
                results=[page_result],
                total_pages_processed=1,
                total_tables_found=len(tables),
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
