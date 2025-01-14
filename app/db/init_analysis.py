from sqlalchemy.orm import Session
import logging
from typing import List

from app.crud import crud_analysis
from app.db.models.analysis import AnalysisTypeEnum, AnalysisStepEnum
from app.db.models.document import DocumentType
from app.schemas.analysis import (
    AnalysisTypeCreate,
    AnalysisStepCreate,
    AlgorithmCreate,
    Parameter
)

logger = logging.getLogger(__name__)

def init_analysis_db(db: Session) -> None:
    """Initialize the analysis database with default types, steps, and algorithms."""
    try:
        # Check if table analysis type already exists
        table_analysis = crud_analysis.analysis_type.get_by_name(db, name=AnalysisTypeEnum.TABLE_ANALYSIS)
        if not table_analysis:
            logger.info("Creating Table Analysis type...")
            table_analysis = crud_analysis.analysis_type.create(
                db,
                obj_in=AnalysisTypeCreate(
                    name=AnalysisTypeEnum.TABLE_ANALYSIS,
                    description="Detect and extract data from tables in documents",
                    supported_document_types=[DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOCX]
                )
            )

            # Create Table Analysis Steps
            table_detection_step = crud_analysis.analysis_step.create(
                db,
                obj_in=AnalysisStepCreate(
                    name=AnalysisStepEnum.TABLE_DETECTION,
                    description="Detect table regions in the document",
                    order=1,
                    analysis_type_id=str(table_analysis.id),
                    base_parameters=[
                        Parameter(
                            name="page_range",
                            description="Range of pages to process (e.g., '1-5' or '1,3,5')",
                            type="string",
                            required=False,
                            default="all"
                        ),
                        Parameter(
                            name="max_tables",
                            description="Maximum number of tables to detect per page",
                            type="integer",
                            required=False,
                            default=10,
                            min_value=1,
                            max_value=50
                        )
                    ]
                )
            )

            structure_recognition_step = crud_analysis.analysis_step.create(
                db,
                obj_in=AnalysisStepCreate(
                    name=AnalysisStepEnum.TABLE_STRUCTURE_RECOGNITION,
                    description="Recognize the structure of detected tables",
                    order=2,
                    analysis_type_id=str(table_analysis.id),
                    base_parameters=[
                        Parameter(
                            name="merge_threshold",
                            description="Threshold for merging cells",
                            type="float",
                            required=False,
                            default=0.5,
                            min_value=0.1,
                            max_value=1.0
                        )
                    ]
                )
            )

            data_extraction_step = crud_analysis.analysis_step.create(
                db,
                obj_in=AnalysisStepCreate(
                    name=AnalysisStepEnum.TABLE_DATA_EXTRACTION,
                    description="Extract data from recognized table structures",
                    order=3,
                    analysis_type_id=str(table_analysis.id),
                    base_parameters=[
                        Parameter(
                            name="ocr_engine",
                            description="OCR engine to use",
                            type="string",
                            required=False,
                            default="tesseract",
                            allowed_values=["tesseract", "azure", "google"]
                        )
                    ]
                )
            )

            # Create Table Detection Algorithms
            crud_analysis.algorithm.create(
                db,
                obj_in=AlgorithmCreate(
                    name="TableDetectionBasic",
                    description="Basic table detection using OpenCV contour detection",
                    version="1.0.0",
                    step_id=str(table_detection_step.id),
                    supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
                    parameters=[
                        Parameter(
                            name="min_table_size",
                            description="Minimum table size as percentage of page size",
                            type="float",
                            required=False,
                            default=0.05,
                            min_value=0.01,
                            max_value=1.0
                        )
                    ]
                )
            )

            crud_analysis.algorithm.create(
                db,
                obj_in=AlgorithmCreate(
                    name="TableDetectionML",
                    description="Table detection using a pre-trained deep learning model",
                    version="1.0.0",
                    step_id=str(table_detection_step.id),
                    supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
                    parameters=[
                        Parameter(
                            name="confidence_threshold",
                            description="Minimum confidence score for table detection",
                            type="float",
                            required=False,
                            default=0.5,
                            min_value=0.1,
                            max_value=1.0
                        )
                    ]
                )
            )

        # Check if text analysis type already exists
        text_analysis = crud_analysis.analysis_type.get_by_name(db, name=AnalysisTypeEnum.TEXT_ANALYSIS)
        if not text_analysis:
            logger.info("Creating Text Analysis type...")
            text_analysis = crud_analysis.analysis_type.create(
                db,
                obj_in=AnalysisTypeCreate(
                    name=AnalysisTypeEnum.TEXT_ANALYSIS,
                    description="Extract and analyze text from documents",
                    supported_document_types=[DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOCX]
                )
            )

        # Check if template conversion type already exists
        template_conversion = crud_analysis.analysis_type.get_by_name(db, name=AnalysisTypeEnum.TEMPLATE_CONVERSION)
        if not template_conversion:
            logger.info("Creating Template Conversion type...")
            template_conversion = crud_analysis.analysis_type.create(
                db,
                obj_in=AnalysisTypeCreate(
                    name=AnalysisTypeEnum.TEMPLATE_CONVERSION,
                    description="Convert documents into structured templates",
                    supported_document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.XLSX]
                )
            )

        logger.info("Analysis database initialization completed")

    except Exception as e:
        logger.error(f"Error initializing analysis database: {str(e)}")
        raise 