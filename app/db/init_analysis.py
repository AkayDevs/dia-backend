from typing import List, Optional, Any
import logging
from sqlalchemy.orm import Session

from app.crud import crud_analysis
from app.schemas.analysis import (
    AnalysisTypeCreate,
    AnalysisStepCreate,
    AlgorithmCreate,
    Parameter
)
from app.db.models.analysis import AnalysisTypeEnum, AnalysisStepEnum
from app.db.models.document import DocumentType

logger = logging.getLogger(__name__)

def get_or_create_step(
    db: Session,
    name: AnalysisStepEnum,
    analysis_type_id: str,
    order: int,
    description: str,
    base_parameters: List[Parameter]
) -> Optional[Any]:
    """Get existing step or create a new one."""
    existing_step = crud_analysis.analysis_step.get_by_name(
        db, name=name, analysis_type_id=analysis_type_id
    )
    if existing_step:
        logger.info(f"Step {name} already exists")
        return existing_step

    logger.info(f"Creating step: {name}")
    return crud_analysis.analysis_step.create(
        db,
        obj_in=AnalysisStepCreate(
            name=name,
            description=description,
            order=order,
            analysis_type_id=analysis_type_id,
            base_parameters=base_parameters
        )
    )

def get_or_create_algorithm(
    db: Session,
    name: str,
    step_id: str,
    description: str,
    version: str,
    supported_document_types: List[DocumentType],
    parameters: List[Parameter]
) -> Optional[Any]:
    """Get existing algorithm or create a new one."""
    existing_algorithm = crud_analysis.algorithm.get_by_name_and_version(
        db, name=name, version=version
    )
    if existing_algorithm and existing_algorithm.step_id == step_id:
        logger.info(f"Algorithm {name} v{version} already exists for step {step_id}")
        return existing_algorithm

    logger.info(f"Creating algorithm: {name} v{version} for step {step_id}")
    return crud_analysis.algorithm.create(
        db,
        obj_in=AlgorithmCreate(
            name=name,
            description=description,
            version=version,
            step_id=step_id,
            supported_document_types=supported_document_types,
            parameters=parameters
        )
    )

def init_analysis_db(db: Session) -> None:
    """Initialize the analysis database with default types, steps, and algorithms.
    This function is idempotent and can be safely run multiple times."""
    try:
        # Table Analysis Type
        table_analysis = crud_analysis.analysis_type.get_by_name(db, name=AnalysisTypeEnum.TABLE_ANALYSIS)
        if not table_analysis:
            logger.info("Creating Table Analysis type")
            table_analysis = crud_analysis.analysis_type.create(
                db,
                obj_in=AnalysisTypeCreate(
                    name=AnalysisTypeEnum.TABLE_ANALYSIS,
                    description="Detect and extract data from tables in documents",
                    supported_document_types=[DocumentType.PDF, DocumentType.IMAGE]
                )
            )
        else:
            logger.info("Table Analysis type already exists")

        # Create or get steps for table analysis
        table_detection_step = get_or_create_step(
            db,
            name=AnalysisStepEnum.TABLE_DETECTION,
            analysis_type_id=str(table_analysis.id),
            order=1,
            description="Detect table locations in the document",
            base_parameters=[
                Parameter(
                    name="page_range",
                    description="Range of pages to process (e.g., '1-5' or '1,3,5')",
                    type="string",
                    required=False,
                    default="all"
                )
            ]
        )

        table_structure_step = get_or_create_step(
            db,
            name=AnalysisStepEnum.TABLE_STRUCTURE_RECOGNITION,
            analysis_type_id=str(table_analysis.id),
            order=2,
            description="Recognize the structure of detected tables",
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

        table_data_step = get_or_create_step(
            db,
            name=AnalysisStepEnum.TABLE_DATA_EXTRACTION,
            analysis_type_id=str(table_analysis.id),
            order=3,
            description="Extract and process table data",
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

        # Register algorithms for each step
        # Table Detection Algorithms
        get_or_create_algorithm(
            db,
            name="TableDetectionBasic",
            description="Basic table detection using OpenCV contour detection",
            version="1.0.0",
            step_id=str(table_detection_step.id),
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            parameters=[
                Parameter(
                    name="max_tables",
                    description="Maximum number of tables to detect per page",
                    type="integer",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=50
                ),
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

        # Table Structure Recognition Algorithms
        get_or_create_algorithm(
            db,
            name="TableStructureBasic",
            description="Basic table structure recognition using grid detection",
            version="1.0.0",
            step_id=str(table_structure_step.id),
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            parameters=[
                Parameter(
                    name="min_line_length",
                    description="Minimum line length as percentage of image dimension",
                    type="float",
                    required=False,
                    default=0.1,
                    min_value=0.05,
                    max_value=0.5
                ),
                Parameter(
                    name="line_threshold",
                    description="Threshold for line detection",
                    type="integer",
                    required=False,
                    default=50,
                    min_value=10,
                    max_value=100
                ),
                Parameter(
                    name="header_row_count",
                    description="Number of header rows to detect",
                    type="integer",
                    required=False,
                    default=1,
                    min_value=0,
                    max_value=5
                )
            ]
        )

        # Table Data Extraction Algorithms
        get_or_create_algorithm(
            db,
            name="TableDataBasic",
            description="Basic table data extraction using OCR",
            version="1.0.0",
            step_id=str(table_data_step.id),
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            parameters=[
                Parameter(
                    name="ocr_lang",
                    description="Language for OCR (e.g., 'eng', 'fra')",
                    type="string",
                    required=False,
                    default="eng"
                ),
                Parameter(
                    name="data_types",
                    description="List of data types to detect",
                    type="array",
                    required=False,
                    default=["text", "number", "date"]
                ),
                Parameter(
                    name="confidence_threshold",
                    description="Minimum confidence threshold for OCR",
                    type="float",
                    required=False,
                    default=0.6,
                    min_value=0.1,
                    max_value=1.0
                )
            ]
        )

        # Text Analysis Type (placeholder for future implementation)
        text_analysis = crud_analysis.analysis_type.get_by_name(db, name=AnalysisTypeEnum.TEXT_ANALYSIS)
        if not text_analysis:
            logger.info("Creating Text Analysis type")
            crud_analysis.analysis_type.create(
                db,
                obj_in=AnalysisTypeCreate(
                    name=AnalysisTypeEnum.TEXT_ANALYSIS,
                    description="Extract and analyze text from documents",
                    supported_document_types=[DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOCX]
                )
            )
        else:
            logger.info("Text Analysis type already exists")

        # Template Conversion Type (placeholder for future implementation)
        template_conversion = crud_analysis.analysis_type.get_by_name(db, name=AnalysisTypeEnum.TEMPLATE_CONVERSION)
        if not template_conversion:
            logger.info("Creating Template Conversion type")
            crud_analysis.analysis_type.create(
                db,
                obj_in=AnalysisTypeCreate(
                    name=AnalysisTypeEnum.TEMPLATE_CONVERSION,
                    description="Convert documents into structured templates",
                    supported_document_types=[DocumentType.PDF, DocumentType.DOCX]
                )
            )
        else:
            logger.info("Template Conversion type already exists")

        logger.info("Analysis database initialization completed successfully")

    except Exception as e:
        logger.error(f"Error initializing analysis database: {str(e)}")
        raise 