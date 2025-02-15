from typing import Dict, Any
from app.analysis.base.base_analysis import BaseAnalysis
from app.analysis.registry.components import AnalysisTypeInfo, AnalysisIdentifier
from app.analysis.registry.registry import AnalysisRegistry
from app.schemas.document import Document
from app.enums.document import DocumentType
from .steps.table_detection import TableDetectionStep
from .steps.table_structure import TableStructureStep
from .steps.table_data import TableDataStep

class TableAnalysis(BaseAnalysis):
    """Table analysis implementation"""
    
    def get_info(self) -> AnalysisTypeInfo:
        return AnalysisTypeInfo(
            identifier=AnalysisIdentifier(
                name="Table Analysis",
                code="table_analysis",
                version="1.0.0"
            ),
            description="Detect and extract data from tables in documents",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            steps=[
                TableDetectionStep().get_info(),
                TableStructureStep().get_info(),
                TableDataStep().get_info()
            ],
            implementation_path="app.analysis.types.table_analysis.analysis.TableAnalysis"
        )
    
    async def validate_input(self, document: Document) -> bool:
        """Validate if document can be processed"""
        return document.type in self.get_info().supported_document_types
    
    async def prepare_document(self, document: Document) -> Dict[str, Any]:
        """Prepare document for analysis"""
        # Implementation here - e.g., convert PDF to images, preprocess images, etc.
        return {
            "document_id": document.id,
            "pages": []  # Add processed pages data
        }
    
    async def cleanup(self, document: Document) -> None:
        """Cleanup temporary resources"""
        # Implementation here - e.g., remove temporary files
        pass

# Register the analysis type
AnalysisRegistry.register_analysis_type(TableAnalysis().get_info()) 