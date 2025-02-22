from typing import Dict, Any, List
from app.services.analysis.configs.base.base_definition import BaseDefinition
from app.schemas.analysis.configs.definitions import AnalysisDefinitionInfo
from app.schemas.document import Document
from app.enums.document import DocumentType
from app.services.analysis.configs.registry import AnalysisRegistry
from .steps.table_detection import TableDetectionStep
from .steps.table_structure import TableStructureStep
from .steps.table_data import TableDataStep

class TableAnalysis(BaseDefinition):
    """Table analysis implementation"""
    
    def get_info(self) -> AnalysisDefinitionInfo:
        return AnalysisDefinitionInfo(
            code="table_analysis",
            name="Table Analysis",
            version="1.0.0",
            description="Detect and extract data from tables in documents",
            supported_document_types=[DocumentType.PDF, DocumentType.IMAGE],
            implementation_path="app.services.analysis.configs.definitions.table_analysis.analysis.TableAnalysis",
            is_active=True
        )
    
    async def validate_requirements(self) -> None:
        """Validate that all required dependencies are available"""
        # Add dependency checks here if needed
        pass
    
    async def validate_document(self, document: Document) -> None:
        """Validate if document can be processed"""
        if document.type not in self.get_info().supported_document_types:
            raise ValueError(f"Document type {document.type} not supported")
    
    async def prepare_document(self, document: Document, document_path: str) -> Dict[str, Any]:
        """Prepare document for analysis"""
        return {
            "document_id": str(document.id),
            "document_path": document_path,
            "type": document.type.value
        }
    
    async def validate_results(self, results: List[Dict[str, Any]], document: Document) -> None:
        """Validate final analysis results"""
        if not results:
            raise ValueError("No results generated from analysis")
    
    async def post_process_results(self, results: List[Dict[str, Any]], document: Document) -> Dict[str, Any]:
        """Post-process all analysis results"""
        return {
            "document_id": str(document.id),
            "tables": results
        }
    
    async def cleanup(self, document: Document) -> None:
        """Cleanup any temporary resources"""
        # Add cleanup logic here if needed
        pass 