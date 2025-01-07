from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseDetector(ABC):
    """Base class for all detectors."""
    
    @abstractmethod
    async def detect(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and extract information from document."""
        pass

class BaseTableDetector(BaseDetector):
    """Base class for table detectors."""
    
    @abstractmethod
    async def detect_tables(self, file_path: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and extract tables from document."""
        pass
        
    async def detect(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement base detect method to return tables."""
        tables = await self.detect_tables(file_path, parameters)
        return {"tables": tables}

class BaseTextExtractor(BaseDetector):
    """Base class for text extractors."""
    
    @abstractmethod
    async def extract_text(self, file_path: str, parameters: Dict[str, Any]) -> str:
        """Extract text from document."""
        pass
        
    async def detect(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement base detect method to return extracted text."""
        text = await self.extract_text(file_path, parameters)
        return {"text": text} 