from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd
import docx
from docx.table import Table as DocxTable, _Cell
from docx.text.paragraph import Paragraph
import numpy as np
from collections import defaultdict

from app.services.ml.base import BaseTableDetector

logger = logging.getLogger(__name__)

class WordTableDetector(BaseTableDetector):
    """Service for detecting and extracting tables from Word documents."""
    
    def __init__(self):
        """Initialize the Word table detection service."""
        logger.debug("Initializing Word Table Detection Service")
        
    @property
    def supported_formats(self) -> List[str]:
        """List of supported file formats."""
        return ["docx", "doc"]

    def _extract_cell_content(self, cell: _Cell) -> Dict[str, Any]:
        """
        Extract content and formatting from a table cell.
        
        Args:
            cell: Word document table cell
            
        Returns:
            Dictionary containing cell content and formatting
        """
        content = {
            "text": "",
            "paragraphs": [],
            "formatting": {
                "background_color": None,
                "vertical_alignment": str(cell._tc.get_or_add_tcPr().vAlign),
                "width": cell.width if cell.width else None,
            }
        }
        
        # Extract text and formatting from each paragraph in the cell
        for paragraph in cell.paragraphs:
            para_content = {
                "text": paragraph.text,
                "alignment": str(paragraph.alignment),
                "style": paragraph.style.name if paragraph.style else None,
                "runs": []
            }
            
            # Extract formatting from each run (text with same formatting)
            for run in paragraph.runs:
                run_content = {
                    "text": run.text,
                    "bold": run.bold,
                    "italic": run.italic,
                    "underline": run.underline,
                    "font": {
                        "name": run.font.name,
                        "size": run.font.size,
                        "color": run.font.color.rgb if run.font.color else None,
                    }
                }
                para_content["runs"].append(run_content)
                
            content["paragraphs"].append(para_content)
            content["text"] += paragraph.text + "\n"
            
        content["text"] = content["text"].strip()
        return content

    def _analyze_table_structure(self, table: DocxTable) -> Dict[str, Any]:
        """
        Analyze table structure including headers and data organization.
        
        Args:
            table: Word document table
            
        Returns:
            Dictionary containing table structure information
        """
        rows = len(table.rows)
        cols = len(table.columns)
        
        # Initialize structure analysis
        structure = {
            "dimensions": {"rows": rows, "cols": cols},
            "has_header": False,
            "header_rows": [],
            "merged_cells": [],
            "column_types": {},
            "empty_cells": []
        }
        
        # Detect merged cells
        for i in range(rows):
            for j in range(cols):
                try:
                    cell = table.cell(i, j)
                    tc = cell._tc
                    
                    # Check vertical merge
                    v_merge = tc.vMerge and tc.vMerge.val
                    
                    # Check horizontal merge using grid span
                    grid_span = tc.tcPr.gridSpan
                    h_merge = grid_span and grid_span.val and grid_span.val > 1
                    
                    if v_merge or h_merge:
                        merge_info = {
                            "row": i,
                            "col": j,
                            "vertical_merge": bool(v_merge),
                            "horizontal_merge": bool(h_merge)
                        }
                        
                        # Add span information if available
                        if h_merge:
                            merge_info["grid_span"] = int(grid_span.val)
                            
                        structure["merged_cells"].append(merge_info)
                        
                except Exception as e:
                    logger.debug(f"Could not check merge status for cell ({i}, {j}): {str(e)}")
                    continue
        
        # Analyze first row for header detection
        try:
            first_row = table.rows[0]
            header_cells = first_row.cells
            
            # Check if first row has header formatting
            header_indicators = []
            for cell in header_cells:
                # Check various header indicators
                is_header = False
                if cell.paragraphs and cell.paragraphs[0].runs:
                    run = cell.paragraphs[0].runs[0]
                    is_header = (
                        run.bold or
                        run.style and "head" in run.style.name.lower() or
                        cell.paragraphs[0].style and "head" in cell.paragraphs[0].style.name.lower()
                    )
                header_indicators.append(is_header)
            
            # If majority of cells have header formatting
            if sum(header_indicators) > len(header_indicators) / 2:
                structure["has_header"] = True
                structure["header_rows"].append(0)
                
        except Exception as e:
            logger.debug(f"Could not analyze header row: {str(e)}")
        
        # Analyze column types
        for j in range(cols):
            column_values = []
            for i in range(1 if structure["has_header"] else 0, rows):
                try:
                    cell_text = table.cell(i, j).text.strip()
                    if cell_text:
                        column_values.append(cell_text)
                    else:
                        structure["empty_cells"].append({"row": i, "col": j})
                except Exception as e:
                    logger.debug(f"Could not get text from cell ({i}, {j}): {str(e)}")
                    continue
            
            # Determine column type
            if column_values:
                col_type = self._determine_column_type(column_values)
                structure["column_types"][j] = col_type
        
        return structure

    def _determine_column_type(self, values: List[str]) -> Dict[str, Any]:
        """
        Determine the data type of a column based on its values.
        
        Args:
            values: List of cell values in the column
            
        Returns:
            Dictionary containing column type information
        """
        numeric_count = 0
        date_count = 0
        total = len(values)
        
        for value in values:
            # Try numeric conversion
            try:
                float(value.replace(',', ''))
                numeric_count += 1
                continue
            except ValueError:
                pass
            
            # Try date parsing
            try:
                pd.to_datetime(value)
                date_count += 1
            except (ValueError, TypeError):
                pass
        
        # Determine predominant type
        if numeric_count / total > 0.8:
            return {
                "type": "numeric",
                "confidence": numeric_count / total,
                "format": "integer" if all('.' not in v for v in values) else "float"
            }
        elif date_count / total > 0.8:
            return {
                "type": "date",
                "confidence": date_count / total
            }
        else:
            return {
                "type": "text",
                "confidence": 1.0
            }

    def _extract_table_data(
        self, 
        table: DocxTable, 
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured data from the table.
        
        Args:
            table: Word document table
            structure: Table structure information
            
        Returns:
            Dictionary containing extracted table data
        """
        rows = structure["dimensions"]["rows"]
        cols = structure["dimensions"]["cols"]
        
        # Initialize data structure
        data = {
            "headers": [],
            "data": [],
            "merged_regions": structure["merged_cells"],
            "column_types": structure["column_types"]
        }
        
        # Extract headers if present
        if structure["has_header"]:
            headers = []
            for j in range(cols):
                cell = table.cell(0, j)
                headers.append(self._extract_cell_content(cell))
            data["headers"] = headers
        
        # Extract data rows
        start_row = 1 if structure["has_header"] else 0
        for i in range(start_row, rows):
            row_data = []
            for j in range(cols):
                cell = table.cell(i, j)
                cell_content = self._extract_cell_content(cell)
                
                # Convert data based on column type if possible
                if j in structure["column_types"]:
                    col_type = structure["column_types"][j]
                    if col_type["type"] == "numeric":
                        try:
                            cell_content["value"] = float(
                                cell_content["text"].replace(',', '')
                            )
                        except ValueError:
                            cell_content["value"] = None
                    elif col_type["type"] == "date":
                        try:
                            cell_content["value"] = pd.to_datetime(
                                cell_content["text"]
                            ).isoformat()
                        except ValueError:
                            cell_content["value"] = None
                    else:
                        cell_content["value"] = cell_content["text"]
                
                row_data.append(cell_content)
            data["data"].append(row_data)
        
        return data

    def detect_tables(
        self, 
        file_path: str, 
        extract_structure: bool = True,
        extract_data: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect and extract tables from a Word document.
        
        Args:
            file_path: Path to the Word document
            extract_structure: Whether to analyze table structure
            extract_data: Whether to extract table data
            
        Returns:
            List of detected tables with their properties
        """
        logger.debug(f"Detecting tables in Word document: {file_path}")
        
        try:
            # Load document
            doc = docx.Document(file_path)
            tables = []
            
            # Process each table
            for table_index, table in enumerate(doc.tables):
                logger.debug(f"Processing table {table_index + 1}")
                
                table_info = {
                    "index": table_index,
                    "dimensions": {
                        "rows": len(table.rows),
                        "cols": len(table.columns)
                    },
                    "location": self._get_table_location(doc, table),
                    "confidence": 1.0  # Word tables are explicit
                }
                
                # Extract structure if requested
                if extract_structure:
                    table_info["structure"] = self._analyze_table_structure(table)
                
                # Extract data if requested
                if extract_data:
                    table_info["data"] = self._extract_table_data(
                        table,
                        table_info.get("structure", {"dimensions": table_info["dimensions"]})
                    )
                
                tables.append(table_info)
            
            logger.info(f"Detected {len(tables)} tables in document")
            return tables
            
        except Exception as e:
            logger.error(f"Word table detection failed: {str(e)}")
            raise

    def _get_table_location(self, doc: docx.Document, target_table: DocxTable) -> Dict[str, Any]:
        """
        Get the location context of a table in the document.
        
        Args:
            doc: Word document
            target_table: Target table to locate
            
        Returns:
            Dictionary containing table location information
        """
        location = {
            "section": 0,
            "preceding_heading": None,
            "preceding_text": None
        }
        
        current_section = 0
        last_heading = None
        last_text = None
        
        for element in doc.element.body:
            if element.tag.endswith('sectPr'):
                current_section += 1
            elif element.tag.endswith('p'):
                p = Paragraph(element, doc)
                if p.style.name and 'Heading' in p.style.name:
                    last_heading = p.text
                else:
                    last_text = p.text
            elif element.tag.endswith('tbl'):
                if element == target_table._tbl:
                    location.update({
                        "section": current_section,
                        "preceding_heading": last_heading,
                        "preceding_text": last_text
                    })
                    break
        
        return location

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid Word document.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            Boolean indicating if file is valid
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"Word document not found: {file_path}")
                return False
            
            # Check if file is a valid Word document
            try:
                doc = docx.Document(file_path)
                return True
            except Exception:
                logger.error(f"Invalid Word document: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"Word document validation failed: {str(e)}")
            return False 