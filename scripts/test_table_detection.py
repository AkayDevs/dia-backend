#!/usr/bin/env python3
"""
Manual test script for table detection.
This script allows testing table detection on real documents.
"""

import argparse
import asyncio
import json
from pathlib import Path
import logging
from typing import Dict, Any
import sys

from app.services.ml.table_detection import (
    ImageTableDetector,
    PDFTableDetector,
    WordTableDetector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_results(results: Dict[str, Any], output_path: Path):
    """Save detection results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

async def process_file(file_path: str, output_dir: str):
    """Process a single file and save results."""
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine file type and select appropriate detector
    ext = file_path.suffix.lower()
    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        detector = ImageTableDetector()
        detector_type = "image"
    elif ext == '.pdf':
        detector = PDFTableDetector()
        detector_type = "pdf"
    elif ext in ['.docx', '.doc']:
        detector = WordTableDetector()
        detector_type = "word"
    else:
        logger.error(f"Unsupported file type: {ext}")
        return
    
    try:
        # Validate file
        if not detector.validate_file(str(file_path)):
            logger.error(f"Invalid file: {file_path}")
            return
        
        # Detect tables with all features enabled
        logger.info(f"Processing {file_path}")
        
        if detector_type == "word":
            tables = detector.detect_tables(
                str(file_path),
                extract_structure=True,
                extract_data=True
            )
        elif detector_type == "pdf":
            tables = detector.detect_tables(
                str(file_path),
                confidence_threshold=0.5,
                use_ml_detection=True
            )
        else:
            tables = detector.detect_tables(
                str(file_path),
                confidence_threshold=0.5,
                enhance_image=True
            )
        
        # Prepare results
        results = {
            "file_path": str(file_path),
            "file_type": detector_type,
            "table_count": len(tables),
            "tables": tables
        }
        
        # Save results
        output_path = output_dir / f"{file_path.stem}_results.json"
        save_results(results, output_path)
        
        # Print summary
        logger.info(f"Found {len(tables)} tables in {file_path}")
        for i, table in enumerate(tables):
            logger.info(f"\nTable {i + 1}:")
            if "dimensions" in table:
                logger.info(f"  Dimensions: {table['dimensions']}")
            if "confidence" in table:
                logger.info(f"  Confidence: {table['confidence']:.2f}")
            if "structure" in table and table["structure"].get("has_header"):
                logger.info("  Headers detected")
                if "data" in table and "headers" in table["data"]:
                    headers = [h["text"] for h in table["data"]["headers"]]
                    logger.info(f"  Header texts: {headers}")
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        raise

async def main():
    parser = argparse.ArgumentParser(description="Test table detection on documents")
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to documents to process"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./detection_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Process each file
    for file_path in args.files:
        await process_file(file_path, args.output)

if __name__ == "__main__":
    asyncio.run(main()) 