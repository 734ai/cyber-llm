#!/usr/bin/env python3
"""
PDF to Text Conversion Script for Cyber-LLM Data Ingestion
Converts PDF files containing APT reports, MITRE ATT&CK documentation, 
and OPSEC guides into structured text format.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import json

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
except ImportError:
    print("pdfminer.six not installed. Run: pip install pdfminer.six")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFConverter:
    """Convert PDF files to structured text with metadata extraction."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """Extract text and metadata from a single PDF file."""
        try:
            # Extract text with custom layout parameters
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
                all_texts=False
            )
            
            text = extract_text(pdf_path, laparams=laparams)
            
            # Clean and structure the text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Extract basic metadata
            metadata = {
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'line_count': len(lines),
                'char_count': len(text),
                'source_type': self._classify_document(pdf_path.name, text[:1000])
            }
            
            return {
                'metadata': metadata,
                'content': text,
                'lines': lines
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def _classify_document(self, filename: str, preview: str) -> str:
        """Classify document type based on filename and content preview."""
        filename_lower = filename.lower()
        preview_lower = preview.lower()
        
        if 'mitre' in filename_lower or 'att&ck' in preview_lower:
            return 'mitre_attack'
        elif any(term in filename_lower for term in ['apt', 'threat', 'actor']):
            return 'apt_report'
        elif any(term in preview_lower for term in ['opsec', 'tradecraft', 'stealth']):
            return 'opsec_guide'
        elif any(term in filename_lower for term in ['malware', 'forensic']):
            return 'malware_analysis'
        else:
            return 'general_cybersec'
    
    def process_directory(self) -> Dict:
        """Process all PDF files in the input directory."""
        pdf_files = list(self.input_dir.glob('**/*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            'processed_files': [],
            'failed_files': [],
            'summary': {}
        }
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            extracted_data = self.extract_text_from_pdf(pdf_file)
            if extracted_data:
                # Save extracted text
                output_file = self.output_dir / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                
                results['processed_files'].append({
                    'input': str(pdf_file),
                    'output': str(output_file),
                    'metadata': extracted_data['metadata']
                })
                
                logger.info(f"✓ Converted: {pdf_file.name} -> {output_file.name}")
            else:
                results['failed_files'].append(str(pdf_file))
                logger.error(f"✗ Failed: {pdf_file.name}")
        
        # Generate summary statistics
        doc_types = {}
        total_chars = 0
        for file_info in results['processed_files']:
            doc_type = file_info['metadata']['source_type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            total_chars += file_info['metadata']['char_count']
        
        results['summary'] = {
            'total_processed': len(results['processed_files']),
            'total_failed': len(results['failed_files']),
            'document_types': doc_types,
            'total_characters': total_chars
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Convert PDF files to text for Cyber-LLM training')
    parser.add_argument('--input', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', required=True, help='Output directory for converted text files')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize converter
    converter = PDFConverter(args.input, args.output)
    
    # Process files
    results = converter.process_directory()
    
    # Save processing report
    report_file = Path(args.output) / 'conversion_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    summary = results['summary']
    logger.info(f"""
    
Conversion Complete!
==================
Files processed: {summary['total_processed']}
Files failed: {summary['total_failed']}
Total characters: {summary['total_characters']:,}
Document types: {summary['document_types']}
Report saved to: {report_file}
    """)

if __name__ == '__main__':
    main()
