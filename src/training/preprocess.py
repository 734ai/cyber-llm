"""
Data Preprocessing Pipeline for Cyber-LLM
Handles cleaning, tokenization, and preparation of cybersecurity training data.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer
    import numpy as np
    from sklearn.model_selection import train_test_split
    import dvc.api
except ImportError:
    print("Required packages not installed. Run: pip install transformers scikit-learn dvc")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data preprocessing."""
    max_sequence_length: int = 2048
    min_sequence_length: int = 128
    overlap_ratio: float = 0.1
    validation_split: float = 0.15
    test_split: float = 0.1
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    special_tokens: Dict[str, str] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                "recon_token": "<|RECON|>",
                "c2_token": "<|C2|>",
                "postexploit_token": "<|POSTEXPLOIT|>",
                "opsec_token": "<|OPSEC|>",
                "explain_token": "<|EXPLAIN|>",
                "safety_token": "<|SAFETY|>"
            }

class CyberDataPreprocessor:
    """Advanced data preprocessor for cybersecurity domain."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tokenizer = None
        self.document_stats = {}
        self._initialize_tokenizer()
        
    def _initialize_tokenizer(self):
        """Initialize tokenizer with special tokens."""
        logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True
        )
        
        # Add special tokens
        special_tokens_dict = {
            'additional_special_tokens': list(self.config.special_tokens.values())
        }
        
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_raw_data(self, data_dir: Path) -> List[Dict]:
        """Load and parse raw JSON data files."""
        logger.info(f"Loading data from: {data_dir}")
        
        json_files = list(data_dir.glob('**/*.json'))
        documents = []
        
        for json_file in json_files:
            if json_file.name in ['conversion_report.json', 'metadata.json']:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract relevant fields
                if 'content' in data and 'metadata' in data:
                    doc = {
                        'id': json_file.stem,
                        'content': data['content'],
                        'source_type': data['metadata'].get('source_type', 'unknown'),
                        'filename': data['metadata'].get('filename', ''),
                        'source_file': str(json_file)
                    }
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def clean_and_structure_text(self, documents: List[Dict]) -> List[Dict]:
        """Clean and structure text content."""
        logger.info("Cleaning and structuring text content")
        
        cleaned_documents = []
        
        for doc in documents:
            try:
                content = doc['content']
                
                # Clean content
                cleaned_content = self._clean_text(content)
                
                # Structure based on document type
                structured_content = self._structure_by_type(
                    cleaned_content, 
                    doc['source_type']
                )
                
                # Add special tokens based on content type
                tagged_content = self._add_domain_tags(
                    structured_content, 
                    doc['source_type']
                )
                
                doc['cleaned_content'] = cleaned_content
                doc['structured_content'] = structured_content
                doc['tagged_content'] = tagged_content
                
                cleaned_documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing document {doc['id']}: {str(e)}")
                continue
        
        logger.info(f"Cleaned {len(cleaned_documents)} documents")
        return cleaned_documents
    
    def _clean_text(self, text: str) -> str:
        """Clean raw text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)  # Word-number separation
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove artifacts
        text = re.sub(r'^\s*[•\-\*]\s*', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _structure_by_type(self, content: str, doc_type: str) -> str:
        """Structure content based on document type."""
        if doc_type == 'mitre_attack':
            return self._structure_mitre_content(content)
        elif doc_type == 'apt_report':
            return self._structure_apt_report(content)
        elif doc_type == 'opsec_guide':
            return self._structure_opsec_content(content)
        elif doc_type == 'malware_analysis':
            return self._structure_malware_content(content)
        else:
            return content
    
    def _structure_mitre_content(self, content: str) -> str:
        """Structure MITRE ATT&CK content."""
        # Find and structure tactics/techniques
        sections = []
        
        # Look for technique patterns
        technique_pattern = r'(T\d{4}(?:\.\d{3})?)\s*[-:]\s*([^\n]+)'
        techniques = re.findall(technique_pattern, content)
        
        if techniques:
            sections.append("MITRE ATT&CK Techniques:")
            for tech_id, tech_name in techniques:
                sections.append(f"- {tech_id}: {tech_name}")
        
        # Look for tactic sections
        tactic_pattern = r'(Initial Access|Execution|Persistence|Privilege Escalation|Defense Evasion|Credential Access|Discovery|Lateral Movement|Collection|Exfiltration|Command and Control|Impact)'
        
        current_section = None
        for line in content.split('\n'):
            tactic_match = re.search(tactic_pattern, line, re.IGNORECASE)
            if tactic_match:
                current_section = tactic_match.group(1)
                sections.append(f"\n{current_section}:")
            elif current_section and line.strip():
                sections.append(f"  {line.strip()}")
        
        return '\n'.join(sections) if sections else content
    
    def _structure_apt_report(self, content: str) -> str:
        """Structure APT report content."""
        sections = []
        
        # Extract IOCs
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        domain_pattern = r'\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+\b'
        hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
        
        ips = re.findall(ip_pattern, content)
        domains = re.findall(domain_pattern, content)
        hashes = re.findall(hash_pattern, content)
        
        if ips or domains or hashes:
            sections.append("Indicators of Compromise (IOCs):")
            if ips:
                sections.append(f"IPs: {', '.join(set(ips[:10]))}")  # Limit to first 10
            if domains:
                sections.append(f"Domains: {', '.join(set(domains[:10]))}")
            if hashes:
                sections.append(f"Hashes: {', '.join(set(hashes[:5]))}")
        
        # Extract TTPs
        ttp_keywords = ['technique', 'tactic', 'procedure', 'method', 'attack']
        ttp_lines = []
        
        for line in content.split('\n'):
            if any(keyword in line.lower() for keyword in ttp_keywords):
                ttp_lines.append(line.strip())
        
        if ttp_lines:
            sections.append("\nTTPs (Tactics, Techniques, Procedures):")
            sections.extend(f"- {line}" for line in ttp_lines[:10])
        
        sections.append(f"\nFull Report:\n{content}")
        return '\n'.join(sections)
    
    def _structure_opsec_content(self, content: str) -> str:
        """Structure OPSEC guide content."""
        sections = []
        
        # Look for OPSEC principles/rules
        opsec_keywords = ['stealth', 'detection', 'evasion', 'anonymity', 'operational security']
        opsec_lines = []
        
        for line in content.split('\n'):
            if any(keyword in line.lower() for keyword in opsec_keywords):
                opsec_lines.append(line.strip())
        
        if opsec_lines:
            sections.append("OPSEC Guidelines:")
            sections.extend(f"- {line}" for line in opsec_lines[:15])
        
        sections.append(f"\nDetailed Content:\n{content}")
        return '\n'.join(sections)
    
    def _structure_malware_content(self, content: str) -> str:
        """Structure malware analysis content."""
        sections = []
        
        # Extract analysis sections
        analysis_sections = ['summary', 'behavior', 'network', 'filesystem', 'registry']
        
        for section in analysis_sections:
            pattern = rf'{section}[:\s]+(.*?)(?=\n[a-z]+:|$)'
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                sections.append(f"{section.title()}: {match.group(1).strip()}")
        
        return '\n'.join(sections) if sections else content
    
    def _add_domain_tags(self, content: str, doc_type: str) -> str:
        """Add domain-specific special tokens."""
        tag_mapping = {
            'mitre_attack': self.config.special_tokens['recon_token'],
            'apt_report': self.config.special_tokens['postexploit_token'],
            'opsec_guide': self.config.special_tokens['opsec_token'],
            'malware_analysis': self.config.special_tokens['postexploit_token']
        }
        
        tag = tag_mapping.get(doc_type, '')
        if tag:
            return f"{tag} {content}"
        return content
    
    def create_training_sequences(self, documents: List[Dict]) -> List[Dict]:
        """Create training sequences with proper tokenization."""
        logger.info("Creating training sequences")
        
        sequences = []
        
        for doc in documents:
            content = doc['tagged_content']
            
            # Tokenize content
            tokens = self.tokenizer.encode(content, add_special_tokens=True)
            
            # Create overlapping sequences
            seq_length = self.config.max_sequence_length
            overlap = int(seq_length * self.config.overlap_ratio)
            
            for i in range(0, len(tokens), seq_length - overlap):
                seq_tokens = tokens[i:i + seq_length]
                
                # Skip sequences that are too short
                if len(seq_tokens) < self.config.min_sequence_length:
                    continue
                
                # Pad sequence if necessary
                if len(seq_tokens) < seq_length:
                    seq_tokens.extend([self.tokenizer.pad_token_id] * (seq_length - len(seq_tokens)))
                
                sequence = {
                    'input_ids': seq_tokens,
                    'attention_mask': [1 if token != self.tokenizer.pad_token_id else 0 for token in seq_tokens],
                    'labels': seq_tokens.copy(),  # For language modeling
                    'source_type': doc['source_type'],
                    'document_id': doc['id'],
                    'sequence_index': i // (seq_length - overlap)
                }
                
                sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} training sequences")
        return sequences
    
    def split_data(self, sequences: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/validation/test sets."""
        logger.info("Splitting data into train/validation/test sets")
        
        # Group sequences by document to ensure proper splitting
        doc_sequences = {}
        for seq in sequences:
            doc_id = seq['document_id']
            if doc_id not in doc_sequences:
                doc_sequences[doc_id] = []
            doc_sequences[doc_id].append(seq)
        
        # Split document IDs
        doc_ids = list(doc_sequences.keys())
        
        # First split: train + temp vs test
        train_temp_ids, test_ids = train_test_split(
            doc_ids, 
            test_size=self.config.test_split, 
            random_state=42,
            shuffle=True
        )
        
        # Second split: train vs validation
        val_size = self.config.validation_split / (1 - self.config.test_split)
        train_ids, val_ids = train_test_split(
            train_temp_ids,
            test_size=val_size,
            random_state=42,
            shuffle=True
        )
        
        # Collect sequences for each split
        train_sequences = []
        val_sequences = []
        test_sequences = []
        
        for doc_id, doc_seqs in doc_sequences.items():
            if doc_id in train_ids:
                train_sequences.extend(doc_seqs)
            elif doc_id in val_ids:
                val_sequences.extend(doc_seqs)
            else:  # test_ids
                test_sequences.extend(doc_seqs)
        
        logger.info(f"Split: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")
        
        return train_sequences, val_sequences, test_sequences
    
    def save_processed_data(self, train_data: List[Dict], val_data: List[Dict], 
                          test_data: List[Dict], output_dir: Path):
        """Save processed data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        datasets = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, data in datasets.items():
            output_file = output_dir / f'{split_name}.json'
            
            logger.info(f"Saving {len(data)} {split_name} sequences to {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save tokenizer
        tokenizer_dir = output_dir / 'tokenizer'
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        # Save preprocessing metadata
        metadata = {
            'config': self.config.__dict__,
            'splits': {
                'train_size': len(train_data),
                'validation_size': len(val_data),
                'test_size': len(test_data)
            },
            'tokenizer_info': {
                'vocab_size': self.tokenizer.vocab_size,
                'model_max_length': self.tokenizer.model_max_length,
                'special_tokens': self.config.special_tokens
            }
        }
        
        metadata_file = output_dir / 'preprocessing_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Preprocessing complete. Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess cybersecurity data for Cyber-LLM')
    parser.add_argument('--input', required=True, help='Input directory with raw JSON files')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--tokenizer', default='microsoft/DialoGPT-medium', help='Tokenizer model name')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProcessingConfig(
        max_sequence_length=args.max_length,
        tokenizer_name=args.tokenizer
    )
    
    # Initialize preprocessor
    preprocessor = CyberDataPreprocessor(config)
    
    # Load and process data
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Load raw data
    documents = preprocessor.load_raw_data(input_dir)
    
    # Clean and structure
    cleaned_documents = preprocessor.clean_and_structure_text(documents)
    
    # Create training sequences
    sequences = preprocessor.create_training_sequences(cleaned_documents)
    
    # Split data
    train_data, val_data, test_data = preprocessor.split_data(sequences)
    
    # Save processed data
    preprocessor.save_processed_data(train_data, val_data, test_data, output_dir)
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == '__main__':
    main()
