"""
Data Validation Tests for Cyber-LLM
Automated checks for schema, duplicates, encoding, and data quality
"""

import pytest
import json
import yaml
import os
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chardet
import logging
from jsonschema import validate, ValidationError
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation utilities for Cyber-LLM datasets"""
    
    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed" 
        self.versioned_dir = self.data_dir / "versioned"
        
        # Define expected schemas
        self.schemas = self._load_schemas()
        
    def _load_schemas(self) -> Dict[str, Dict]:
        """Load data schemas for validation"""
        return {
            "attack_technique": {
                "type": "object",
                "required": ["technique_id", "name", "description", "tactics", "platforms"],
                "properties": {
                    "technique_id": {"type": "string", "pattern": "^T[0-9]{4}(\\.[0-9]{3})?$"},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 10},
                    "tactics": {"type": "array", "items": {"type": "string"}},
                    "platforms": {"type": "array", "items": {"type": "string"}},
                    "data_sources": {"type": "array", "items": {"type": "string"}},
                    "mitigations": {"type": "array", "items": {"type": "string"}},
                    "detection": {"type": "string"},
                    "references": {"type": "array", "items": {"type": "string"}}
                }
            },
            "apt_report": {
                "type": "object",
                "required": ["report_id", "title", "threat_actor", "date", "content"],
                "properties": {
                    "report_id": {"type": "string"},
                    "title": {"type": "string", "minLength": 5},
                    "threat_actor": {"type": "string"},
                    "date": {"type": "string", "format": "date"},
                    "content": {"type": "string", "minLength": 100},
                    "techniques": {"type": "array", "items": {"type": "string"}},
                    "iocs": {"type": "array", "items": {"type": "string"}},
                    "malware_families": {"type": "array", "items": {"type": "string"}},
                    "targeted_sectors": {"type": "array", "items": {"type": "string"}}
                }
            },
            "training_example": {
                "type": "object",
                "required": ["input", "output", "category"],
                "properties": {
                    "input": {"type": "string", "minLength": 10},
                    "output": {"type": "string", "minLength": 10},
                    "category": {"type": "string", "enum": ["recon", "c2", "post_exploit", "general"]},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "safety_score": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        }

class TestDataIntegrity:
    """Test data integrity and quality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = DataValidator()
        
    def test_directory_structure(self):
        """Test that required directories exist"""
        assert self.validator.data_dir.exists(), "Data directory does not exist"
        assert self.validator.raw_dir.exists(), "Raw data directory does not exist"
        assert self.validator.processed_dir.exists(), "Processed data directory does not exist"
        assert self.validator.versioned_dir.exists(), "Versioned data directory does not exist"
        
    def test_file_encoding(self):
        """Test that all text files use valid encoding"""
        for data_dir in [self.validator.raw_dir, self.validator.processed_dir]:
            if not data_dir.exists():
                continue
                
            for file_path in data_dir.rglob("*.txt"):
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    
                assert result['confidence'] > 0.7, f"Low confidence encoding detection for {file_path}"
                assert result['encoding'] in ['utf-8', 'ascii', 'iso-8859-1'], \
                    f"Unsupported encoding {result['encoding']} in {file_path}"
    
    def test_json_validity(self):
        """Test that all JSON files are valid"""
        for data_dir in [self.validator.raw_dir, self.validator.processed_dir]:
            if not data_dir.exists():
                continue
                
            for file_path in data_dir.rglob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {file_path}: {e}")
    
    def test_yaml_validity(self):
        """Test that all YAML files are valid"""
        for data_dir in [self.validator.raw_dir, self.validator.processed_dir]:
            if not data_dir.exists():
                continue
                
            for file_path in data_dir.rglob("*.yaml"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {file_path}: {e}")
    
    def test_file_sizes(self):
        """Test that files are within expected size ranges"""
        for data_dir in [self.validator.raw_dir, self.validator.processed_dir]:
            if not data_dir.exists():
                continue
                
            for file_path in data_dir.rglob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    # Basic size checks
                    assert size_mb > 0, f"Empty file: {file_path}"
                    assert size_mb < 1000, f"File too large (>{size_mb:.1f}MB): {file_path}"
    
    def test_duplicate_detection(self):
        """Test for duplicate files using hash comparison"""
        file_hashes = {}
        
        for data_dir in [self.validator.raw_dir, self.validator.processed_dir]:
            if not data_dir.exists():
                continue
                
            for file_path in data_dir.rglob("*"):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                        
                    if file_hash in file_hashes:
                        pytest.fail(f"Duplicate file detected: {file_path} and {file_hashes[file_hash]}")
                    
                    file_hashes[file_hash] = file_path

class TestSchemaValidation:
    """Test data against expected schemas"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = DataValidator()
    
    def test_attack_technique_schema(self):
        """Test ATT&CK technique data against schema"""
        attack_files = list(self.validator.processed_dir.rglob("*attack*.json"))
        
        for file_path in attack_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single objects and arrays
            if isinstance(data, list):
                for item in data:
                    self._validate_attack_technique(item, file_path)
            else:
                self._validate_attack_technique(data, file_path)
    
    def _validate_attack_technique(self, data: Dict, file_path: Path):
        """Validate a single attack technique"""
        try:
            validate(instance=data, schema=self.validator.schemas["attack_technique"])
        except ValidationError as e:
            pytest.fail(f"Schema validation failed for {file_path}: {e.message}")
    
    def test_apt_report_schema(self):
        """Test APT report data against schema"""
        apt_files = list(self.validator.processed_dir.rglob("*apt*.json")) + \
                   list(self.validator.processed_dir.rglob("*threat*.json"))
        
        for file_path in apt_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    self._validate_apt_report(item, file_path)
            else:
                self._validate_apt_report(data, file_path)
    
    def _validate_apt_report(self, data: Dict, file_path: Path):
        """Validate a single APT report"""
        try:
            validate(instance=data, schema=self.validator.schemas["apt_report"])
        except ValidationError as e:
            pytest.fail(f"Schema validation failed for {file_path}: {e.message}")
    
    def test_training_example_schema(self):
        """Test training examples against schema"""
        training_files = list(self.validator.processed_dir.rglob("*training*.json")) + \
                        list(self.validator.processed_dir.rglob("*examples*.json"))
        
        for file_path in training_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    self._validate_training_example(item, file_path)
            else:
                self._validate_training_example(data, file_path)
    
    def _validate_training_example(self, data: Dict, file_path: Path):
        """Validate a single training example"""
        try:
            validate(instance=data, schema=self.validator.schemas["training_example"])
        except ValidationError as e:
            pytest.fail(f"Schema validation failed for {file_path}: {e.message}")

class TestDataQuality:
    """Test data quality and consistency"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = DataValidator()
    
    def test_technique_id_format(self):
        """Test that technique IDs follow correct format"""
        pattern = re.compile(r'^T[0-9]{4}(\.[0-9]{3})?$')
        
        for file_path in self.validator.processed_dir.rglob("*attack*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                techniques = data
            else:
                techniques = [data]
            
            for technique in techniques:
                if 'technique_id' in technique:
                    tid = technique['technique_id']
                    assert pattern.match(tid), f"Invalid technique ID format: {tid} in {file_path}"
    
    def test_date_formats(self):
        """Test date format consistency"""
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        
        for file_path in self.validator.processed_dir.rglob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._check_dates_recursive(data, date_pattern, file_path)
    
    def _check_dates_recursive(self, data: Any, pattern: re.Pattern, file_path: Path):
        """Recursively check date formats in data structure"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ['date', 'created_date', 'modified_date', 'published_date']:
                    if isinstance(value, str) and not pattern.match(value):
                        pytest.fail(f"Invalid date format '{value}' for key '{key}' in {file_path}")
                else:
                    self._check_dates_recursive(value, pattern, file_path)
        elif isinstance(data, list):
            for item in data:
                self._check_dates_recursive(item, pattern, file_path)
    
    def test_safety_scores(self):
        """Test that safety scores are within valid range"""
        for file_path in self.validator.processed_dir.rglob("*training*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                examples = data
            else:
                examples = [data]
            
            for example in examples:
                if 'safety_score' in example:
                    score = example['safety_score']
                    assert 0 <= score <= 1, f"Safety score {score} out of range [0,1] in {file_path}"
    
    def test_text_quality(self):
        """Test text content quality"""
        for file_path in self.validator.processed_dir.rglob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._check_text_quality_recursive(data, file_path)
    
    def _check_text_quality_recursive(self, data: Any, file_path: Path):
        """Recursively check text quality in data structure"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ['description', 'content', 'input', 'output'] and isinstance(value, str):
                    # Check for minimum length
                    assert len(value.strip()) >= 10, f"Text too short for '{key}' in {file_path}"
                    
                    # Check for placeholder text
                    placeholders = ['TODO', 'TBD', 'PLACEHOLDER', 'XXX']
                    for placeholder in placeholders:
                        assert placeholder not in value.upper(), f"Placeholder '{placeholder}' found in '{key}' in {file_path}"
                
                self._check_text_quality_recursive(value, file_path)
        elif isinstance(data, list):
            for item in data:
                self._check_text_quality_recursive(item, file_path)

class TestDataConsistency:
    """Test data consistency across files"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = DataValidator()
    
    def test_technique_id_consistency(self):
        """Test that technique IDs are consistent across files"""
        technique_refs = {}
        
        # Collect all technique references
        for file_path in self.validator.processed_dir.rglob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._collect_technique_refs(data, technique_refs, file_path)
        
        # Check for inconsistencies
        for tid, refs in technique_refs.items():
            if len(refs) > 1:
                names = {ref['name'] for ref in refs if 'name' in ref}
                if len(names) > 1:
                    pytest.fail(f"Inconsistent names for technique {tid}: {names}")
    
    def _collect_technique_refs(self, data: Any, refs: Dict, file_path: Path):
        """Recursively collect technique references"""
        if isinstance(data, dict):
            if 'technique_id' in data:
                tid = data['technique_id']
                if tid not in refs:
                    refs[tid] = []
                refs[tid].append({
                    'name': data.get('name'),
                    'file': str(file_path)
                })
            
            for value in data.values():
                self._collect_technique_refs(value, refs, file_path)
        elif isinstance(data, list):
            for item in data:
                self._collect_technique_refs(item, refs, file_path)

# Utility functions for manual validation
def validate_data_directory(data_dir: str = "src/data") -> Dict[str, Any]:
    """
    Manually validate data directory and return results
    """
    validator = DataValidator(data_dir)
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "directory": data_dir,
        "checks": {}
    }
    
    # Directory structure check
    try:
        assert validator.data_dir.exists()
        assert validator.raw_dir.exists()
        assert validator.processed_dir.exists()
        results["checks"]["directory_structure"] = {"status": "PASS", "message": "All directories exist"}
    except AssertionError:
        results["checks"]["directory_structure"] = {"status": "FAIL", "message": "Missing directories"}
    
    # File encoding check
    encoding_issues = []
    for data_dir_path in [validator.raw_dir, validator.processed_dir]:
        if data_dir_path.exists():
            for file_path in data_dir_path.rglob("*.txt"):
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)
                        if result['confidence'] <= 0.7:
                            encoding_issues.append(f"{file_path}: low confidence ({result['confidence']})")
                except Exception as e:
                    encoding_issues.append(f"{file_path}: {str(e)}")
    
    results["checks"]["encoding"] = {
        "status": "PASS" if not encoding_issues else "FAIL",
        "issues": encoding_issues
    }
    
    return results

if __name__ == "__main__":
    # Run validation manually
    results = validate_data_directory()
    print(json.dumps(results, indent=2))
