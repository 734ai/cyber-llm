"""
Automated Data Quality Monitoring System
Monitors data quality metrics, detects anomalies, and ensures data integrity
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
import statistics

class QualityMetricType(Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityMetric:
    """Represents a data quality metric measurement"""
    metric_id: str
    dataset_id: str
    metric_type: QualityMetricType
    value: float
    threshold_min: float
    threshold_max: float
    measured_at: str
    passed: bool
    details: Dict[str, Any]

@dataclass
class QualityAlert:
    """Represents a data quality alert"""
    alert_id: str
    dataset_id: str
    metric_type: QualityMetricType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    created_at: str
    resolved_at: Optional[str]
    resolved: bool

@dataclass
class DatasetProfile:
    """Statistical profile of a dataset"""
    dataset_id: str
    total_records: int
    total_columns: int
    null_percentage: float
    duplicate_percentage: float
    schema_hash: str
    last_updated: str
    column_profiles: Dict[str, Any]

class DataQualityMonitor:
    """Automated data quality monitoring system"""
    
    def __init__(self, db_path: str = "data/quality/data_quality.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.quality_thresholds = self._load_default_thresholds()
        
    def _init_database(self):
        """Initialize the quality monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Quality Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                metric_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                threshold_min REAL NOT NULL,
                threshold_max REAL NOT NULL,
                measured_at TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                details TEXT NOT NULL
            )
        """)
        
        # Quality Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_alerts (
                alert_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Dataset Profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_profiles (
                dataset_id TEXT PRIMARY KEY,
                total_records INTEGER NOT NULL,
                total_columns INTEGER NOT NULL,
                null_percentage REAL NOT NULL,
                duplicate_percentage REAL NOT NULL,
                schema_hash TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                column_profiles TEXT NOT NULL
            )
        """)
        
        # Quality Rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_rules (
                rule_id TEXT PRIMARY KEY,
                dataset_pattern TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                threshold_min REAL,
                threshold_max REAL,
                severity TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_dataset ON quality_metrics(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON quality_metrics(metric_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_dataset ON quality_alerts(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON quality_alerts(severity)")
        
        conn.commit()
        conn.close()
    
    def _load_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load default quality thresholds for cybersecurity data"""
        return {
            "mitre_attack": {
                "completeness": {"min": 0.95, "max": 1.0},
                "accuracy": {"min": 0.90, "max": 1.0},
                "consistency": {"min": 0.85, "max": 1.0},
                "validity": {"min": 0.95, "max": 1.0},
                "uniqueness": {"min": 0.98, "max": 1.0}
            },
            "cve_data": {
                "completeness": {"min": 0.90, "max": 1.0},
                "accuracy": {"min": 0.95, "max": 1.0},
                "timeliness": {"min": 0.80, "max": 1.0},
                "validity": {"min": 0.95, "max": 1.0}
            },
            "threat_intel": {
                "completeness": {"min": 0.85, "max": 1.0},
                "accuracy": {"min": 0.90, "max": 1.0},
                "timeliness": {"min": 0.90, "max": 1.0},
                "relevance": {"min": 0.80, "max": 1.0}
            },
            "red_team_logs": {
                "completeness": {"min": 0.98, "max": 1.0},
                "consistency": {"min": 0.90, "max": 1.0},
                "validity": {"min": 0.95, "max": 1.0}
            }
        }
    
    def measure_completeness(self, data: pd.DataFrame) -> float:
        """Measure data completeness (percentage of non-null values)"""
        if data.empty:
            return 0.0
        
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = total_cells - data.isnull().sum().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    def measure_accuracy(self, data: pd.DataFrame, dataset_type: str) -> float:
        """Measure data accuracy based on validation rules"""
        if data.empty:
            return 0.0
        
        accuracy_score = 1.0
        total_checks = 0
        failed_checks = 0
        
        # Cybersecurity-specific accuracy checks
        if dataset_type == "mitre_attack":
            # Check technique ID format
            if 'technique_id' in data.columns:
                technique_pattern = re.compile(r'^T\d{4}(\.\d{3})?$')
                invalid_ids = ~data['technique_id'].str.match(technique_pattern, na=False)
                failed_checks += invalid_ids.sum()
                total_checks += len(data)
        
        elif dataset_type == "cve_data":
            # Check CVE ID format
            if 'cve_id' in data.columns:
                cve_pattern = re.compile(r'^CVE-\d{4}-\d{4,}$')
                invalid_cves = ~data['cve_id'].str.match(cve_pattern, na=False)
                failed_checks += invalid_cves.sum()
                total_checks += len(data)
        
        elif dataset_type == "threat_intel":
            # Check IP address format
            if 'ip_address' in data.columns:
                ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
                invalid_ips = ~data['ip_address'].str.match(ip_pattern, na=False)
                failed_checks += invalid_ips.sum()
                total_checks += len(data)
        
        # General accuracy checks
        for column in data.select_dtypes(include=['object']).columns:
            # Check for suspicious patterns
            suspicious_patterns = ['<script>', 'javascript:', 'null', 'undefined', 'NaN']
            for pattern in suspicious_patterns:
                if data[column].astype(str).str.contains(pattern, case=False, na=False).any():
                    failed_checks += data[column].astype(str).str.contains(pattern, case=False, na=False).sum()
                    total_checks += len(data)
        
        if total_checks > 0:
            accuracy_score = (total_checks - failed_checks) / total_checks
        
        return max(0.0, min(1.0, accuracy_score))
    
    def measure_consistency(self, data: pd.DataFrame) -> float:
        """Measure data consistency across columns and records"""
        if data.empty:
            return 0.0
        
        consistency_score = 1.0
        consistency_checks = 0
        failed_consistency = 0
        
        # Check data type consistency within columns
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check for mixed data types in string columns
                non_null_values = data[column].dropna()
                if len(non_null_values) > 0:
                    # Simple heuristic: check if values look like different data types
                    numeric_count = sum(str(val).replace('.', '').replace('-', '').isdigit() 
                                      for val in non_null_values)
                    if 0 < numeric_count < len(non_null_values):
                        failed_consistency += 1
                    consistency_checks += 1
        
        # Check for consistent naming conventions
        string_columns = data.select_dtypes(include=['object']).columns
        for column in string_columns:
            values = data[column].dropna().astype(str)
            if len(values) > 0:
                # Check case consistency
                upper_count = sum(val.isupper() for val in values if val.isalpha())
                lower_count = sum(val.islower() for val in values if val.isalpha())
                mixed_count = len(values) - upper_count - lower_count
                
                if mixed_count > 0 and (upper_count > 0 or lower_count > 0):
                    # Mixed case inconsistency
                    consistency_ratio = 1 - (mixed_count / len(values))
                    if consistency_ratio < 0.8:
                        failed_consistency += 1
                consistency_checks += 1
        
        if consistency_checks > 0:
            consistency_score = (consistency_checks - failed_consistency) / consistency_checks
        
        return max(0.0, min(1.0, consistency_score))
    
    def measure_validity(self, data: pd.DataFrame, dataset_type: str) -> float:
        """Measure data validity based on domain-specific rules"""
        if data.empty:
            return 0.0
        
        validity_score = 1.0
        total_validations = 0
        failed_validations = 0
        
        # Cybersecurity-specific validity checks
        if dataset_type == "threat_intel":
            # Validate confidence scores
            if 'confidence' in data.columns:
                invalid_confidence = (data['confidence'] < 0) | (data['confidence'] > 100)
                failed_validations += invalid_confidence.sum()
                total_validations += len(data)
            
            # Validate severity levels
            if 'severity' in data.columns:
                valid_severities = ['low', 'medium', 'high', 'critical']
                invalid_severity = ~data['severity'].str.lower().isin(valid_severities)
                failed_validations += invalid_severity.sum()
                total_validations += len(data)
        
        elif dataset_type == "cve_data":
            # Validate CVSS scores
            if 'cvss_score' in data.columns:
                invalid_cvss = (data['cvss_score'] < 0) | (data['cvss_score'] > 10)
                failed_validations += invalid_cvss.sum()
                total_validations += len(data)
        
        # General validity checks
        for column in data.select_dtypes(include=['int64', 'float64']).columns:
            # Check for unrealistic values (e.g., negative counts where they shouldn't be)
            if 'count' in column.lower() or 'number' in column.lower():
                negative_values = data[column] < 0
                failed_validations += negative_values.sum()
                total_validations += len(data)
        
        if total_validations > 0:
            validity_score = (total_validations - failed_validations) / total_validations
        
        return max(0.0, min(1.0, validity_score))
    
    def measure_uniqueness(self, data: pd.DataFrame) -> float:
        """Measure data uniqueness (percentage of unique records)"""
        if data.empty:
            return 1.0
        
        total_records = len(data)
        unique_records = len(data.drop_duplicates())
        return unique_records / total_records if total_records > 0 else 1.0
    
    def measure_timeliness(self, data: pd.DataFrame, dataset_type: str) -> float:
        """Measure data timeliness based on timestamps"""
        if data.empty:
            return 0.0
        
        # Look for timestamp columns
        timestamp_columns = []
        for column in data.columns:
            if any(keyword in column.lower() for keyword in ['time', 'date', 'created', 'updated']):
                try:
                    pd.to_datetime(data[column].dropna().iloc[0])
                    timestamp_columns.append(column)
                except:
                    continue
        
        if not timestamp_columns:
            return 1.0  # No timestamp data to evaluate
        
        # Calculate timeliness based on most recent timestamp
        most_recent_col = timestamp_columns[0]
        try:
            timestamps = pd.to_datetime(data[most_recent_col].dropna())
            if len(timestamps) == 0:
                return 0.0
            
            now = datetime.now()
            max_age_days = 30  # Consider data stale after 30 days for cybersecurity
            
            # Calculate age of most recent record
            most_recent = timestamps.max()
            age_days = (now - most_recent).days
            
            # Timeliness score: 1.0 for fresh data, decreasing with age
            timeliness_score = max(0.0, 1.0 - (age_days / max_age_days))
            return timeliness_score
            
        except Exception:
            return 0.0
    
    def measure_relevance(self, data: pd.DataFrame, dataset_type: str) -> float:
        """Measure data relevance based on content analysis"""
        if data.empty:
            return 0.0
        
        relevance_score = 1.0
        
        # Cybersecurity-specific relevance checks
        cybersec_keywords = [
            'attack', 'threat', 'vulnerability', 'exploit', 'malware',
            'phishing', 'breach', 'intrusion', 'security', 'defense',
            'detection', 'prevention', 'mitigation', 'incident'
        ]
        
        text_columns = data.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            total_relevance = 0
            relevance_checks = 0
            
            for column in text_columns:
                text_data = data[column].dropna().astype(str).str.lower()
                if len(text_data) > 0:
                    # Count records containing cybersecurity keywords
                    relevant_records = 0
                    for text in text_data:
                        if any(keyword in text for keyword in cybersec_keywords):
                            relevant_records += 1
                    
                    column_relevance = relevant_records / len(text_data)
                    total_relevance += column_relevance
                    relevance_checks += 1
            
            if relevance_checks > 0:
                relevance_score = total_relevance / relevance_checks
        
        return max(0.0, min(1.0, relevance_score))
    
    def create_dataset_profile(self, dataset_id: str, data: pd.DataFrame) -> DatasetProfile:
        """Create a statistical profile of a dataset"""
        if data.empty:
            return DatasetProfile(
                dataset_id=dataset_id,
                total_records=0,
                total_columns=0,
                null_percentage=1.0,
                duplicate_percentage=0.0,
                schema_hash="",
                last_updated=datetime.now().isoformat(),
                column_profiles={}
            )
        
        # Calculate basic statistics
        total_records = len(data)
        total_columns = len(data.columns)
        null_percentage = data.isnull().sum().sum() / (total_records * total_columns)
        duplicate_percentage = (total_records - len(data.drop_duplicates())) / total_records
        
        # Create schema hash
        schema_info = f"{list(data.columns)}_{list(data.dtypes)}"
        schema_hash = hashlib.md5(schema_info.encode()).hexdigest()
        
        # Profile each column
        column_profiles = {}
        for column in data.columns:
            col_data = data[column]
            profile = {
                "data_type": str(col_data.dtype),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(col_data.isnull().sum() / len(col_data)),
                "unique_count": int(col_data.nunique()),
                "unique_percentage": float(col_data.nunique() / len(col_data))
            }
            
            if col_data.dtype in ['int64', 'float64']:
                profile.update({
                    "min": float(col_data.min()) if not col_data.isna().all() else None,
                    "max": float(col_data.max()) if not col_data.isna().all() else None,
                    "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                    "std": float(col_data.std()) if not col_data.isna().all() else None
                })
            elif col_data.dtype == 'object':
                profile.update({
                    "avg_length": float(col_data.astype(str).str.len().mean()) if not col_data.isna().all() else None,
                    "max_length": int(col_data.astype(str).str.len().max()) if not col_data.isna().all() else None
                })
            
            column_profiles[column] = profile
        
        return DatasetProfile(
            dataset_id=dataset_id,
            total_records=total_records,
            total_columns=total_columns,
            null_percentage=null_percentage,
            duplicate_percentage=duplicate_percentage,
            schema_hash=schema_hash,
            last_updated=datetime.now().isoformat(),
            column_profiles=column_profiles
        )
    
    def monitor_dataset(self, dataset_id: str, data: pd.DataFrame, dataset_type: str) -> List[QualityMetric]:
        """Monitor a dataset and return quality metrics"""
        metrics = []
        timestamp = datetime.now().isoformat()
        
        # Get thresholds for this dataset type
        thresholds = self.quality_thresholds.get(dataset_type, {})
        
        # Measure each quality dimension
        quality_measures = {
            QualityMetricType.COMPLETENESS: self.measure_completeness(data),
            QualityMetricType.ACCURACY: self.measure_accuracy(data, dataset_type),
            QualityMetricType.CONSISTENCY: self.measure_consistency(data),
            QualityMetricType.VALIDITY: self.measure_validity(data, dataset_type),
            QualityMetricType.UNIQUENESS: self.measure_uniqueness(data),
            QualityMetricType.TIMELINESS: self.measure_timeliness(data, dataset_type),
            QualityMetricType.RELEVANCE: self.measure_relevance(data, dataset_type)
        }
        
        # Create quality metrics
        for metric_type, value in quality_measures.items():
            metric_name = metric_type.value
            threshold = thresholds.get(metric_name, {"min": 0.8, "max": 1.0})
            
            metric = QualityMetric(
                metric_id=f"{dataset_id}_{metric_name}_{timestamp.replace(':', '')}",
                dataset_id=dataset_id,
                metric_type=metric_type,
                value=value,
                threshold_min=threshold["min"],
                threshold_max=threshold["max"],
                measured_at=timestamp,
                passed=threshold["min"] <= value <= threshold["max"],
                details={
                    "dataset_type": dataset_type,
                    "threshold_min": threshold["min"],
                    "threshold_max": threshold["max"],
                    "measurement_context": f"Automated monitoring at {timestamp}"
                }
            )
            
            metrics.append(metric)
            
            # Store metric in database
            self._store_metric(metric)
            
            # Check if alert should be generated
            if not metric.passed:
                self._generate_alert(metric)
        
        # Create and store dataset profile
        profile = self.create_dataset_profile(dataset_id, data)
        self._store_profile(profile)
        
        return metrics
    
    def _store_metric(self, metric: QualityMetric):
        """Store a quality metric in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO quality_metrics 
            (metric_id, dataset_id, metric_type, value, threshold_min, threshold_max,
             measured_at, passed, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.metric_id, metric.dataset_id, metric.metric_type.value,
            metric.value, metric.threshold_min, metric.threshold_max,
            metric.measured_at, metric.passed, json.dumps(metric.details)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_profile(self, profile: DatasetProfile):
        """Store a dataset profile in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO dataset_profiles 
            (dataset_id, total_records, total_columns, null_percentage,
             duplicate_percentage, schema_hash, last_updated, column_profiles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.dataset_id, profile.total_records, profile.total_columns,
            profile.null_percentage, profile.duplicate_percentage,
            profile.schema_hash, profile.last_updated, json.dumps(profile.column_profiles)
        ))
        
        conn.commit()
        conn.close()
    
    def _generate_alert(self, metric: QualityMetric):
        """Generate a quality alert for a failed metric"""
        # Determine severity based on how far the value is from threshold
        if metric.value < metric.threshold_min:
            deviation = (metric.threshold_min - metric.value) / metric.threshold_min
        else:
            deviation = (metric.value - metric.threshold_max) / metric.threshold_max
        
        if deviation > 0.5:
            severity = AlertSeverity.CRITICAL
        elif deviation > 0.3:
            severity = AlertSeverity.HIGH
        elif deviation > 0.1:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        alert = QualityAlert(
            alert_id=f"alert_{metric.metric_id}",
            dataset_id=metric.dataset_id,
            metric_type=metric.metric_type,
            severity=severity,
            message=f"{metric.metric_type.value} quality check failed: {metric.value:.3f} outside threshold [{metric.threshold_min}, {metric.threshold_max}]",
            value=metric.value,
            threshold=metric.threshold_min if metric.value < metric.threshold_min else metric.threshold_max,
            created_at=datetime.now().isoformat(),
            resolved_at=None,
            resolved=False
        )
        
        # Store alert in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO quality_alerts 
            (alert_id, dataset_id, metric_type, severity, message, value,
             threshold, created_at, resolved_at, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.dataset_id, alert.metric_type.value,
            alert.severity.value, alert.message, alert.value,
            alert.threshold, alert.created_at, alert.resolved_at, alert.resolved
        ))
        
        conn.commit()
        conn.close()
    
    def generate_quality_report(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive data quality report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "scope": "all_datasets" if dataset_id is None else f"dataset_{dataset_id}",
            "summary": {},
            "metrics_summary": {},
            "alerts_summary": {},
            "recommendations": []
        }
        
        # Build WHERE clause for dataset filtering
        where_clause = ""
        params = []
        if dataset_id:
            where_clause = "WHERE dataset_id = ?"
            params.append(dataset_id)
        
        # Summary statistics
        cursor.execute(f"SELECT COUNT(DISTINCT dataset_id) FROM quality_metrics {where_clause}", params)
        total_datasets = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM quality_metrics {where_clause}", params)
        total_metrics = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM quality_alerts {where_clause} AND resolved = 0", params)
        active_alerts = cursor.fetchone()[0]
        
        report["summary"] = {
            "total_datasets": total_datasets,
            "total_metrics": total_metrics,
            "active_alerts": active_alerts
        }
        
        # Metrics summary by type
        cursor.execute(f"""
            SELECT metric_type, 
                   COUNT(*) as count,
                   AVG(value) as avg_value,
                   MIN(value) as min_value,
                   MAX(value) as max_value,
                   SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pass_rate
            FROM quality_metrics {where_clause}
            GROUP BY metric_type
        """, params)
        
        for row in cursor.fetchall():
            report["metrics_summary"][row[0]] = {
                "count": row[1],
                "average_value": row[2],
                "min_value": row[3],
                "max_value": row[4],
                "pass_rate": row[5]
            }
        
        # Alerts summary by severity
        cursor.execute(f"""
            SELECT severity, COUNT(*) as count
            FROM quality_alerts {where_clause} AND resolved = 0
            GROUP BY severity
        """, params)
        
        for row in cursor.fetchall():
            report["alerts_summary"][row[0]] = row[1]
        
        # Generate recommendations
        for metric_type, stats in report["metrics_summary"].items():
            if stats["pass_rate"] < 90:
                report["recommendations"].append(
                    f"Improve {metric_type} quality (current pass rate: {stats['pass_rate']:.1f}%)"
                )
        
        if report["summary"]["active_alerts"] > 0:
            report["recommendations"].append(
                f"Address {report['summary']['active_alerts']} active quality alerts"
            )
        
        conn.close()
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize the monitor
    monitor = DataQualityMonitor("data/quality/data_quality.db")
    
    # Create sample cybersecurity data for testing
    sample_data = pd.DataFrame({
        'technique_id': ['T1001', 'T1002', 'T1003', 'INVALID', 'T1005'],
        'technique_name': ['Data Obfuscation', 'Data Compressed', 'OS Credential Dumping', 'Test', 'Data from Local System'],
        'confidence': [95, 87, 92, 150, 88],  # 150 is invalid (out of range)
        'severity': ['high', 'medium', 'high', 'unknown', 'medium'],  # 'unknown' is invalid
        'last_updated': ['2024-08-01', '2024-08-02', '2024-07-15', '2024-08-03', '2024-08-01']
    })
    
    # Monitor the dataset
    metrics = monitor.monitor_dataset("test_mitre_data", sample_data, "mitre_attack")
    
    print("Quality Metrics:")
    for metric in metrics:
        status = "✅ PASS" if metric.passed else "❌ FAIL"
        print(f"  {metric.metric_type.value}: {metric.value:.3f} {status}")
    
    # Generate quality report
    report = monitor.generate_quality_report("test_mitre_data")
    print("\nQuality Report:")
    print(json.dumps(report, indent=2))
    
    print("✅ Automated Data Quality Monitoring System implemented and tested")
