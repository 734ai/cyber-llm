"""
Automated Model Monitoring and Performance Tracking
Comprehensive system for monitoring deployed models, detecting drift, and performance degradation
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import os
import logging
import hashlib
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetrics:
    """Model performance metrics at a point in time"""
    model_id: str
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    prediction_latency_ms: float
    throughput_rps: float
    error_rate: float
    custom_metrics: Dict[str, float]

@dataclass
class DataDriftMetric:
    """Data drift measurement"""
    feature_name: str
    drift_score: float
    drift_method: str
    threshold: float
    is_drifting: bool
    timestamp: str

@dataclass
class ModelAlert:
    """Alert for model performance issues"""
    alert_id: str
    model_id: str
    alert_type: str  # drift, performance, error
    severity: str   # low, medium, high, critical
    message: str
    timestamp: str
    acknowledged: bool
    resolved: bool

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    model_id: str
    metric_name: str
    baseline_value: float
    threshold_lower: float
    threshold_upper: float
    created_at: str

class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "/home/o1/Desktop/cyber_llm/data/mlops/model_monitor.db"
        self.logger = logging.getLogger(__name__)
        self.alert_queue = deque(maxlen=1000)
        self._setup_database()
    
    def _setup_database(self):
        """Initialize monitoring database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            
            # Model metadata table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    model_version TEXT,
                    model_type TEXT,
                    deployed_at TEXT,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            
            # Performance metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    timestamp TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    prediction_latency_ms REAL,
                    throughput_rps REAL,
                    error_rate REAL,
                    custom_metrics TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Data drift table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    feature_name TEXT,
                    drift_score REAL,
                    drift_method TEXT,
                    threshold_value REAL,
                    is_drifting INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Model alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_alerts (
                    alert_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    timestamp TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Performance baselines table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    metric_name TEXT,
                    baseline_value REAL,
                    threshold_lower REAL,
                    threshold_upper REAL,
                    created_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Prediction logs table (sample for analysis)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    timestamp TEXT,
                    input_hash TEXT,
                    prediction TEXT,
                    confidence REAL,
                    actual_label TEXT,
                    is_correct INTEGER,
                    latency_ms REAL,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Create indices
            indices = [
                'CREATE INDEX IF NOT EXISTS idx_model_metrics_model_id ON model_metrics(model_id)',
                'CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_data_drift_model_id ON data_drift(model_id)',
                'CREATE INDEX IF NOT EXISTS idx_data_drift_timestamp ON data_drift(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_alerts_model_id ON model_alerts(model_id)',
                'CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON model_alerts(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_id ON prediction_logs(model_id)',
                'CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON prediction_logs(timestamp)'
            ]
            
            for index_sql in indices:
                conn.execute(index_sql)
            
            conn.commit()
    
    def register_model(self, model_id: str, model_name: str, model_version: str,
                      model_type: str, metadata: Dict[str, Any] = None) -> None:
        """Register a new model for monitoring"""
        metadata = metadata or {}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO models 
                (model_id, model_name, model_version, model_type, deployed_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                model_name,
                model_version,
                model_type,
                datetime.now().isoformat(),
                'active',
                json.dumps(metadata)
            ))
            conn.commit()
        
        self.logger.info(f"Registered model for monitoring: {model_id}")
    
    def log_metrics(self, model_id: str, metrics: ModelMetrics) -> None:
        """Log performance metrics for a model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_metrics 
                (model_id, timestamp, accuracy, precision_score, recall_score, f1_score,
                 auc_roc, prediction_latency_ms, throughput_rps, error_rate, custom_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                metrics.timestamp,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.auc_roc,
                metrics.prediction_latency_ms,
                metrics.throughput_rps,
                metrics.error_rate,
                json.dumps(metrics.custom_metrics)
            ))
            conn.commit()
        
        # Check for performance alerts
        self._check_performance_alerts(model_id, metrics)
    
    def log_prediction(self, model_id: str, input_data: Any, prediction: Any,
                      confidence: float, actual_label: Any = None,
                      latency_ms: float = None) -> None:
        """Log individual prediction for analysis"""
        # Create hash of input for privacy/efficiency
        input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
        
        is_correct = None
        if actual_label is not None:
            is_correct = int(prediction == actual_label)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO prediction_logs 
                (model_id, timestamp, input_hash, prediction, confidence, 
                 actual_label, is_correct, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                datetime.now().isoformat(),
                input_hash,
                str(prediction),
                confidence,
                str(actual_label) if actual_label is not None else None,
                is_correct,
                latency_ms
            ))
            conn.commit()
    
    def detect_data_drift(self, model_id: str, feature_data: Dict[str, np.ndarray],
                         baseline_data: Dict[str, np.ndarray] = None,
                         method: str = 'ks_test', threshold: float = 0.05) -> List[DataDriftMetric]:
        """Detect data drift in model features"""
        drift_metrics = []
        
        # If no baseline provided, use recent historical data
        if baseline_data is None:
            baseline_data = self._get_baseline_feature_data(model_id)
        
        for feature_name, current_data in feature_data.items():
            if feature_name not in baseline_data:
                continue
                
            baseline_feature_data = baseline_data[feature_name]
            
            # Calculate drift score based on method
            if method == 'ks_test':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(baseline_feature_data, current_data)
                drift_score = p_value
                is_drifting = p_value < threshold
            
            elif method == 'chi2_test':
                # Chi-square test for categorical features
                try:
                    # Create histograms for comparison
                    bins = min(len(np.unique(baseline_feature_data)), len(np.unique(current_data)), 10)
                    baseline_hist, bin_edges = np.histogram(baseline_feature_data, bins=bins)
                    current_hist, _ = np.histogram(current_data, bins=bin_edges)
                    
                    # Avoid zero frequencies
                    baseline_hist = baseline_hist + 1
                    current_hist = current_hist + 1
                    
                    statistic, p_value = stats.chisquare(current_hist, baseline_hist)
                    drift_score = p_value
                    is_drifting = p_value < threshold
                except:
                    drift_score = 1.0
                    is_drifting = False
            
            elif method == 'psi':
                # Population Stability Index
                drift_score = self._calculate_psi(baseline_feature_data, current_data)
                is_drifting = drift_score > threshold
            
            else:
                # Default to statistical distance
                drift_score = abs(np.mean(current_data) - np.mean(baseline_feature_data))
                is_drifting = drift_score > threshold
            
            drift_metric = DataDriftMetric(
                feature_name=feature_name,
                drift_score=drift_score,
                drift_method=method,
                threshold=threshold,
                is_drifting=is_drifting,
                timestamp=datetime.now().isoformat()
            )
            
            drift_metrics.append(drift_metric)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO data_drift 
                    (model_id, feature_name, drift_score, drift_method, threshold_value, is_drifting, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id,
                    feature_name,
                    drift_score,
                    method,
                    threshold,
                    int(is_drifting),
                    drift_metric.timestamp
                ))
                conn.commit()
            
            # Create alert if significant drift detected
            if is_drifting:
                self._create_alert(
                    model_id, 
                    'drift', 
                    'medium',
                    f"Data drift detected in feature '{feature_name}' (score: {drift_score:.4f})"
                )
        
        return drift_metrics
    
    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on baseline data
            bin_edges = np.linspace(baseline.min(), baseline.max(), buckets + 1)
            
            # Calculate distributions
            baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
            current_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to proportions
            baseline_prop = baseline_hist / len(baseline) + 1e-10  # Avoid division by zero
            current_prop = current_hist / len(current) + 1e-10
            
            # Calculate PSI
            psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
            return psi
        except:
            return 0.0
    
    def _get_baseline_feature_data(self, model_id: str, days_back: int = 30) -> Dict[str, np.ndarray]:
        """Get baseline feature data from historical predictions"""
        # This would typically pull from a feature store or logged data
        # For now, return dummy baseline data
        return {
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.exponential(2, 1000)
        }
    
    def set_performance_baseline(self, model_id: str, metric_name: str,
                                baseline_value: float, tolerance: float = 0.1) -> None:
        """Set performance baseline for monitoring"""
        threshold_lower = baseline_value * (1 - tolerance)
        threshold_upper = baseline_value * (1 + tolerance)
        
        baseline = PerformanceBaseline(
            model_id=model_id,
            metric_name=metric_name,
            baseline_value=baseline_value,
            threshold_lower=threshold_lower,
            threshold_upper=threshold_upper,
            created_at=datetime.now().isoformat()
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_baselines 
                (model_id, metric_name, baseline_value, threshold_lower, threshold_upper, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                baseline.model_id,
                baseline.metric_name,
                baseline.baseline_value,
                baseline.threshold_lower,
                baseline.threshold_upper,
                baseline.created_at
            ))
            conn.commit()
    
    def _check_performance_alerts(self, model_id: str, metrics: ModelMetrics) -> None:
        """Check if performance metrics trigger alerts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT metric_name, baseline_value, threshold_lower, threshold_upper
                FROM performance_baselines
                WHERE model_id = ?
            ''', (model_id,))
            
            baselines = cursor.fetchall()
            
            for metric_name, baseline_value, threshold_lower, threshold_upper in baselines:
                current_value = getattr(metrics, metric_name.replace('_score', ''), None)
                if current_value is None:
                    current_value = metrics.custom_metrics.get(metric_name, baseline_value)
                
                if current_value < threshold_lower:
                    severity = 'high' if current_value < baseline_value * 0.8 else 'medium'
                    self._create_alert(
                        model_id, 
                        'performance', 
                        severity,
                        f"Performance degradation: {metric_name} dropped to {current_value:.4f} (baseline: {baseline_value:.4f})"
                    )
                elif current_value > threshold_upper and metric_name in ['error_rate']:
                    severity = 'high' if current_value > baseline_value * 1.5 else 'medium'
                    self._create_alert(
                        model_id, 
                        'performance', 
                        severity,
                        f"Performance degradation: {metric_name} increased to {current_value:.4f} (baseline: {baseline_value:.4f})"
                    )
    
    def _create_alert(self, model_id: str, alert_type: str, severity: str, message: str) -> str:
        """Create a new alert"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        alert = ModelAlert(
            alert_id=alert_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now().isoformat(),
            acknowledged=False,
            resolved=False
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_alerts 
                (alert_id, model_id, alert_type, severity, message, timestamp, acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.model_id,
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.timestamp,
                int(alert.acknowledged),
                int(alert.resolved)
            ))
            conn.commit()
        
        # Add to alert queue
        self.alert_queue.append(alert)
        
        self.logger.warning(f"Alert created: {alert_type} - {severity} - {message}")
        return alert_id
    
    def get_model_health_report(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive model health report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        report = {
            "model_id": model_id,
            "report_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "generated_at": end_time.isoformat(),
            "health_score": 0.0,
            "performance_metrics": {},
            "drift_analysis": {},
            "alerts_summary": {},
            "recommendations": []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent performance metrics
            cursor = conn.execute('''
                SELECT accuracy, precision_score, recall_score, f1_score, 
                       prediction_latency_ms, throughput_rps, error_rate
                FROM model_metrics 
                WHERE model_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (model_id, start_time.isoformat()))
            
            metrics_data = cursor.fetchall()
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data, columns=[
                    'accuracy', 'precision', 'recall', 'f1_score', 
                    'latency_ms', 'throughput_rps', 'error_rate'
                ])
                
                # Calculate average performance
                report["performance_metrics"] = {
                    "avg_accuracy": float(metrics_df['accuracy'].mean()),
                    "avg_precision": float(metrics_df['precision'].mean()),
                    "avg_recall": float(metrics_df['recall'].mean()),
                    "avg_f1_score": float(metrics_df['f1_score'].mean()),
                    "avg_latency_ms": float(metrics_df['latency_ms'].mean()),
                    "avg_throughput_rps": float(metrics_df['throughput_rps'].mean()),
                    "avg_error_rate": float(metrics_df['error_rate'].mean()),
                    "performance_trend": "stable"  # Could implement trend analysis
                }
            
            # Get drift information
            cursor = conn.execute('''
                SELECT feature_name, drift_score, is_drifting
                FROM data_drift 
                WHERE model_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (model_id, start_time.isoformat()))
            
            drift_data = cursor.fetchall()
            if drift_data:
                total_features = len(set(row[0] for row in drift_data))
                drifting_features = len(set(row[0] for row in drift_data if row[2]))
                
                report["drift_analysis"] = {
                    "total_features_monitored": total_features,
                    "features_with_drift": drifting_features,
                    "drift_percentage": (drifting_features / total_features * 100) if total_features > 0 else 0,
                    "max_drift_score": max(row[1] for row in drift_data) if drift_data else 0
                }
            
            # Get alerts summary
            cursor = conn.execute('''
                SELECT alert_type, severity, COUNT(*) as count
                FROM model_alerts 
                WHERE model_id = ? AND timestamp >= ?
                GROUP BY alert_type, severity
            ''', (model_id, start_time.isoformat()))
            
            alert_summary = {}
            total_alerts = 0
            for alert_type, severity, count in cursor.fetchall():
                if alert_type not in alert_summary:
                    alert_summary[alert_type] = {}
                alert_summary[alert_type][severity] = count
                total_alerts += count
            
            report["alerts_summary"] = {
                "total_alerts": total_alerts,
                "by_type_and_severity": alert_summary
            }
        
        # Calculate health score (0-100)
        health_score = 100.0
        
        # Reduce score for poor performance
        if report["performance_metrics"]:
            avg_accuracy = report["performance_metrics"]["avg_accuracy"]
            avg_error_rate = report["performance_metrics"]["avg_error_rate"]
            
            if avg_accuracy < 0.8:
                health_score -= (0.8 - avg_accuracy) * 100
            if avg_error_rate > 0.1:
                health_score -= (avg_error_rate - 0.1) * 200
        
        # Reduce score for data drift
        if report["drift_analysis"]:
            drift_penalty = report["drift_analysis"]["drift_percentage"] * 0.5
            health_score -= drift_penalty
        
        # Reduce score for alerts
        critical_alerts = sum(
            alert_summary.get(alert_type, {}).get('critical', 0) 
            for alert_type in alert_summary
        )
        high_alerts = sum(
            alert_summary.get(alert_type, {}).get('high', 0) 
            for alert_type in alert_summary
        )
        
        health_score -= critical_alerts * 10 + high_alerts * 5
        
        report["health_score"] = max(0.0, min(100.0, health_score))
        
        # Generate recommendations
        if report["health_score"] < 70:
            report["recommendations"].append("Model health is concerning - investigate performance issues")
        if report["drift_analysis"].get("drift_percentage", 0) > 20:
            report["recommendations"].append("Significant data drift detected - consider retraining")
        if report["performance_metrics"].get("avg_error_rate", 0) > 0.1:
            report["recommendations"].append("High error rate - review model and input data quality")
        if critical_alerts > 0:
            report["recommendations"].append("Critical alerts present - immediate attention required")
        
        return report
    
    def get_alerts(self, model_id: str = None, severity: str = None, 
                   unresolved_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """Get model alerts with filtering options"""
        query = 'SELECT * FROM model_alerts WHERE 1=1'
        params = []
        
        if model_id:
            query += ' AND model_id = ?'
            params.append(model_id)
        
        if severity:
            query += ' AND severity = ?'
            params.append(severity)
        
        if unresolved_only:
            query += ' AND resolved = 0'
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            alerts = []
            for row in cursor.fetchall():
                alert = dict(zip(columns, row))
                alert['acknowledged'] = bool(alert['acknowledged'])
                alert['resolved'] = bool(alert['resolved'])
                alerts.append(alert)
            
            return alerts
    
    def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an alert"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE model_alerts 
                SET acknowledged = 1
                WHERE alert_id = ?
            ''', (alert_id,))
            conn.commit()
    
    def resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE model_alerts 
                SET resolved = 1, acknowledged = 1
                WHERE alert_id = ?
            ''', (alert_id,))
            conn.commit()

# Example usage and testing
if __name__ == "__main__":
    # Initialize model monitor
    monitor = ModelMonitor()
    
    print("📊 Model Monitoring System Testing:")
    print("=" * 50)
    
    # Register test models
    model_ids = ["threat_cnn_001", "anomaly_lstm_001"]
    
    monitor.register_model(
        "threat_cnn_001", 
        "Threat Detection CNN", 
        "v1.0", 
        "cnn",
        metadata={"purpose": "malware detection", "training_date": "2024-01-15"}
    )
    
    monitor.register_model(
        "anomaly_lstm_001", 
        "Network Anomaly LSTM", 
        "v1.0", 
        "lstm",
        metadata={"purpose": "network anomaly detection", "training_date": "2024-01-20"}
    )
    
    # Set performance baselines
    print("\n🎯 Setting performance baselines...")
    monitor.set_performance_baseline("threat_cnn_001", "accuracy", 0.95, tolerance=0.05)
    monitor.set_performance_baseline("threat_cnn_001", "error_rate", 0.02, tolerance=0.5)
    monitor.set_performance_baseline("anomaly_lstm_001", "accuracy", 0.92, tolerance=0.05)
    
    # Simulate monitoring data
    print("\n📈 Logging performance metrics...")
    for i in range(10):
        # Good performance metrics
        metrics1 = ModelMetrics(
            model_id="threat_cnn_001",
            timestamp=datetime.now().isoformat(),
            accuracy=0.94 + np.random.normal(0, 0.01),
            precision=0.93 + np.random.normal(0, 0.01),
            recall=0.92 + np.random.normal(0, 0.01),
            f1_score=0.925 + np.random.normal(0, 0.01),
            auc_roc=0.96 + np.random.normal(0, 0.005),
            prediction_latency_ms=45 + np.random.normal(0, 5),
            throughput_rps=100 + np.random.normal(0, 10),
            error_rate=0.02 + np.random.normal(0, 0.005),
            custom_metrics={"cyber_threat_score": 0.87 + np.random.normal(0, 0.02)}
        )
        
        # Degrading performance for second model
        performance_decay = 0.05 * i  # Gradually decrease performance
        metrics2 = ModelMetrics(
            model_id="anomaly_lstm_001",
            timestamp=datetime.now().isoformat(),
            accuracy=0.92 - performance_decay + np.random.normal(0, 0.01),
            precision=0.90 - performance_decay + np.random.normal(0, 0.01),
            recall=0.89 - performance_decay + np.random.normal(0, 0.01),
            f1_score=0.895 - performance_decay + np.random.normal(0, 0.01),
            auc_roc=0.93 - performance_decay + np.random.normal(0, 0.005),
            prediction_latency_ms=60 + i * 5 + np.random.normal(0, 5),  # Increasing latency
            throughput_rps=80 - i * 2 + np.random.normal(0, 5),  # Decreasing throughput
            error_rate=0.03 + i * 0.01 + np.random.normal(0, 0.005),  # Increasing errors
            custom_metrics={"anomaly_detection_score": 0.85 - performance_decay}
        )
        
        monitor.log_metrics("threat_cnn_001", metrics1)
        monitor.log_metrics("anomaly_lstm_001", metrics2)
    
    # Test data drift detection
    print("\n🌊 Testing data drift detection...")
    # Simulate feature data
    baseline_features = {
        'packet_size': np.random.normal(1000, 200, 1000),
        'connection_duration': np.random.exponential(5, 1000),
        'port_number': np.random.choice(range(1, 65536), 1000)
    }
    
    # Simulate drifted current data
    current_features = {
        'packet_size': np.random.normal(1200, 300, 500),  # Different distribution
        'connection_duration': np.random.exponential(8, 500),  # Different parameter
        'port_number': np.random.choice(range(1, 65536), 500)  # Same distribution
    }
    
    drift_metrics = monitor.detect_data_drift(
        "anomaly_lstm_001", 
        current_features, 
        baseline_features,
        method='ks_test',
        threshold=0.05
    )
    
    print(f"  Detected drift in {sum(1 for m in drift_metrics if m.is_drifting)} out of {len(drift_metrics)} features")
    for metric in drift_metrics:
        status = "🚨 DRIFT" if metric.is_drifting else "✅ OK"
        print(f"    {metric.feature_name}: {status} (score: {metric.drift_score:.4f})")
    
    # Generate health reports
    print("\n🏥 Generating model health reports...")
    for model_id in model_ids:
        report = monitor.get_model_health_report(model_id, days=7)
        print(f"\n  {model_id}:")
        print(f"    Health Score: {report['health_score']:.1f}/100")
        print(f"    Alerts: {report['alerts_summary']['total_alerts']}")
        if report['performance_metrics']:
            print(f"    Avg Accuracy: {report['performance_metrics']['avg_accuracy']:.3f}")
            print(f"    Avg Error Rate: {report['performance_metrics']['avg_error_rate']:.3f}")
        if report['recommendations']:
            print(f"    Recommendations: {len(report['recommendations'])}")
            for rec in report['recommendations'][:2]:  # Show first 2
                print(f"      - {rec}")
    
    # Show recent alerts
    print("\n🚨 Recent Alerts:")
    alerts = monitor.get_alerts(limit=10)
    for alert in alerts[:5]:  # Show top 5
        print(f"  - {alert['severity'].upper()}: {alert['message']}")
    
    print(f"\n✅ Model Monitoring System implemented - {len(alerts)} alerts generated")
