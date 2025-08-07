"""
MLOps Experiment Tracking and Management System
Comprehensive experiment tracking, versioning, and analysis for cybersecurity AI models
"""

import json
import sqlite3
import hashlib
import uuid
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import numpy as np
import logging

@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment"""
    experiment_id: str
    name: str
    description: str
    tags: List[str]
    parameters: Dict[str, Any]
    model_type: str
    dataset_info: Dict[str, Any]
    environment_info: Dict[str, Any]
    created_at: str
    created_by: str

@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment"""
    experiment_id: str
    step: int
    timestamp: str
    metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    custom_metrics: Dict[str, Any]

@dataclass
class ExperimentArtifact:
    """Artifact associated with an experiment"""
    artifact_id: str
    experiment_id: str
    name: str
    type: str  # model, dataset, plot, log, etc.
    path: str
    metadata: Dict[str, Any]
    checksum: str
    size_bytes: int
    created_at: str

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    experiment_id: str
    model_name: str
    version: str
    status: str  # training, validation, deployed, archived
    performance_metrics: Dict[str, float]
    model_path: str
    metadata: Dict[str, Any]
    created_at: str

class ExperimentTracker:
    """Comprehensive MLOps experiment tracking system"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "/home/o1/Desktop/cyber_llm/data/mlops/experiments.db"
        self.logger = logging.getLogger(__name__)
        self._setup_database()
        
    def _setup_database(self):
        """Initialize the experiment tracking database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            
            # Experiments table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,  -- JSON array
                    parameters TEXT,  -- JSON object
                    model_type TEXT,
                    dataset_info TEXT,  -- JSON object
                    environment_info TEXT,  -- JSON object
                    status TEXT DEFAULT 'active',
                    created_at TEXT,
                    created_by TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Experiment metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    step INTEGER,
                    timestamp TEXT,
                    metrics TEXT,  -- JSON object
                    validation_metrics TEXT,  -- JSON object
                    custom_metrics TEXT,  -- JSON object
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Artifacts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    type TEXT,
                    path TEXT,
                    metadata TEXT,  -- JSON object
                    checksum TEXT,
                    size_bytes INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Model versions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    model_name TEXT,
                    version TEXT,
                    status TEXT,
                    performance_metrics TEXT,  -- JSON object
                    model_path TEXT,
                    metadata TEXT,  -- JSON object
                    created_at TEXT,
                    deployed_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Experiment comparisons table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    name TEXT,
                    experiment_ids TEXT,  -- JSON array
                    comparison_metrics TEXT,  -- JSON object
                    notes TEXT,
                    created_at TEXT
                )
            ''')
            
            # Create indices for performance
            indices = [
                'CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)',
                'CREATE INDEX IF NOT EXISTS idx_experiments_model_type ON experiments(model_type)',
                'CREATE INDEX IF NOT EXISTS idx_experiments_tags ON experiments(tags)',
                'CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id ON experiment_metrics(experiment_id)',
                'CREATE INDEX IF NOT EXISTS idx_metrics_step ON experiment_metrics(step)',
                'CREATE INDEX IF NOT EXISTS idx_artifacts_experiment_id ON experiment_artifacts(experiment_id)',
                'CREATE INDEX IF NOT EXISTS idx_artifacts_type ON experiment_artifacts(type)',
                'CREATE INDEX IF NOT EXISTS idx_model_versions_experiment_id ON model_versions(experiment_id)',
                'CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions(status)'
            ]
            
            for index_sql in indices:
                conn.execute(index_sql)
            
            conn.commit()
    
    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def create_experiment(self, name: str, description: str = None, 
                         tags: List[str] = None, parameters: Dict[str, Any] = None,
                         model_type: str = None, dataset_info: Dict[str, Any] = None) -> str:
        """Create a new experiment"""
        experiment_id = str(uuid.uuid4())
        tags = tags or []
        parameters = parameters or {}
        dataset_info = dataset_info or {}
        
        # Gather environment information
        environment_info = {
            "python_version": "3.8+",
            "platform": "linux",
            "timestamp": datetime.now().isoformat(),
            "working_directory": os.getcwd()
        }
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description or "",
            tags=tags,
            parameters=parameters,
            model_type=model_type or "unknown",
            dataset_info=dataset_info,
            environment_info=environment_info,
            created_at=datetime.now().isoformat(),
            created_by="cyber_llm_user"
        )
        
        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT INTO experiments 
                (experiment_id, name, description, tags, parameters, model_type, 
                 dataset_info, environment_info, created_at, created_by, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.experiment_id,
                config.name,
                config.description,
                json.dumps(config.tags),
                json.dumps(config.parameters),
                config.model_type,
                json.dumps(config.dataset_info),
                json.dumps(config.environment_info),
                config.created_at,
                config.created_by,
                config.created_at
            ))
            conn.commit()
        
        self.logger.info(f"Created experiment: {experiment_id} - {name}")
        return experiment_id
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float],
                   validation_metrics: Dict[str, float] = None,
                   custom_metrics: Dict[str, Any] = None,
                   step: int = None) -> None:
        """Log metrics for an experiment"""
        if step is None:
            # Auto-increment step
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    'SELECT MAX(step) FROM experiment_metrics WHERE experiment_id = ?',
                    (experiment_id,)
                )
                max_step = cursor.fetchone()[0]
                step = (max_step or 0) + 1
        
        validation_metrics = validation_metrics or {}
        custom_metrics = custom_metrics or {}
        
        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT INTO experiment_metrics 
                (experiment_id, step, timestamp, metrics, validation_metrics, custom_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                step,
                datetime.now().isoformat(),
                json.dumps(metrics),
                json.dumps(validation_metrics),
                json.dumps(custom_metrics)
            ))
            conn.commit()
    
    def log_artifact(self, experiment_id: str, name: str, path: str, 
                    artifact_type: str = "file", metadata: Dict[str, Any] = None) -> str:
        """Log an artifact for an experiment"""
        artifact_id = str(uuid.uuid4())
        metadata = metadata or {}
        
        # Calculate file checksum and size if path exists
        checksum = ""
        size_bytes = 0
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = f.read()
                checksum = hashlib.sha256(content).hexdigest()
                size_bytes = len(content)
        
        artifact = ExperimentArtifact(
            artifact_id=artifact_id,
            experiment_id=experiment_id,
            name=name,
            type=artifact_type,
            path=path,
            metadata=metadata,
            checksum=checksum,
            size_bytes=size_bytes,
            created_at=datetime.now().isoformat()
        )
        
        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT INTO experiment_artifacts 
                (artifact_id, experiment_id, name, type, path, metadata, checksum, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                artifact.artifact_id,
                artifact.experiment_id,
                artifact.name,
                artifact.type,
                artifact.path,
                json.dumps(artifact.metadata),
                artifact.checksum,
                artifact.size_bytes,
                artifact.created_at
            ))
            conn.commit()
        
        return artifact_id
    
    def log_model_version(self, experiment_id: str, model_name: str, 
                         model_path: str, performance_metrics: Dict[str, float],
                         version: str = None, metadata: Dict[str, Any] = None) -> str:
        """Log a model version"""
        version_id = str(uuid.uuid4())
        version = version or "v1.0"
        metadata = metadata or {}
        
        model_version = ModelVersion(
            version_id=version_id,
            experiment_id=experiment_id,
            model_name=model_name,
            version=version,
            status="training",
            performance_metrics=performance_metrics,
            model_path=model_path,
            metadata=metadata,
            created_at=datetime.now().isoformat()
        )
        
        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT INTO model_versions 
                (version_id, experiment_id, model_name, version, status, 
                 performance_metrics, model_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_version.version_id,
                model_version.experiment_id,
                model_version.model_name,
                model_version.version,
                model_version.status,
                json.dumps(model_version.performance_metrics),
                model_version.model_path,
                json.dumps(model_version.metadata),
                model_version.created_at
            ))
            conn.commit()
        
        return version_id
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details"""
        with self.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            experiment = dict(zip(columns, row))
            
            # Parse JSON fields
            experiment['tags'] = json.loads(experiment.get('tags', '[]'))
            experiment['parameters'] = json.loads(experiment.get('parameters', '{}'))
            experiment['dataset_info'] = json.loads(experiment.get('dataset_info', '{}'))
            experiment['environment_info'] = json.loads(experiment.get('environment_info', '{}'))
            
            return experiment
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all metrics for an experiment"""
        with self.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM experiment_metrics 
                WHERE experiment_id = ? 
                ORDER BY step
            ''', (experiment_id,))
            
            metrics = []
            for row in cursor.fetchall():
                columns = [desc[0] for desc in cursor.description]
                metric = dict(zip(columns, row))
                
                # Parse JSON fields
                metric['metrics'] = json.loads(metric.get('metrics', '{}'))
                metric['validation_metrics'] = json.loads(metric.get('validation_metrics', '{}'))
                metric['custom_metrics'] = json.loads(metric.get('custom_metrics', '{}'))
                
                metrics.append(metric)
            
            return metrics
    
    def get_experiment_artifacts(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts for an experiment"""
        with self.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM experiment_artifacts 
                WHERE experiment_id = ? 
                ORDER BY created_at
            ''', (experiment_id,))
            
            artifacts = []
            for row in cursor.fetchall():
                columns = [desc[0] for desc in cursor.description]
                artifact = dict(zip(columns, row))
                artifact['metadata'] = json.loads(artifact.get('metadata', '{}'))
                artifacts.append(artifact)
            
            return artifacts
    
    def list_experiments(self, tags: List[str] = None, model_type: str = None,
                        limit: int = None) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        query = 'SELECT * FROM experiments WHERE 1=1'
        params = []
        
        if tags:
            # Simple tag filtering (could be improved with proper JSON querying)
            for tag in tags:
                query += ' AND tags LIKE ?'
                params.append(f'%"{tag}"%')
        
        if model_type:
            query += ' AND model_type = ?'
            params.append(model_type)
        
        query += ' ORDER BY created_at DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        with self.get_db_connection() as conn:
            cursor = conn.execute(query, params)
            
            experiments = []
            for row in cursor.fetchall():
                columns = [desc[0] for desc in cursor.description]
                experiment = dict(zip(columns, row))
                
                # Parse JSON fields
                experiment['tags'] = json.loads(experiment.get('tags', '[]'))
                experiment['parameters'] = json.loads(experiment.get('parameters', '{}'))
                experiment['dataset_info'] = json.loads(experiment.get('dataset_info', '{}'))
                experiment['environment_info'] = json.loads(experiment.get('environment_info', '{}'))
                
                experiments.append(experiment)
            
            return experiments
    
    def compare_experiments(self, experiment_ids: List[str], 
                          comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """Compare multiple experiments"""
        comparison_id = str(uuid.uuid4())
        comparison_metrics = comparison_metrics or ['accuracy', 'loss', 'f1_score']
        
        comparison_data = {
            "comparison_id": comparison_id,
            "experiment_ids": experiment_ids,
            "experiments": {},
            "metric_comparison": {},
            "summary": {}
        }
        
        # Get experiment details
        for exp_id in experiment_ids:
            experiment = self.get_experiment(exp_id)
            if experiment:
                metrics = self.get_experiment_metrics(exp_id)
                comparison_data["experiments"][exp_id] = {
                    "name": experiment["name"],
                    "parameters": experiment["parameters"],
                    "metrics": metrics
                }
        
        # Compare metrics
        for metric_name in comparison_metrics:
            metric_values = {}
            for exp_id in experiment_ids:
                if exp_id in comparison_data["experiments"]:
                    exp_metrics = comparison_data["experiments"][exp_id]["metrics"]
                    if exp_metrics:
                        # Get the latest metric value
                        latest_metric = exp_metrics[-1]
                        if metric_name in latest_metric.get("metrics", {}):
                            metric_values[exp_id] = latest_metric["metrics"][metric_name]
                        elif metric_name in latest_metric.get("validation_metrics", {}):
                            metric_values[exp_id] = latest_metric["validation_metrics"][metric_name]
            
            if metric_values:
                comparison_data["metric_comparison"][metric_name] = {
                    "values": metric_values,
                    "best_experiment": max(metric_values, key=metric_values.get) if metric_name != 'loss' else min(metric_values, key=metric_values.get),
                    "worst_experiment": min(metric_values, key=metric_values.get) if metric_name != 'loss' else max(metric_values, key=metric_values.get),
                    "range": max(metric_values.values()) - min(metric_values.values()) if metric_values else 0
                }
        
        # Generate summary
        if comparison_data["metric_comparison"]:
            best_experiments = {}
            for metric_name, metric_data in comparison_data["metric_comparison"].items():
                best_exp = metric_data["best_experiment"]
                if best_exp not in best_experiments:
                    best_experiments[best_exp] = 0
                best_experiments[best_exp] += 1
            
            if best_experiments:
                overall_best = max(best_experiments, key=best_experiments.get)
                comparison_data["summary"]["overall_best_experiment"] = overall_best
                comparison_data["summary"]["best_experiment_name"] = comparison_data["experiments"][overall_best]["name"]
        
        # Store comparison in database
        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT INTO experiment_comparisons 
                (comparison_id, name, experiment_ids, comparison_metrics, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                comparison_id,
                f"Comparison of {len(experiment_ids)} experiments",
                json.dumps(experiment_ids),
                json.dumps(comparison_data),
                datetime.now().isoformat()
            ))
            conn.commit()
        
        return comparison_data
    
    def get_model_leaderboard(self, model_type: str = None, 
                             metric_name: str = "accuracy", 
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Get model leaderboard based on performance metrics"""
        query = '''
            SELECT e.experiment_id, e.name, e.model_type, e.created_at,
                   mv.version_id, mv.model_name, mv.version, mv.performance_metrics
            FROM experiments e
            JOIN model_versions mv ON e.experiment_id = mv.experiment_id
            WHERE mv.status != 'archived'
        '''
        params = []
        
        if model_type:
            query += ' AND e.model_type = ?'
            params.append(model_type)
        
        query += ' ORDER BY e.created_at DESC'
        
        with self.get_db_connection() as conn:
            cursor = conn.execute(query, params)
            
            models = []
            for row in cursor.fetchall():
                columns = [desc[0] for desc in cursor.description]
                model = dict(zip(columns, row))
                
                # Parse performance metrics
                performance_metrics = json.loads(model.get('performance_metrics', '{}'))
                model['performance_metrics'] = performance_metrics
                
                # Extract target metric for sorting
                model['target_metric_value'] = performance_metrics.get(metric_name, 0)
                
                models.append(model)
            
            # Sort by target metric (descending for most metrics, ascending for loss)
            reverse_sort = metric_name.lower() != 'loss'
            models.sort(key=lambda x: x['target_metric_value'], reverse=reverse_sort)
            
            return models[:limit] if limit else models
    
    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """Update experiment status"""
        with self.get_db_connection() as conn:
            conn.execute('''
                UPDATE experiments 
                SET status = ?, updated_at = ?
                WHERE experiment_id = ?
            ''', (status, datetime.now().isoformat(), experiment_id))
            conn.commit()
    
    def archive_experiment(self, experiment_id: str) -> None:
        """Archive an experiment"""
        self.update_experiment_status(experiment_id, 'archived')
        
        # Archive associated model versions
        with self.get_db_connection() as conn:
            conn.execute('''
                UPDATE model_versions 
                SET status = 'archived'
                WHERE experiment_id = ?
            ''', (experiment_id,))
            conn.commit()
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get overall experiment tracking statistics"""
        with self.get_db_connection() as conn:
            # Total experiments
            cursor = conn.execute('SELECT COUNT(*) FROM experiments')
            total_experiments = cursor.fetchone()[0]
            
            # Experiments by status
            cursor = conn.execute('''
                SELECT status, COUNT(*) 
                FROM experiments 
                GROUP BY status
            ''')
            status_counts = dict(cursor.fetchall())
            
            # Experiments by model type
            cursor = conn.execute('''
                SELECT model_type, COUNT(*) 
                FROM experiments 
                GROUP BY model_type
            ''')
            model_type_counts = dict(cursor.fetchall())
            
            # Total artifacts
            cursor = conn.execute('SELECT COUNT(*) FROM experiment_artifacts')
            total_artifacts = cursor.fetchone()[0]
            
            # Total model versions
            cursor = conn.execute('SELECT COUNT(*) FROM model_versions')
            total_model_versions = cursor.fetchone()[0]
            
            # Recent activity
            cursor = conn.execute('''
                SELECT COUNT(*) FROM experiments 
                WHERE created_at >= datetime('now', '-7 days')
            ''')
            experiments_last_week = cursor.fetchone()[0]
            
            return {
                "total_experiments": total_experiments,
                "status_distribution": status_counts,
                "model_type_distribution": model_type_counts,
                "total_artifacts": total_artifacts,
                "total_model_versions": total_model_versions,
                "experiments_last_week": experiments_last_week,
                "generated_at": datetime.now().isoformat()
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    print("🧪 MLOps Experiment Tracker Testing:")
    print("=" * 50)
    
    # Create sample experiments
    exp1_id = tracker.create_experiment(
        name="Cybersecurity Threat Detection - CNN",
        description="Convolutional Neural Network for malware detection",
        tags=["malware", "cnn", "classification"],
        parameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 50},
        model_type="cnn",
        dataset_info={"name": "malware_samples", "size": 10000, "features": 128}
    )
    
    exp2_id = tracker.create_experiment(
        name="Network Anomaly Detection - LSTM",
        description="LSTM network for network traffic anomaly detection",
        tags=["anomaly", "lstm", "network"],
        parameters={"learning_rate": 0.0001, "batch_size": 64, "epochs": 30},
        model_type="lstm",
        dataset_info={"name": "network_traffic", "size": 50000, "features": 64}
    )
    
    # Log metrics for experiments
    print(f"\n📊 Logging metrics for experiments...")
    for step in range(1, 6):
        # CNN experiment metrics
        tracker.log_metrics(exp1_id, {
            "accuracy": 0.7 + step * 0.05,
            "loss": 0.5 - step * 0.05,
            "precision": 0.75 + step * 0.03,
            "recall": 0.72 + step * 0.04
        }, validation_metrics={
            "val_accuracy": 0.68 + step * 0.04,
            "val_loss": 0.52 - step * 0.04
        }, step=step)
        
        # LSTM experiment metrics
        tracker.log_metrics(exp2_id, {
            "accuracy": 0.65 + step * 0.06,
            "loss": 0.6 - step * 0.06,
            "f1_score": 0.7 + step * 0.05
        }, validation_metrics={
            "val_accuracy": 0.62 + step * 0.05,
            "val_loss": 0.65 - step * 0.05
        }, step=step)
    
    # Log model versions
    print("🤖 Logging model versions...")
    model1_id = tracker.log_model_version(
        exp1_id, 
        "ThreatDetectionCNN", 
        "/models/threat_cnn_v1.pkl",
        {"accuracy": 0.95, "precision": 0.93, "recall": 0.92},
        version="v1.0"
    )
    
    model2_id = tracker.log_model_version(
        exp2_id,
        "AnomalyDetectionLSTM",
        "/models/anomaly_lstm_v1.pkl", 
        {"accuracy": 0.91, "f1_score": 0.89},
        version="v1.0"
    )
    
    # Log artifacts
    print("📁 Logging artifacts...")
    tracker.log_artifact(exp1_id, "training_plot.png", "/artifacts/training_plot.png", "plot")
    tracker.log_artifact(exp2_id, "confusion_matrix.png", "/artifacts/confusion_matrix.png", "plot")
    
    # List experiments
    print("\n📋 Listing experiments:")
    experiments = tracker.list_experiments(limit=5)
    for exp in experiments:
        print(f"  - {exp['name']} ({exp['model_type']}) - {len(exp['tags'])} tags")
    
    # Compare experiments
    print("\n⚖️ Comparing experiments:")
    comparison = tracker.compare_experiments([exp1_id, exp2_id], ["accuracy", "loss"])
    if comparison["summary"]:
        best_exp_name = comparison["summary"]["best_experiment_name"]
        print(f"  Best overall experiment: {best_exp_name}")
    
    # Get leaderboard
    print("\n🏆 Model Leaderboard (by accuracy):")
    leaderboard = tracker.get_model_leaderboard(metric_name="accuracy", limit=5)
    for i, model in enumerate(leaderboard, 1):
        accuracy = model.get('target_metric_value', 0)
        print(f"  {i}. {model['model_name']} - Accuracy: {accuracy:.3f}")
    
    # Get statistics
    print("\n📈 Experiment Statistics:")
    stats = tracker.get_experiment_stats()
    print(f"  Total Experiments: {stats['total_experiments']}")
    print(f"  Total Model Versions: {stats['total_model_versions']}")
    print(f"  Total Artifacts: {stats['total_artifacts']}")
    print(f"  Experiments Last Week: {stats['experiments_last_week']}")
    
    print("\n✅ MLOps Experiment Tracker implemented and tested")
