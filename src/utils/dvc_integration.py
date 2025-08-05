"""
DVC (Data Version Control) Integration for Cyber-LLM
Provides data versioning, experiment tracking, and pipeline management
"""

import os
import json
import yaml
import subprocess
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import tempfile

from .logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory, retry_with_backoff

@dataclass
class DVCMetrics:
    """DVC metrics for model evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    stealth_score: Optional[float] = None
    chain_success_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    safety_compliance: Optional[float] = None

@dataclass
class DVCParameters:
    """DVC parameters for experiments"""
    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str
    dataset_version: str
    adapter_rank: Optional[int] = None
    dropout_rate: Optional[float] = None
    warmup_steps: Optional[int] = None

@dataclass
class DVCExperiment:
    """DVC experiment information"""
    id: str
    name: str
    timestamp: datetime
    parameters: DVCParameters
    metrics: DVCMetrics
    git_commit: str
    status: str
    duration: Optional[float] = None

class DVCManager:
    """DVC integration manager for data versioning and experiment tracking"""
    
    def __init__(self, 
                 repo_path: str = ".",
                 remote_name: str = "origin",
                 logger: Optional[CyberLLMLogger] = None):
        
        self.repo_path = Path(repo_path).resolve()
        self.remote_name = remote_name
        self.logger = logger or CyberLLMLogger(name="dvc_manager")
        
        # DVC configuration paths
        self.dvc_dir = self.repo_path / ".dvc"
        self.params_file = self.repo_path / "params.yaml"
        self.metrics_file = self.repo_path / "metrics.yaml"
        self.dvcfile = self.repo_path / "dvc.yaml"
        
        # Initialize DVC if not already initialized
        self._ensure_dvc_initialized()
    
    def _ensure_dvc_initialized(self):
        """Ensure DVC is initialized in the repository"""
        if not self.dvc_dir.exists():
            self.logger.info("Initializing DVC repository")
            self._run_dvc_command(["init"])
            self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default DVC configuration"""
        
        # Create default params.yaml
        default_params = {
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 8,
                "epochs": 3,
                "model_name": "microsoft/DialoGPT-medium",
                "dataset_version": "v1.0",
                "adapter_rank": 16,
                "dropout_rate": 0.1,
                "warmup_steps": 500
            },
            "data": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "max_length": 512,
                "min_samples_per_class": 100
            },
            "evaluation": {
                "batch_size": 16,
                "max_samples": 1000,
                "metrics_threshold": {
                    "accuracy": 0.85,
                    "stealth_score": 0.7,
                    "safety_compliance": 0.95
                }
            }
        }
        
        if not self.params_file.exists():
            with open(self.params_file, 'w') as f:
                yaml.dump(default_params, f, default_flow_style=False)
            self.logger.info("Created default params.yaml")
        
        # Create default dvc.yaml pipeline
        default_pipeline = {
            "stages": {
                "data_preparation": {
                    "cmd": "python src/training/data_preprocessing.py",
                    "deps": [
                        "src/training/data_preprocessing.py",
                        "data/raw/"
                    ],
                    "outs": [
                        "data/processed/train.jsonl",
                        "data/processed/val.jsonl",
                        "data/processed/test.jsonl"
                    ],
                    "params": [
                        "data.train_split",
                        "data.val_split",
                        "data.test_split"
                    ]
                },
                "training": {
                    "cmd": "python src/training/train_model.py",
                    "deps": [
                        "src/training/train_model.py",
                        "data/processed/train.jsonl",
                        "data/processed/val.jsonl"
                    ],
                    "outs": [
                        "models/cyber_llm_adapter/"
                    ],
                    "params": [
                        "training"
                    ],
                    "metrics": [
                        "metrics/training.yaml"
                    ]
                },
                "evaluation": {
                    "cmd": "python src/evaluation/evaluate_model.py",
                    "deps": [
                        "src/evaluation/evaluate_model.py",
                        "models/cyber_llm_adapter/",
                        "data/processed/test.jsonl"
                    ],
                    "metrics": [
                        "metrics/evaluation.yaml"
                    ],
                    "params": [
                        "evaluation"
                    ]
                }
            }
        }
        
        if not self.dvcfile.exists():
            with open(self.dvcfile, 'w') as f:
                yaml.dump(default_pipeline, f, default_flow_style=False)
            self.logger.info("Created default dvc.yaml pipeline")
    
    def _run_dvc_command(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run DVC command and return result"""
        cmd = ["dvc"] + args
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.stdout:
                self.logger.debug(f"DVC command output: {result.stdout.strip()}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            error_msg = f"DVC command failed: {' '.join(cmd)}\nError: {e.stderr}"
            self.logger.error(error_msg)
            raise CyberLLMError(error_msg, ErrorCategory.SYSTEM)
        except FileNotFoundError:
            raise CyberLLMError("DVC not found. Please install DVC: pip install dvc", ErrorCategory.SYSTEM)
    
    @retry_with_backoff(max_retries=3)
    async def add_data(self, data_path: str, remote: bool = True) -> bool:
        """Add data to DVC tracking"""
        try:
            # Add to DVC
            self._run_dvc_command(["add", data_path])
            
            # Add .dvc file to git
            dvc_file = f"{data_path}.dvc"
            if os.path.exists(dvc_file):
                subprocess.run(["git", "add", dvc_file], 
                             cwd=self.repo_path, check=True)
            
            # Push to remote if requested
            if remote:
                await self.push_data()
            
            self.logger.info(f"Added data to DVC: {data_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add data to DVC: {data_path}", error=str(e))
            return False
    
    async def push_data(self) -> bool:
        """Push data to DVC remote"""
        try:
            # Run push in background
            process = await asyncio.create_subprocess_exec(
                "dvc", "push",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("Successfully pushed data to DVC remote")
                return True
            else:
                self.logger.error("Failed to push data to DVC remote", 
                                error=stderr.decode())
                return False
                
        except Exception as e:
            self.logger.error("DVC push failed", error=str(e))
            return False
    
    async def pull_data(self) -> bool:
        """Pull data from DVC remote"""
        try:
            process = await asyncio.create_subprocess_exec(
                "dvc", "pull",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("Successfully pulled data from DVC remote")
                return True
            else:
                self.logger.error("Failed to pull data from DVC remote", 
                                error=stderr.decode())
                return False
                
        except Exception as e:
            self.logger.error("DVC pull failed", error=str(e))
            return False
    
    def create_experiment(self, 
                         name: str,
                         parameters: DVCParameters,
                         description: str = "") -> str:
        """Create a new DVC experiment"""
        
        # Generate experiment ID
        exp_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # Update params.yaml with experiment parameters
        self.update_parameters(asdict(parameters))
        
        # Create experiment branch
        try:
            self._run_dvc_command(["exp", "run", "--name", name, "--set-param", f"experiment.id={exp_id}"])
            
            self.logger.info(f"Created DVC experiment: {name} (ID: {exp_id})")
            return exp_id
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {name}", error=str(e))
            raise
    
    def update_parameters(self, params: Dict[str, Any]):
        """Update parameters file"""
        try:
            # Load existing params
            existing_params = {}
            if self.params_file.exists():
                with open(self.params_file, 'r') as f:
                    existing_params = yaml.safe_load(f) or {}
            
            # Deep merge new parameters
            def deep_merge(base: dict, update: dict) -> dict:
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
                return base
            
            merged_params = deep_merge(existing_params, params)
            
            # Write updated params
            with open(self.params_file, 'w') as f:
                yaml.dump(merged_params, f, default_flow_style=False)
            
            self.logger.debug("Updated parameters file")
            
        except Exception as e:
            self.logger.error("Failed to update parameters", error=str(e))
            raise
    
    def log_metrics(self, metrics: DVCMetrics, stage: str = "evaluation"):
        """Log metrics to DVC"""
        try:
            metrics_dir = self.repo_path / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            metrics_file = metrics_dir / f"{stage}.yaml"
            
            # Convert metrics to dict
            metrics_dict = asdict(metrics)
            
            # Write metrics
            with open(metrics_file, 'w') as f:
                yaml.dump(metrics_dict, f, default_flow_style=False)
            
            self.logger.info(f"Logged metrics for stage: {stage}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics for stage: {stage}", error=str(e))
            raise
    
    def get_experiments(self) -> List[DVCExperiment]:
        """Get list of DVC experiments"""
        try:
            result = self._run_dvc_command(["exp", "show", "--json"])
            
            experiments = []
            if result.stdout:
                exp_data = json.loads(result.stdout)
                
                for exp_info in exp_data:
                    # Parse experiment data
                    exp = DVCExperiment(
                        id=exp_info.get("id", ""),
                        name=exp_info.get("name", ""),
                        timestamp=datetime.fromisoformat(exp_info.get("timestamp", datetime.now().isoformat())),
                        parameters=DVCParameters(**exp_info.get("params", {})),
                        metrics=DVCMetrics(**exp_info.get("metrics", {})),
                        git_commit=exp_info.get("rev", ""),
                        status=exp_info.get("status", "unknown"),
                        duration=exp_info.get("duration")
                    )
                    experiments.append(exp)
            
            return experiments
            
        except Exception as e:
            self.logger.error("Failed to get experiments", error=str(e))
            return []
    
    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        try:
            cmd = ["exp", "diff"] + exp_ids
            result = self._run_dvc_command(cmd)
            
            # Parse diff output (simplified)
            comparison = {
                "experiments": exp_ids,
                "timestamp": datetime.now().isoformat(),
                "raw_output": result.stdout
            }
            
            self.logger.info(f"Compared experiments: {', '.join(exp_ids)}")
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare experiments: {exp_ids}", error=str(e))
            return {}
    
    async def run_pipeline(self, 
                          stages: Optional[List[str]] = None,
                          force: bool = False) -> bool:
        """Run DVC pipeline"""
        try:
            cmd = ["repro"]
            
            if force:
                cmd.append("--force")
            
            if stages:
                cmd.extend(stages)
            
            self.logger.info(f"Starting DVC pipeline: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                "dvc", *cmd,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("DVC pipeline completed successfully")
                if stdout:
                    self.logger.debug(f"Pipeline output: {stdout.decode()}")
                return True
            else:
                self.logger.error("DVC pipeline failed", error=stderr.decode())
                return False
                
        except Exception as e:
            self.logger.error("Failed to run DVC pipeline", error=str(e))
            return False
    
    def setup_remote_storage(self, 
                           storage_type: str,
                           config: Dict[str, str]) -> bool:
        """Setup DVC remote storage"""
        try:
            remote_name = config.get("name", "default")
            
            if storage_type == "s3":
                url = f"s3://{config['bucket']}/{config.get('prefix', '')}"
                self._run_dvc_command(["remote", "add", "-d", remote_name, url])
                
                # Set AWS credentials if provided
                if "access_key_id" in config:
                    self._run_dvc_command(["remote", "modify", remote_name, 
                                          "access_key_id", config["access_key_id"]])
                if "secret_access_key" in config:
                    self._run_dvc_command(["remote", "modify", remote_name, 
                                          "secret_access_key", config["secret_access_key"]])
                if "region" in config:
                    self._run_dvc_command(["remote", "modify", remote_name, 
                                          "region", config["region"]])
            
            elif storage_type == "azure":
                url = f"azure://{config['container']}/{config.get('prefix', '')}"
                self._run_dvc_command(["remote", "add", "-d", remote_name, url])
                
                if "account_name" in config:
                    self._run_dvc_command(["remote", "modify", remote_name, 
                                          "account_name", config["account_name"]])
            
            elif storage_type == "gcs":
                url = f"gs://{config['bucket']}/{config.get('prefix', '')}"
                self._run_dvc_command(["remote", "add", "-d", remote_name, url])
            
            elif storage_type == "ssh":
                url = f"ssh://{config['host']}{config['path']}"
                self._run_dvc_command(["remote", "add", "-d", remote_name, url])
                
                if "user" in config:
                    self._run_dvc_command(["remote", "modify", remote_name, 
                                          "user", config["user"]])
            
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
            
            self.logger.info(f"Setup DVC remote storage: {storage_type} ({remote_name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup remote storage: {storage_type}", error=str(e))
            return False
    
    def get_data_info(self, data_path: str) -> Dict[str, Any]:
        """Get information about tracked data"""
        try:
            dvc_file = f"{data_path}.dvc"
            
            if not os.path.exists(dvc_file):
                return {"tracked": False}
            
            # Parse .dvc file
            with open(dvc_file, 'r') as f:
                dvc_data = yaml.safe_load(f)
            
            # Get file info
            file_info = {
                "tracked": True,
                "path": data_path,
                "dvc_file": dvc_file,
                "md5": dvc_data.get("outs", [{}])[0].get("md5", ""),
                "size": os.path.getsize(data_path) if os.path.exists(data_path) else 0,
                "remote_available": self._check_remote_availability(data_path)
            }
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Failed to get data info: {data_path}", error=str(e))
            return {"tracked": False, "error": str(e)}
    
    def _check_remote_availability(self, data_path: str) -> bool:
        """Check if data is available in remote storage"""
        try:
            result = self._run_dvc_command(["status", data_path], check=False)
            return "not in cache" not in result.stdout.lower()
        except:
            return False

# Convenience functions
def init_dvc_project(repo_path: str = ".") -> DVCManager:
    """Initialize DVC project"""
    return DVCManager(repo_path=repo_path)

async def track_dataset(dataset_path: str, 
                       dvc_manager: Optional[DVCManager] = None) -> bool:
    """Track dataset with DVC"""
    manager = dvc_manager or DVCManager()
    return await manager.add_data(dataset_path)

def create_training_experiment(name: str, 
                             learning_rate: float = 2e-5,
                             batch_size: int = 8,
                             epochs: int = 3,
                             dvc_manager: Optional[DVCManager] = None) -> str:
    """Create training experiment"""
    manager = dvc_manager or DVCManager()
    
    params = DVCParameters(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        model_name="microsoft/DialoGPT-medium",
        dataset_version="v1.0"
    )
    
    return manager.create_experiment(name, params)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize DVC manager
        dvc = DVCManager()
        
        # Track a dataset
        success = await dvc.add_data("data/raw/cyber_dataset.jsonl")
        print(f"Dataset tracking: {'success' if success else 'failed'}")
        
        # Create experiment
        params = DVCParameters(
            learning_rate=1e-4,
            batch_size=16,
            epochs=5,
            model_name="microsoft/DialoGPT-medium",
            dataset_version="v1.0",
            adapter_rank=32
        )
        
        exp_id = dvc.create_experiment("experiment_001", params)
        print(f"Created experiment: {exp_id}")
        
        # Log metrics
        metrics = DVCMetrics(
            accuracy=0.87,
            precision=0.85,
            recall=0.89,
            f1_score=0.87,
            loss=0.23,
            stealth_score=0.73,
            safety_compliance=0.96
        )
        
        dvc.log_metrics(metrics)
        
        # Run pipeline
        success = await dvc.run_pipeline()
        print(f"Pipeline execution: {'success' if success else 'failed'}")
    
    asyncio.run(main())
