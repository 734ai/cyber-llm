"""
Model Configuration for Cyber-LLM Training
Defines model architectures, hyperparameters, and training configurations.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Base model configuration."""
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal_lm"  # causal_lm, seq2seq
    cache_dir: str = "./models/cache"
    trust_remote_code: bool = False
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    
@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    r: int = 16  # Rank of adaptation
    lora_alpha: int = 32  # LoRA scaling parameter
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"
    
@dataclass 
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Basic training parameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Advanced training parameters
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Optimization
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["wandb", "mlflow"])
    
@dataclass
class AdapterConfig:
    """Configuration for domain-specific adapters."""
    adapter_name: str
    description: str
    target_domain: str  # recon, c2, postexploit, explainability, safety
    lora_config: LoRAConfig
    training_config: TrainingConfig
    special_tokens: Dict[str, str] = field(default_factory=dict)
    
    # Domain-specific configurations
    domain_weight: float = 1.0  # Weight for domain-specific loss
    curriculum_stages: List[str] = field(default_factory=list)
    
@dataclass
class CyberLLMConfig:
    """Complete configuration for Cyber-LLM training."""
    # Model configuration
    base_model: ModelConfig
    
    # Adapter configurations
    adapters: Dict[str, AdapterConfig]
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_length": 2048,
        "train_file": "data/processed/train.json",
        "val_file": "data/processed/validation.json",
        "test_file": "data/processed/test.json"
    })
    
    # Output configuration
    output_config: Dict[str, Any] = field(default_factory=lambda: {
        "output_dir": "./models/cyber-llm",
        "adapter_dir": "./adapters",
        "logs_dir": "./logs",
        "checkpoint_dir": "./checkpoints"
    })
    
    # Experiment tracking
    experiment_config: Dict[str, Any] = field(default_factory=lambda: {
        "project_name": "cyber-llm",
        "experiment_name": "base-training",
        "tags": ["cybersecurity", "red-team", "llm"],
        "notes": "Cyber-LLM training experiment"
    })

# Predefined configurations for different adapters
def get_recon_adapter_config() -> AdapterConfig:
    """Configuration for reconnaissance adapter."""
    return AdapterConfig(
        adapter_name="ReconOps",
        description="Reconnaissance and information gathering operations",
        target_domain="recon",
        lora_config=LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        ),
        training_config=TrainingConfig(
            learning_rate=3e-5,
            batch_size=2,
            num_epochs=5,
            warmup_ratio=0.15
        ),
        special_tokens={
            "domain_token": "<|RECON|>",
            "task_tokens": ["<|NMAP|>", "<|OSINT|>", "<|PASSIVE|>"]
        },
        curriculum_stages=["basic_scanning", "advanced_techniques", "opsec_aware"]
    )

def get_c2_adapter_config() -> AdapterConfig:
    """Configuration for C2 adapter."""
    return AdapterConfig(
        adapter_name="C2Ops",
        description="Command and Control operations and management",
        target_domain="c2",
        lora_config=LoRAConfig(
            r=20,
            lora_alpha=40,
            lora_dropout=0.05
        ),
        training_config=TrainingConfig(
            learning_rate=2e-5,
            batch_size=3,
            num_epochs=4,
            warmup_ratio=0.1
        ),
        special_tokens={
            "domain_token": "<|C2|>",
            "task_tokens": ["<|EMPIRE|>", "<|COBALT|>", "<|PAYLOAD|>"]
        },
        curriculum_stages=["c2_basics", "payload_generation", "opsec_c2"]
    )

def get_postexploit_adapter_config() -> AdapterConfig:
    """Configuration for post-exploitation adapter."""
    return AdapterConfig(
        adapter_name="PostExploit",
        description="Post-exploitation and lateral movement operations",
        target_domain="postexploit",
        lora_config=LoRAConfig(
            r=18,
            lora_alpha=36,
            lora_dropout=0.1
        ),
        training_config=TrainingConfig(
            learning_rate=2.5e-5,
            batch_size=2,
            num_epochs=6,
            warmup_ratio=0.12
        ),
        special_tokens={
            "domain_token": "<|POSTEXPLOIT|>",
            "task_tokens": ["<|CREDS|>", "<|LATERAL|>", "<|PERSIST|>"]
        },
        curriculum_stages=["credential_access", "lateral_movement", "persistence"]
    )

def get_explainability_adapter_config() -> AdapterConfig:
    """Configuration for explainability adapter."""
    return AdapterConfig(
        adapter_name="Explainability",
        description="Decision explanation and rationale generation",
        target_domain="explainability",
        lora_config=LoRAConfig(
            r=12,
            lora_alpha=24,
            lora_dropout=0.15
        ),
        training_config=TrainingConfig(
            learning_rate=1.5e-5,
            batch_size=4,
            num_epochs=3,
            warmup_ratio=0.08
        ),
        special_tokens={
            "domain_token": "<|EXPLAIN|>",
            "task_tokens": ["<|REASONING|>", "<|RISK|>", "<|MITIGATION|>"]
        },
        curriculum_stages=["basic_explanation", "risk_assessment", "mitigation_advice"]
    )

def get_safety_adapter_config() -> AdapterConfig:
    """Configuration for safety adapter."""
    return AdapterConfig(
        adapter_name="Safety",
        description="OPSEC compliance and safety validation",
        target_domain="safety",
        lora_config=LoRAConfig(
            r=14,
            lora_alpha=28,
            lora_dropout=0.12
        ),
        training_config=TrainingConfig(
            learning_rate=1e-5,
            batch_size=6,
            num_epochs=4,
            warmup_ratio=0.2
        ),
        special_tokens={
            "domain_token": "<|SAFETY|>",
            "task_tokens": ["<|OPSEC|>", "<|VALIDATE|>", "<|APPROVE|>"]
        },
        curriculum_stages=["opsec_basics", "risk_detection", "safety_validation"]
    )

def create_default_config() -> CyberLLMConfig:
    """Create default Cyber-LLM configuration."""
    
    # Base model configuration
    base_model = ModelConfig(
        model_name="microsoft/DialoGPT-medium",
        model_type="causal_lm",
        torch_dtype="float16"
    )
    
    # Create adapter configurations
    adapters = {
        "recon": get_recon_adapter_config(),
        "c2": get_c2_adapter_config(),
        "postexploit": get_postexploit_adapter_config(),
        "explainability": get_explainability_adapter_config(),
        "safety": get_safety_adapter_config()
    }
    
    return CyberLLMConfig(
        base_model=base_model,
        adapters=adapters
    )

def save_config(config: CyberLLMConfig, config_path: Path):
    """Save configuration to JSON file."""
    config_dict = {}
    
    # Convert dataclasses to dictionaries
    config_dict["base_model"] = config.base_model.__dict__
    
    config_dict["adapters"] = {}
    for name, adapter_config in config.adapters.items():
        config_dict["adapters"][name] = {
            "adapter_name": adapter_config.adapter_name,
            "description": adapter_config.description,
            "target_domain": adapter_config.target_domain,
            "lora_config": adapter_config.lora_config.__dict__,
            "training_config": adapter_config.training_config.__dict__,
            "special_tokens": adapter_config.special_tokens,
            "domain_weight": adapter_config.domain_weight,
            "curriculum_stages": adapter_config.curriculum_stages
        }
    
    config_dict["data_config"] = config.data_config
    config_dict["output_config"] = config.output_config
    config_dict["experiment_config"] = config.experiment_config
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def load_config(config_path: Path) -> CyberLLMConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Reconstruct base model config
    base_model = ModelConfig(**config_dict["base_model"])
    
    # Reconstruct adapter configs
    adapters = {}
    for name, adapter_dict in config_dict["adapters"].items():
        lora_config = LoRAConfig(**adapter_dict["lora_config"])
        training_config = TrainingConfig(**adapter_dict["training_config"])
        
        adapter_config = AdapterConfig(
            adapter_name=adapter_dict["adapter_name"],
            description=adapter_dict["description"],
            target_domain=adapter_dict["target_domain"],
            lora_config=lora_config,
            training_config=training_config,
            special_tokens=adapter_dict["special_tokens"],
            domain_weight=adapter_dict["domain_weight"],
            curriculum_stages=adapter_dict["curriculum_stages"]
        )
        adapters[name] = adapter_config
    
    return CyberLLMConfig(
        base_model=base_model,
        adapters=adapters,
        data_config=config_dict["data_config"],
        output_config=config_dict["output_config"],
        experiment_config=config_dict["experiment_config"]
    )

# Example usage
if __name__ == "__main__":
    # Create and save default configuration
    config = create_default_config()
    save_config(config, Path("configs/model_config.json"))
    
    print("Default configuration saved to configs/model_config.json")
    
    # Print configuration summary
    print(f"Base model: {config.base_model.model_name}")
    print(f"Number of adapters: {len(config.adapters)}")
    for name, adapter in config.adapters.items():
        print(f"  - {name}: {adapter.description}")
