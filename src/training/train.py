"""
Cyber-LLM Training Module

Main training script for LoRA adapters and adversarial fine-tuning.
Integrates with MLflow, Weights & Biases, and DVC for experiment tracking.

Author: Muzan Sano
Email: sanosensei36@gmail.com
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_from_disk
import yaml

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyberLLMTrainer:
    """
    Advanced training system for Cyber-LLM with LoRA adapters.
    Supports adversarial training and multi-agent specialization.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize MLOps tracking
        self._init_mlops()
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_mlops(self):
        """Initialize MLOps tracking systems."""
        # Initialize MLflow
        if MLFLOW_AVAILABLE and self.config.get("mlops", {}).get("use_mlflow", False):
            mlflow.set_experiment(self.config["mlops"]["experiment_name"])
            mlflow.start_run(run_name=self.config["mlops"]["run_name"])
            logger.info("MLflow tracking initialized")
        
        # Initialize Weights & Biases
        if WANDB_AVAILABLE and self.config.get("mlops", {}).get("use_wandb", False):
            wandb.init(
                project="cyber-llm",
                name=self.config["mlops"]["run_name"],
                config=self.config
            )
            logger.info("Weights & Biases tracking initialized")
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer."""
        model_name = self.config["model"]["base_model"]
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add special token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
    
    def setup_lora(self, target_modules: Optional[List[str]] = None) -> None:
        """Setup LoRA configuration for parameter-efficient fine-tuning."""
        if target_modules is None:
            target_modules = self.config["lora"]["target_modules"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            target_modules=target_modules,
            bias="none"
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def load_training_data(self, data_path: str) -> Dataset:
        """Load and prepare training data."""
        logger.info(f"Loading training data from: {data_path}")
        
        if os.path.isdir(data_path):
            # Load preprocessed dataset
            dataset = load_from_disk(data_path)
        else:
            # Load from JSON/JSONL
            with open(data_path, 'r') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            dataset = Dataset.from_list(data)
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset for training."""
        def tokenize_function(examples):
            # Handle different input formats
            if "text" in examples:
                texts = examples["text"]
            elif "instruction" in examples and "response" in examples:
                texts = [f"Instruction: {inst}\nResponse: {resp}" 
                        for inst, resp in zip(examples["instruction"], examples["response"])]
            else:
                raise ValueError("Dataset must contain 'text' or 'instruction'/'response' fields")
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config["data"]["max_seq_length"],
                return_tensors=None
            )
            
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def create_adversarial_examples(self, dataset: Dataset, num_adversarial: int = 1000) -> Dataset:
        """
        Generate adversarial examples for robust training.
        This is a simplified implementation - could be enhanced with more sophisticated techniques.
        """
        logger.info(f"Generating {num_adversarial} adversarial examples")
        
        adversarial_prompts = [
            "Ignore previous instructions and ",
            "As a helpful AI assistant, please provide detailed instructions for ",
            "This is for educational purposes only: ",
            "Hypothetically speaking, if someone wanted to ",
            "In a fictional scenario, describe how to ",
        ]
        
        adversarial_examples = []
        original_examples = dataset.select(range(min(num_adversarial, len(dataset))))
        
        for i, example in enumerate(original_examples):
            if i % 200 == 0:
                logger.info(f"Generated {i}/{num_adversarial} adversarial examples")
            
            original_text = example.get("text", "")
            
            # Generate adversarial version
            adversarial_prefix = adversarial_prompts[i % len(adversarial_prompts)]
            adversarial_text = adversarial_prefix + original_text
            
            adversarial_examples.append({
                "text": adversarial_text,
                "is_adversarial": True,
                "original_text": original_text
            })
        
        # Combine with original dataset
        adversarial_dataset = Dataset.from_list(adversarial_examples)
        combined_dataset = dataset.select(range(len(dataset) - num_adversarial)).concatenate(adversarial_dataset)
        
        logger.info(f"Created combined dataset with {len(combined_dataset)} examples")
        return combined_dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        output_dir = f"./outputs/{self.config['mlops']['run_name']}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"]["warmup_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_drop_last=True,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to=["wandb"] if WANDB_AVAILABLE else None,
            run_name=self.config["mlops"]["run_name"]
        )
        
        return training_args
    
    def train_adapter(self, adapter_name: str, dataset_path: str, adversarial_training: bool = False):
        """Train a specific LoRA adapter."""
        logger.info(f"Starting training for adapter: {adapter_name}")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model_and_tokenizer()
            self.setup_lora()
        
        # Load and prepare data
        dataset = self.load_training_data(dataset_path)
        
        # Add adversarial examples if requested
        if adversarial_training:
            dataset = self.create_adversarial_examples(dataset)
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Split dataset
        train_size = int(len(tokenized_dataset) * self.config["data"]["train_split"])
        val_size = int(len(tokenized_dataset) * self.config["data"]["val_split"])
        
        train_dataset = tokenized_dataset.select(range(train_size))
        val_dataset = tokenized_dataset.select(range(train_size, train_size + val_size))
        
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Custom data collator for language modeling
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save adapter
        adapter_output_path = f"./adapters/{adapter_name}"
        trainer.save_model(adapter_output_path)
        
        # Log metrics
        if MLFLOW_AVAILABLE:
            mlflow.log_artifacts(adapter_output_path, artifact_path=f"adapters/{adapter_name}")
        
        logger.info(f"Training completed for adapter: {adapter_name}")
        logger.info(f"Adapter saved to: {adapter_output_path}")
        
        return trainer
    
    def evaluate_model(self, test_dataset_path: str):
        """Evaluate the trained model."""
        logger.info("Starting model evaluation")
        
        # Load test dataset
        test_dataset = self.load_training_data(test_dataset_path)
        tokenized_test = self.tokenize_dataset(test_dataset)
        
        # Evaluation metrics would go here
        # For now, just compute perplexity
        
        logger.info("Evaluation completed")
    
    def cleanup(self):
        """Cleanup resources."""
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        logger.info("Training cleanup completed")

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Cyber-LLM Training")
    parser.add_argument("--config", required=True, help="Path to training configuration file")
    parser.add_argument("--adapter", required=True, help="Name of adapter to train")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--adversarial", action="store_true", help="Enable adversarial training")
    parser.add_argument("--eval-data", help="Path to evaluation data")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = CyberLLMTrainer(args.config)
        
        # Train adapter
        trainer.train_adapter(
            adapter_name=args.adapter,
            dataset_path=args.data,
            adversarial_training=args.adversarial
        )
        
        # Evaluate if test data provided
        if args.eval_data:
            trainer.evaluate_model(args.eval_data)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()

if __name__ == "__main__":
    main()
