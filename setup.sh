#!/bin/bash
# Cyber-LLM Project Setup Script
# Author: Muzan Sano
# Email: sanosensei36@gmail.com

set -e

echo "🚀 Setting up Cyber-LLM project..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "📋 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p models
chmod +x src/deployment/cli/cyber_cli.py

# Set up DVC (if available)
if command -v dvc &> /dev/null; then
    echo "📊 Initializing DVC..."
    dvc init --no-scm 2>/dev/null || echo "DVC already initialized"
fi

# Set up pre-commit hooks (if available)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install 2>/dev/null || echo "Pre-commit hooks setup skipped"
fi

# Download sample data (placeholder)
echo "📥 Setting up sample data..."
mkdir -p src/data/raw/samples
echo "Sample cybersecurity dataset placeholder" > src/data/raw/samples/sample.txt

# Create initial configuration files
echo "⚙️  Creating configuration files..."
cat > configs/training_config.yaml << 'EOF'
# Training Configuration for Cyber-LLM
model:
  base_model: "microsoft/Phi-3-mini-4k-instruct"
  max_length: 2048

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1

training:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3

mlops:
  use_wandb: false
  use_mlflow: false
  experiment_name: "cyber-llm-local"
EOF

# Run initial tests
echo "🧪 Running initial tests..."
python -c "
import sys
print('✅ Python import test passed')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} available')
    print(f'   CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('⚠️  PyTorch not available - install manually if needed')

try:
    import transformers
    print(f'✅ Transformers {transformers.__version__} available')
except ImportError:
    print('⚠️  Transformers not available - install manually if needed')
"

# Create sample workflow
echo "📋 Creating sample workflow files..."
mkdir -p src/orchestration/workflows
cat > src/orchestration/workflows/basic_red_team.yaml << 'EOF'
name: "Basic Red Team Assessment"
description: "Standard red team workflow"
phases:
  - name: "reconnaissance"
    agents: ["recon"]
    parallel: false
    safety_check: true
    human_approval: true
  - name: "initial_access"
    agents: ["c2"]
    parallel: false
    safety_check: true
    human_approval: true
    depends_on: ["reconnaissance"]
EOF

echo ""
echo "✅ Cyber-LLM setup completed successfully!"
echo ""
echo "📖 Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run CLI: python src/deployment/cli/cyber_cli.py --help"
echo "   3. Train adapters: python src/training/train.py --help"
echo "   4. Check README.md for detailed instructions"
echo ""
echo "🔐 For red team operations, ensure you have proper authorization!"
echo "📧 Questions? Contact: sanosensei36@gmail.com"
echo ""
