#!/bin/bash
# Cyber-LLM Development Runner Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Cyber-LLM Development Environment${NC}"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "Python version: ${GREEN}$python_version${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p data/raw data/processed data/embeddings
mkdir -p models/cache models/adapters models/checkpoints
mkdir -p logs/training logs/evaluation
mkdir -p monitoring/prometheus_data monitoring/grafana_data

# Generate initial configuration
echo -e "${YELLOW}Generating initial configuration...${NC}"
python configs/model_config.py

# Run basic system check
echo -e "${YELLOW}Running system checks...${NC}"
python src/deployment/cli/cyber_cli.py status

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Available commands:"
echo "  ./run.sh cli     - Start CLI interface"
echo "  ./run.sh test    - Run tests"
echo "  ./run.sh train   - Start training pipeline"
echo "  ./run.sh api     - Start API server"
echo ""

# Handle command line arguments
case "${1:-help}" in
    "cli")
        echo -e "${BLUE}Starting Cyber-LLM CLI...${NC}"
        python src/deployment/cli/cyber_cli.py "${@:2}"
        ;;
    "test")
        echo -e "${BLUE}Running tests...${NC}"
        pytest tests/ -v
        ;;
    "train")
        echo -e "${BLUE}Starting training pipeline...${NC}"
        echo "Training pipeline not yet implemented"
        ;;
    "api")
        echo -e "${BLUE}Starting API server...${NC}"
        echo "API server not yet implemented"
        ;;
    "help"|*)
        echo -e "${BLUE}Usage: ./run.sh [command]${NC}"
        echo "Commands:"
        echo "  cli    - Start CLI interface"
        echo "  test   - Run tests"
        echo "  train  - Start training pipeline"
        echo "  api    - Start API server"
        echo "  help   - Show this help message"
        ;;
esac
