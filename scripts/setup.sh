#!/bin/bash
# PINN Seismic Inversion - Setup Script
# This script sets up the development environment

set -e

echo "================================================"
echo "  PINN Seismic Inversion - Setup Script"
echo "================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (detect CUDA)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Python dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data checkpoints logs experiments
echo "✓ Directories created"

# Setup frontend
echo ""
echo "Setting up frontend..."
cd frontend

if command -v npm &> /dev/null; then
    echo "Installing frontend dependencies..."
    npm install
    echo "✓ Frontend dependencies installed"
else
    echo "Warning: npm not found. Please install Node.js and run 'npm install' in the frontend directory."
fi

cd ..

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << EOF
# PINN Seismic Inversion Configuration

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Data directories
DATA_DIR=./data
CHECKPOINT_DIR=./checkpoints
EXPERIMENTS_DIR=./experiments
LOGS_DIR=./logs

# Training defaults
DEFAULT_DEVICE=auto
DEFAULT_BATCH_SIZE=4096
DEFAULT_LEARNING_RATE=0.0001

# GPU Settings (optional)
CUDA_VISIBLE_DEVICES=0
EOF
    echo "✓ .env file created"
fi

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "To get started:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start the backend API:"
echo "     python -m uvicorn src.api.main:app --reload"
echo ""
echo "  3. In a new terminal, start the frontend:"
echo "     cd frontend && npm run dev"
echo ""
echo "  4. Open http://localhost:5173 in your browser"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up --build"
echo ""
echo "For GPU Docker deployment:"
echo "  docker-compose -f docker-compose.gpu.yml up --build"
echo ""
