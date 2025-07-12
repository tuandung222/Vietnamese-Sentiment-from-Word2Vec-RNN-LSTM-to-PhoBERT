#!/bin/bash

# Setup script for Vietnamese Sentiment Analysis project

echo "Setting up Vietnamese Sentiment Analysis project..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install production dependencies
echo "Installing production dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p data/processed
mkdir -p models/saved

# Set up git hooks
echo "Setting up git hooks..."
if [ -d ".git" ]; then
    echo "Git repository found. Setting up hooks..."
    git config core.hooksPath .git/hooks
else
    echo "Not a git repository. Skipping git hooks setup."
fi

# Run initial tests
echo "Running initial tests..."
python -m pytest tests/ -v --tb=short

echo "✅ Setup completed successfully!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run the main experiment:"
echo "  python run.py"
echo ""
echo "To start development:"
echo "  jupyter notebook interactive.ipynb"