#!/bin/bash

# Deployment script for Vietnamese Sentiment Analysis

echo "Starting deployment process..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository. Please initialize git first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run all tests
echo "Running tests..."
python -m pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Deployment aborted."
    exit 1
fi

# Run code quality checks
echo "Running code quality checks..."
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127
if [ $? -ne 0 ]; then
    echo "❌ Linting failed. Deployment aborted."
    exit 1
fi

# Check formatting
echo "Checking code formatting..."
black --check --diff .
if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed. Run 'black .' to fix."
    exit 1
fi

# Type checking
echo "Running type checking..."
mypy . --ignore-missing-imports
if [ $? -ne 0 ]; then
    echo "⚠️  Type checking warnings found, but continuing deployment..."
fi

# Security checks
echo "Running security checks..."
bandit -r . -f json -o bandit-report.json
safety check

# Build documentation
echo "Building documentation..."
cd docs
make html
cd ..

# Create deployment package
echo "Creating deployment package..."
mkdir -p dist
tar -czf dist/vietnamese-sentiment-analysis-$(date +%Y%m%d-%H%M%S).tar.gz \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.coverage' \
    --exclude='htmlcov' \
    --exclude='checkpoints' \
    --exclude='logs' \
    .

echo "✅ Deployment package created successfully!"
echo ""
echo "Deployment package location: dist/"
echo ""
echo "To deploy to production:"
echo "1. Upload the package to your server"
echo "2. Extract the package"
echo "3. Run: pip install -r requirements.txt"
echo "4. Run: python run.py"