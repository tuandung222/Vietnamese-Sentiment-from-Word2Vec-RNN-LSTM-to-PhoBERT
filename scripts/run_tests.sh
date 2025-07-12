#!/bin/bash

# Run all tests and quality checks

echo "Running tests and quality checks..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests
echo "Running tests..."
python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# Run linting
echo "Running linting..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run type checking
echo "Running type checking..."
mypy . --ignore-missing-imports

# Run formatting check
echo "Running formatting check..."
black --check --diff .

# Run security checks
echo "Running security checks..."
bandit -r . -f json -o bandit-report.json || true
safety check

echo "All checks completed!"