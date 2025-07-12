# Vietnamese Sentiment Analysis - VLSP 2016

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-blue.svg)](https://pytest.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements state-of-the-art deep learning models for Vietnamese sentiment analysis using the **VLSP 2016 Vietnamese Sentiment Analysis** dataset. The project is designed with MLOps best practices, including automated testing, CI/CD pipelines, and comprehensive documentation.

### Key Features

- **Multiple Model Architectures**: CNN, LSTM, Hybrid CNN-LSTM, and Transformer-based models
- **Data Augmentation**: Vietnamese-specific text augmentation techniques
- **Experiment Tracking**: Weights & Biases integration for experiment management
- **Automated Testing**: Comprehensive unit tests and integration tests
- **CI/CD Pipeline**: Automated testing and deployment workflows
- **Professional Documentation**: Detailed API documentation and usage examples

## 🚀 Features

### Model Architectures
- **CNN Classifier**: Convolutional Neural Network for text classification
- **LSTM Classifier**: Bidirectional LSTM with attention mechanisms
- **Hybrid CNN-LSTM**: Combined architecture for enhanced performance
- **PhoBERT**: Vietnamese-specific transformer model
- **PhoW2Vec**: Vietnamese word embeddings

### Data Processing
- **Vietnamese EDA**: Easy Data Augmentation for Vietnamese text
- **Dataset Builder**: Automated dataset preparation and splitting
- **Text Preprocessing**: Vietnamese-specific tokenization and cleaning

### MLOps Features
- **Automated Testing**: Unit tests, integration tests, and model validation
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Code Quality**: Black formatting, flake8 linting, and type hints
- **Documentation**: Comprehensive API documentation and tutorials

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/vietnamese-sentiment-analysis.git

cd vietnamese-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests to verify installation
pytest tests/
```

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Install package in development mode
pip install -e .
```

## 🎮 Usage

### Quick Experiment

```bash
# Run the main experiment pipeline
./run.sh

# Or run directly with Python
python run.py
```

### Interactive Development

```bash
# Start Jupyter notebook for interactive development
jupyter notebook interactive.ipynb
```

### Model Training

```python
from engine import MyTrainer, TraininingConfig
from models.cnn import CNNClassifier
from data.dataset_builder import DatasetBuilder

# Build dataset
dataset_builder = DatasetBuilder()
train_set, val_set, test_set = dataset_builder.build()

# Initialize model
model = CNNClassifier(
    word2vec_model=word2vec_model,
    input_dim=300,
    num_filters=300,
    filter_sizes=[3, 4, 5],
    output_dim=3,
    dropout=0.3,
)

# Configure training
config = TraininingConfig(
    max_epochs=50,
    batch_size=1024,
    lr=7.5e-4,
    output_dir="checkpoints"
)

# Train model
trainer = MyTrainer(config)
result = trainer.train_evaluate_pipeline(
    model=model,
    train_dataset=train_set,
    val_dataset=val_set,
    test_dataset=test_set
)
```

## 📁 Project Structure

```
vietnamese-sentiment-analysis/
├── .github/                    # GitHub Actions CI/CD
│   └── workflows/
├── data/                       # Data processing modules
│   ├── dataset_builder.py
│   └── vietnamese_eda.py
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── cnn.py
│   ├── lstm.py
│   ├── hybrid_cnn_lstm.py
│   ├── hf_wrapper.py
│   └── phow2vec.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_engine.py
├── docs/                       # Documentation
│   ├── api.md
│   └── tutorials/
├── scripts/                    # Utility scripts
│   ├── setup.sh
│   └── deploy.sh
├── engine.py                   # Training engine
├── run.py                      # Main experiment script
├── run.sh                      # Quick run script
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                   # Package configuration
├── pyproject.toml            # Project configuration
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .flake8                   # Linting configuration
└── README.md                 # This file
```

## 🤖 Models

### CNN Classifier
- **Architecture**: Convolutional Neural Network with multiple filter sizes
- **Features**: Text classification using word embeddings
- **Best Performance**: 70.19% accuracy on VLSP 2016

### LSTM Classifier
- **Architecture**: Bidirectional LSTM with attention
- **Features**: Sequential text processing with memory
- **Performance**: 63.43% accuracy on VLSP 2016

### Hybrid CNN-LSTM
- **Architecture**: Combined CNN and LSTM layers
- **Features**: Captures both local and sequential patterns
- **Performance**: 68.10% accuracy on VLSP 2016

### PhoBERT
- **Architecture**: Vietnamese-specific transformer
- **Features**: Pre-trained on Vietnamese text corpus
- **Performance**: 50.10% accuracy on VLSP 2016

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-----------------|--------------|
| CNN | 70.19% | 70.24% | 70.34% | 70.19% |
| LSTM | 63.43% | 63.43% | 63.56% | 63.43% |
| Hybrid CNN-LSTM | 68.10% | 68.28% | 69.09% | 68.10% |
| PhoBERT | 50.10% | 49.00% | 48.56% | 50.10% |

### Key Findings
- **CNN performs best** on the VLSP 2016 dataset
- **Hybrid models** show competitive performance
- **Transformer models** may need more Vietnamese-specific pre-training

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run all quality checks
pre-commit run --all-files
```

### Building Documentation

```bash
# Build documentation
cd docs
make html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VLSP 2016 dataset providers
- PhoBERT and PhoW2Vec authors
- Weights & Biases for experiment tracking
- The open-source community

## 📞 Contact

- **Project Link**: [https://github.com/yourusername/vietnamese-sentiment-analysis](https://github.com/yourusername/vietnamese-sentiment-analysis)
- **Issues**: [GitHub Issues](https://github.com/yourusername/vietnamese-sentiment-analysis/issues)

---

**Note**: This project is actively maintained and follows MLOps best practices. For the latest updates, please check the [releases page](https://github.com/yourusername/vietnamese-sentiment-analysis/releases).

