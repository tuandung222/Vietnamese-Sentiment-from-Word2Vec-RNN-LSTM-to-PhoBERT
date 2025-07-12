# Vietnamese Sentiment Analysis - Professional Deep Learning Project

## � Project Overview

This project implements state-of-the-art deep learning models for Vietnamese sentiment analysis using the VLSP 2016 dataset. The project follows professional deep learning development practices with comprehensive testing, documentation, and MLOps integration.

## 🏗️ Project Structure

```
vietnamese-sentiment-analysis/
├── configs/                 # Configuration files (Hydra)
├── src/                     # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # Model architectures
│   ├── training/           # Training pipeline
│   ├── evaluation/         # Evaluation utilities
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks
├── .github/                # GitHub workflows
└── docker/                 # Docker configurations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd vietnamese-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Experiments

```bash
# Run a single experiment
python src/main.py experiment=cnn_baseline

# Run multiple experiments
python src/main.py -m experiment=cnn_baseline,lstm_baseline,phobert_baseline

# Run with custom configuration
python src/main.py experiment=cnn_baseline training.batch_size=256 training.lr=1e-4
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test category
pytest tests/test_models.py
```

## 📊 Model Architectures

- **CNN**: Convolutional Neural Network with PhoW2Vec embeddings
- **LSTM**: Bidirectional LSTM with attention mechanism
- **Hybrid CNN-LSTM**: Combined architecture for enhanced performance
- **PhoBERT**: Pre-trained Vietnamese BERT model

## 🔧 Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/experiment/`: Experiment configurations
- `configs/training/`: Training hyperparameters
- `configs/data/`: Dataset configurations
- `configs/model/`: Model architectures

## 📈 Experiment Tracking

We use Weights & Biases for experiment tracking. Configure your W&B credentials:

```bash
wandb login
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t vietnamese-sentiment .

# Run container
docker run -it --gpus all vietnamese-sentiment
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Model Architecture](docs/models.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VLSP 2016 dataset organizers
- PhoBERT and PhoW2Vec model developers
- The open-source NLP community
