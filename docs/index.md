# Vietnamese Sentiment Analysis Documentation

Welcome to the Vietnamese Sentiment Analysis documentation! This project implements state-of-the-art deep learning models for Vietnamese sentiment analysis using the VLSP 2016 dataset.

## Quick Start

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

## Features

- **Multiple Model Architectures**: CNN, LSTM, Hybrid CNN-LSTM, and Transformer-based models
- **Data Augmentation**: Vietnamese-specific text augmentation techniques
- **Experiment Tracking**: Weights & Biases integration for experiment management
- **Automated Testing**: Comprehensive unit tests and integration tests
- **CI/CD Pipeline**: Automated testing and deployment workflows
- **Professional Documentation**: Detailed API documentation and usage examples

## Installation

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

## Model Performance

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-----------------|--------------|
| CNN | 70.19% | 70.24% | 70.34% | 70.19% |
| LSTM | 63.43% | 63.43% | 63.56% | 63.43% |
| Hybrid CNN-LSTM | 68.10% | 68.28% | 69.09% | 68.10% |
| PhoBERT | 50.10% | 49.00% | 48.56% | 50.10% |

## Documentation Sections

- [API Reference](api.md) - Complete API documentation
- [Tutorials](tutorials/) - Step-by-step tutorials
- [Models](models/) - Model architecture details
- [Data Processing](data/) - Data pipeline documentation
- [Training](training/) - Training configuration and procedures

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.