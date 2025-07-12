# API Reference

This document provides comprehensive API documentation for the Vietnamese Sentiment Analysis project.

## Models

### CNNClassifier

Convolutional Neural Network for text classification.

```python
class CNNClassifier(nn.Module):
    def __init__(
        self,
        word2vec_model,
        input_dim: int,
        num_filters: int,
        filter_sizes: List[int],
        output_dim: int,
        dropout: float = 0.3
    ):
        """
        Initialize CNN classifier.
        
        Args:
            word2vec_model: Word embedding model
            input_dim: Input dimension
            num_filters: Number of filters
            filter_sizes: List of filter sizes
            output_dim: Output dimension
            dropout: Dropout rate
        """
```

**Methods:**
- `forward(x)`: Forward pass through the network

### LSTMClassifier

Bidirectional LSTM with attention for text classification.

```python
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        word2vec_model,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM classifier.
        
        Args:
            word2vec_model: Word embedding model
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            n_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
        """
```

**Methods:**
- `forward(x)`: Forward pass through the network

### HybridClassifer

Combined CNN and LSTM architecture.

```python
class HybridClassifer(nn.Module):
    def __init__(
        self,
        word2vec_model,
        input_dim: int,
        lstm_hidden_dim: int,
        dropout: float = 0.3,
        cnn_num_filters: int = 300,
        cnn_filter_sizes: List[int] = [3, 4, 5]
    ):
        """
        Initialize hybrid CNN-LSTM classifier.
        
        Args:
            word2vec_model: Word embedding model
            input_dim: Input dimension
            lstm_hidden_dim: LSTM hidden dimension
            dropout: Dropout rate
            cnn_num_filters: Number of CNN filters
            cnn_filter_sizes: List of CNN filter sizes
        """
```

**Methods:**
- `forward(x)`: Forward pass through the network

### HuggingFaceModelWrapper

Wrapper for HuggingFace transformer models.

```python
class HuggingFaceModelWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        max_length: int = 64
    ):
        """
        Initialize HuggingFace model wrapper.
        
        Args:
            model_name: Name of the HuggingFace model
            num_classes: Number of output classes
            max_length: Maximum sequence length
        """
```

**Methods:**
- `forward(texts)`: Forward pass through the network

## Data Processing

### DatasetBuilder

Dataset builder for VLSP 2016 sentiment analysis.

```python
class DatasetBuilder:
    def __init__(self, train_augment_transform=None):
        """
        Initialize dataset builder.
        
        Args:
            train_augment_transform: Data augmentation transform for training
        """
```

**Methods:**
- `build()`: Build train, validation, and test datasets
- `read_data(file_path, is_train)`: Read data from CSV file
- `get_transform(is_train)`: Get data transform
- `collate_fn(batch)`: Collate function for DataLoader

### VietnameseEDATransform

Vietnamese-specific text augmentation.

```python
class VietnameseEDATransform:
    def __init__(self):
        """Initialize Vietnamese EDA transform."""
```

**Methods:**
- `augment(text)`: Apply data augmentation to text
- `synonym_replacement(text, n=1)`: Replace words with synonyms
- `random_deletion(text, p=0.1)`: Randomly delete words
- `random_swap(text, n=1)`: Randomly swap word positions
- `random_insertion(text, n=1)`: Randomly insert synonyms

## Training

### MyTrainer

Training engine for sentiment analysis models.

```python
class MyTrainer:
    def __init__(self, config: TraininingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
```

**Methods:**
- `train_evaluate_pipeline(...)`: Complete training and evaluation pipeline
- `train(...)`: Train the model
- `evaluate(...)`: Evaluate the model
- `calculate_accuracy(...)`: Calculate accuracy metrics

### TraininingConfig

Training configuration dataclass.

```python
@dataclass
class TraininingConfig:
    max_epochs: int
    max_patience: int
    batch_size: int
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    output_dir: str
```

## Visualization

### Visualization Functions

```python
def show_compared_test_table_prettytable(result_objects):
    """Display test results in a pretty table format."""
    
def draw_training_history(result_objects):
    """Draw training history plots."""
```

## Configuration

### Model Configuration Examples

```python
# CNN Configuration
cnn_config = {
    "input_dim": 300,
    "num_filters": 300,
    "filter_sizes": [3, 4, 5, 6, 7, 8],
    "output_dim": 3,
    "dropout": 0.3
}

# LSTM Configuration
lstm_config = {
    "input_dim": 300,
    "hidden_dims": [384, 384],
    "output_dim": 3,
    "n_layers": 2,
    "bidirectional": True,
    "dropout": 0.3
}

# Training Configuration
training_config = TraininingConfig(
    max_epochs=128,
    max_patience=12,
    batch_size=1024,
    lr=7.5e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.995),
    output_dir="checkpoints"
)
```

## Usage Examples

### Basic Training Pipeline

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

### Data Augmentation

```python
from data.vietnamese_eda import VietnameseEDATransform

# Initialize augmentation
transform = VietnameseEDATransform()

# Augment text
original_text = "Sản phẩm này rất tốt"
augmented_text = transform.augment(original_text)
print(f"Original: {original_text}")
print(f"Augmented: {augmented_text}")
```

### Model Evaluation

```python
from engine import MyTrainer

# Evaluate model
trainer = MyTrainer(config)
test_results = trainer.evaluate(
    model=model,
    test_dataset=test_set,
    device="cuda"
)

print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test F1: {test_results['macro_f1']:.4f}")
```