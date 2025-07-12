"""
Unit tests for model architectures.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# Import models
from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier
from models.hybrid_cnn_lstm import HybridClassifer
from models.hf_wrapper import HuggingFaceModelWrapper
from models.phow2vec import PhoW2VecWrapper


class TestPhoW2VecWrapper:
    """Test cases for PhoW2VecWrapper."""

    def test_initialization(self):
        """Test PhoW2VecWrapper initialization."""
        model = PhoW2VecWrapper(max_length=64)
        assert model.max_length == 64
        assert hasattr(model, 'word2vec_model')

    def test_forward_pass(self):
        """Test forward pass of PhoW2VecWrapper."""
        model = PhoW2VecWrapper(max_length=64)
        batch_size = 4
        seq_length = 10
        
        # Mock input tensor
        input_tensor = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Mock word2vec model
        model.word2vec_model = Mock()
        model.word2vec_model.return_value = torch.randn(batch_size, seq_length, 300)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, seq_length, 300)
        model.word2vec_model.assert_called_once()


class TestCNNClassifier:
    """Test cases for CNNClassifier."""

    def test_initialization(self):
        """Test CNNClassifier initialization."""
        word2vec_model = Mock()
        model = CNNClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            num_filters=300,
            filter_sizes=[3, 4, 5],
            output_dim=3,
            dropout=0.3
        )
        
        assert model.input_dim == 300
        assert model.num_filters == 300
        assert model.filter_sizes == [3, 4, 5]
        assert model.output_dim == 3
        assert model.dropout == 0.3

    def test_forward_pass(self):
        """Test forward pass of CNNClassifier."""
        word2vec_model = Mock()
        word2vec_model.return_value = torch.randn(4, 64, 300)
        
        model = CNNClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            num_filters=300,
            filter_sizes=[3, 4, 5],
            output_dim=3,
            dropout=0.3
        )
        
        input_tensor = torch.randint(0, 1000, (4, 64))
        output = model(input_tensor)
        
        assert output.shape == (4, 3)
        assert torch.is_tensor(output)

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        word2vec_model = Mock()
        model = CNNClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            num_filters=300,
            filter_sizes=[3, 4, 5],
            output_dim=3,
            dropout=0.3
        )
        
        # Check that model has parameters
        assert len(list(model.parameters())) > 0
        
        # Check that parameters are trainable
        for param in model.parameters():
            assert param.requires_grad


class TestLSTMClassifier:
    """Test cases for LSTMClassifier."""

    def test_initialization(self):
        """Test LSTMClassifier initialization."""
        word2vec_model = Mock()
        model = LSTMClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            hidden_dims=[384, 384],
            output_dim=3,
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        assert model.input_dim == 300
        assert model.hidden_dims == [384, 384]
        assert model.output_dim == 3
        assert model.n_layers == 2
        assert model.bidirectional is True
        assert model.dropout == 0.3

    def test_forward_pass(self):
        """Test forward pass of LSTMClassifier."""
        word2vec_model = Mock()
        word2vec_model.return_value = torch.randn(4, 64, 300)
        
        model = LSTMClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            hidden_dims=[384, 384],
            output_dim=3,
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        input_tensor = torch.randint(0, 1000, (4, 64))
        output = model(input_tensor)
        
        assert output.shape == (4, 3)
        assert torch.is_tensor(output)

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        word2vec_model = Mock()
        model = LSTMClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            hidden_dims=[384, 384],
            output_dim=3,
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        # Check that model has parameters
        assert len(list(model.parameters())) > 0
        
        # Check that parameters are trainable
        for param in model.parameters():
            assert param.requires_grad


class TestHybridClassifier:
    """Test cases for HybridClassifer."""

    def test_initialization(self):
        """Test HybridClassifer initialization."""
        word2vec_model = Mock()
        model = HybridClassifer(
            word2vec_model=word2vec_model,
            input_dim=300,
            lstm_hidden_dim=384,
            dropout=0.3,
            cnn_num_filters=300,
            cnn_filter_sizes=[3, 4, 5]
        )
        
        assert model.input_dim == 300
        assert model.lstm_hidden_dim == 384
        assert model.dropout == 0.3
        assert model.cnn_num_filters == 300
        assert model.cnn_filter_sizes == [3, 4, 5]

    def test_forward_pass(self):
        """Test forward pass of HybridClassifer."""
        word2vec_model = Mock()
        word2vec_model.return_value = torch.randn(4, 64, 300)
        
        model = HybridClassifer(
            word2vec_model=word2vec_model,
            input_dim=300,
            lstm_hidden_dim=384,
            dropout=0.3,
            cnn_num_filters=300,
            cnn_filter_sizes=[3, 4, 5]
        )
        
        input_tensor = torch.randint(0, 1000, (4, 64))
        output = model(input_tensor)
        
        assert output.shape == (4, 3)
        assert torch.is_tensor(output)

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        word2vec_model = Mock()
        model = HybridClassifer(
            word2vec_model=word2vec_model,
            input_dim=300,
            lstm_hidden_dim=384,
            dropout=0.3,
            cnn_num_filters=300,
            cnn_filter_sizes=[3, 4, 5]
        )
        
        # Check that model has parameters
        assert len(list(model.parameters())) > 0
        
        # Check that parameters are trainable
        for param in model.parameters():
            assert param.requires_grad


class TestHuggingFaceModelWrapper:
    """Test cases for HuggingFaceModelWrapper."""

    @patch('models.hf_wrapper.AutoTokenizer')
    @patch('models.hf_wrapper.AutoModel')
    def test_initialization(self, mock_auto_model, mock_auto_tokenizer):
        """Test HuggingFaceModelWrapper initialization."""
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        model = HuggingFaceModelWrapper(
            model_name="vinai/phobert-base-v2",
            num_classes=3,
            max_length=64
        )
        
        assert model.model_name == "vinai/phobert-base-v2"
        assert model.num_classes == 3
        assert model.max_length == 64
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("vinai/phobert-base-v2")
        mock_auto_model.from_pretrained.assert_called_once_with("vinai/phobert-base-v2")

    @patch('models.hf_wrapper.AutoTokenizer')
    @patch('models.hf_wrapper.AutoModel')
    def test_forward_pass(self, mock_auto_model, mock_auto_tokenizer):
        """Test forward pass of HuggingFaceModelWrapper."""
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (4, 64)),
            'attention_mask': torch.ones(4, 64)
        }
        
        # Mock model output
        mock_model.return_value.last_hidden_state = torch.randn(4, 64, 768)
        
        model = HuggingFaceModelWrapper(
            model_name="vinai/phobert-base-v2",
            num_classes=3,
            max_length=64
        )
        
        input_texts = ["Hello world", "Test sentence", "Another test", "Final test"]
        output = model(input_texts)
        
        assert output.shape == (4, 3)
        assert torch.is_tensor(output)

    @patch('models.hf_wrapper.AutoTokenizer')
    @patch('models.hf_wrapper.AutoModel')
    def test_model_parameters(self, mock_auto_model, mock_auto_tokenizer):
        """Test that model has trainable parameters."""
        # Mock the tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        model = HuggingFaceModelWrapper(
            model_name="vinai/phobert-base-v2",
            num_classes=3,
            max_length=64
        )
        
        # Check that model has parameters
        assert len(list(model.parameters())) > 0
        
        # Check that parameters are trainable
        for param in model.parameters():
            assert param.requires_grad


class TestModelIntegration:
    """Integration tests for model components."""

    def test_model_output_shapes(self):
        """Test that all models produce correct output shapes."""
        batch_size = 4
        seq_length = 64
        num_classes = 3
        
        # Test CNN
        word2vec_model = Mock()
        word2vec_model.return_value = torch.randn(batch_size, seq_length, 300)
        
        cnn_model = CNNClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            num_filters=300,
            filter_sizes=[3, 4, 5],
            output_dim=num_classes,
            dropout=0.3
        )
        
        input_tensor = torch.randint(0, 1000, (batch_size, seq_length))
        cnn_output = cnn_model(input_tensor)
        assert cnn_output.shape == (batch_size, num_classes)
        
        # Test LSTM
        lstm_model = LSTMClassifier(
            word2vec_model=word2vec_model,
            input_dim=300,
            hidden_dims=[384, 384],
            output_dim=num_classes,
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        lstm_output = lstm_model(input_tensor)
        assert lstm_output.shape == (batch_size, num_classes)
        
        # Test Hybrid
        hybrid_model = HybridClassifer(
            word2vec_model=word2vec_model,
            input_dim=300,
            lstm_hidden_dim=384,
            dropout=0.3,
            cnn_num_filters=300,
            cnn_filter_sizes=[3, 4, 5]
        )
        
        hybrid_output = hybrid_model(input_tensor)
        assert hybrid_output.shape == (batch_size, num_classes)

    def test_model_device_compatibility(self):
        """Test that models can be moved to different devices."""
        word2vec_model = Mock()
        word2vec_model.return_value = torch.randn(4, 64, 300)
        
        models = [
            CNNClassifier(
                word2vec_model=word2vec_model,
                input_dim=300,
                num_filters=300,
                filter_sizes=[3, 4, 5],
                output_dim=3,
                dropout=0.3
            ),
            LSTMClassifier(
                word2vec_model=word2vec_model,
                input_dim=300,
                hidden_dims=[384, 384],
                output_dim=3,
                n_layers=2,
                bidirectional=True,
                dropout=0.3
            ),
            HybridClassifer(
                word2vec_model=word2vec_model,
                input_dim=300,
                lstm_hidden_dim=384,
                dropout=0.3,
                cnn_num_filters=300,
                cnn_filter_sizes=[3, 4, 5]
            )
        ]
        
        for model in models:
            # Test CPU
            model_cpu = model.to('cpu')
            assert next(model_cpu.parameters()).device.type == 'cpu'
            
            # Test CUDA if available
            if torch.cuda.is_available():
                model_cuda = model.to('cuda')
                assert next(model_cuda.parameters()).device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__])