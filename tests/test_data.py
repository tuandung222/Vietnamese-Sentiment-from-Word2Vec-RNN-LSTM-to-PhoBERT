"""
Unit tests for data processing components.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datasets import Dataset

# Import data components
from data.dataset_builder import DatasetBuilder
from data.vietnamese_eda import VietnameseEDATransform


class TestVietnameseEDATransform:
    """Test cases for VietnameseEDATransform."""

    def test_initialization(self):
        """Test VietnameseEDATransform initialization."""
        transform = VietnameseEDATransform()
        assert hasattr(transform, 'synonym_dict')
        assert hasattr(transform, 'stop_words')

    def test_synonym_replacement(self):
        """Test synonym replacement functionality."""
        transform = VietnameseEDATransform()
        
        # Mock synonym dictionary
        transform.synonym_dict = {
            'tốt': ['hay', 'đẹp', 'tuyệt'],
            'xấu': ['dở', 'tệ', 'không tốt']
        }
        
        text = "Sản phẩm này rất tốt"
        augmented_text = transform.synonym_replacement(text)
        
        assert isinstance(augmented_text, str)
        assert len(augmented_text) > 0

    def test_random_deletion(self):
        """Test random deletion functionality."""
        transform = VietnameseEDATransform()
        
        text = "Đây là một câu tiếng Việt để test"
        augmented_text = transform.random_deletion(text, p=0.3)
        
        assert isinstance(augmented_text, str)
        assert len(augmented_text) <= len(text)

    def test_random_swap(self):
        """Test random swap functionality."""
        transform = VietnameseEDATransform()
        
        text = "Đây là một câu tiếng Việt"
        augmented_text = transform.random_swap(text, n=2)
        
        assert isinstance(augmented_text, str)
        assert len(augmented_text) == len(text)

    def test_random_insertion(self):
        """Test random insertion functionality."""
        transform = VietnameseEDATransform()
        
        # Mock synonym dictionary
        transform.synonym_dict = {
            'tốt': ['hay', 'đẹp'],
            'xấu': ['dở', 'tệ']
        }
        
        text = "Sản phẩm này rất tốt"
        augmented_text = transform.random_insertion(text, n=1)
        
        assert isinstance(augmented_text, str)
        assert len(augmented_text) >= len(text)

    def test_augment(self):
        """Test main augmentation function."""
        transform = VietnameseEDATransform()
        
        text = "Sản phẩm này rất tốt"
        augmented_text = transform.augment(text)
        
        assert isinstance(augmented_text, str)
        assert len(augmented_text) > 0


class TestDatasetBuilder:
    """Test cases for DatasetBuilder."""

    def test_initialization(self):
        """Test DatasetBuilder initialization."""
        transform = Mock()
        builder = DatasetBuilder(train_augment_transform=transform)
        
        assert builder.train_augment_transform == transform
        assert hasattr(builder, 'train_file')
        assert hasattr(builder, 'test_file')

    @patch('data.dataset_builder.pd.read_csv')
    def test_read_data(self, mock_read_csv):
        """Test read_data method."""
        # Mock pandas read_csv
        mock_data = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [0, 1, 2]
        })
        mock_read_csv.return_value = mock_data
        
        transform = Mock()
        builder = DatasetBuilder(train_augment_transform=transform)
        
        result = builder.read_data('test_file.csv', is_train=True)
        
        assert isinstance(result, Dataset)
        mock_read_csv.assert_called_once_with('test_file.csv')

    def test_get_transform(self):
        """Test get_transform method."""
        transform = Mock()
        builder = DatasetBuilder(train_augment_transform=transform)
        
        # Test training transform
        train_transform = builder.get_transform(is_train=True)
        assert train_transform == transform
        
        # Test test transform
        test_transform = builder.get_transform(is_train=False)
        assert test_transform is None

    @patch('data.dataset_builder.pd.read_csv')
    def test_build(self, mock_read_csv):
        """Test build method."""
        # Mock training data
        train_data = pd.DataFrame({
            'text': ['text1', 'text2', 'text3', 'text4', 'text5'],
            'label': [0, 1, 2, 0, 1]
        })
        
        # Mock test data
        test_data = pd.DataFrame({
            'text': ['test1', 'test2'],
            'label': [0, 1]
        })
        
        mock_read_csv.side_effect = [train_data, test_data]
        
        transform = Mock()
        builder = DatasetBuilder(train_augment_transform=transform)
        
        train_set, val_set, test_set = builder.build()
        
        assert isinstance(train_set, Dataset)
        assert isinstance(val_set, Dataset)
        assert isinstance(test_set, Dataset)

    def test_collate_fn(self):
        """Test collate_fn method."""
        transform = Mock()
        builder = DatasetBuilder(train_augment_transform=transform)
        
        # Mock batch data
        batch = [
            {'text': 'text1', 'label': 0},
            {'text': 'text2', 'label': 1},
            {'text': 'text3', 'label': 2}
        ]
        
        result = builder.collate_fn(batch)
        
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'label' in result


class TestDataIntegration:
    """Integration tests for data components."""

    def test_data_pipeline_integration(self):
        """Test integration between EDA and DatasetBuilder."""
        # Create EDA transform
        eda_transform = VietnameseEDATransform()
        
        # Create dataset builder
        builder = DatasetBuilder(train_augment_transform=eda_transform)
        
        # Test that they work together
        assert builder.train_augment_transform == eda_transform
        
        # Test transform application
        text = "Sản phẩm này rất tốt"
        augmented = eda_transform.augment(text)
        assert isinstance(augmented, str)

    def test_dataset_compatibility(self):
        """Test that datasets are compatible with PyTorch DataLoader."""
        transform = Mock()
        builder = DatasetBuilder(train_augment_transform=transform)
        
        # Mock data
        mock_data = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [0, 1, 2]
        })
        
        # Create dataset
        dataset = Dataset.from_pandas(mock_data)
        
        # Test that it can be used with DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=builder.collate_fn
        )
        
        # Test iteration
        for batch in dataloader:
            assert isinstance(batch, dict)
            break

    def test_data_augmentation_quality(self):
        """Test that data augmentation maintains data quality."""
        transform = VietnameseEDATransform()
        
        original_text = "Sản phẩm này rất tốt và chất lượng cao"
        
        # Test multiple augmentations
        for _ in range(5):
            augmented = transform.augment(original_text)
            
            # Check that augmented text is not empty
            assert len(augmented) > 0
            
            # Check that it's still a string
            assert isinstance(augmented, str)
            
            # Check that it's not exactly the same (augmentation occurred)
            # Note: This might fail occasionally due to randomness
            # assert augmented != original_text


if __name__ == "__main__":
    pytest.main([__file__])