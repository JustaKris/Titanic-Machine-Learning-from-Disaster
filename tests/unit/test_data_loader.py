"""
Unit tests for data loading module.
"""

from pathlib import Path

import pandas as pd
import pytest

from titanic_ml.data.loader import DataLoader, load_kaggle_test_data
from titanic_ml.utils.exception import CustomException


@pytest.mark.unit
class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_init_default_paths(self):
        """Test DataLoader initialization with default paths."""
        loader = DataLoader()
        assert loader.raw_train_path is not None
        assert loader.test_size == 0.2
        assert loader.random_state == 42

    def test_init_custom_paths(self, temp_dir):
        """Test DataLoader initialization with custom paths."""
        train_path = temp_dir / "train.csv"
        test_path = temp_dir / "test.csv"

        loader = DataLoader(
            raw_train_path=train_path,
            raw_test_path=test_path,
            test_size=0.3,
            random_state=123,
        )

        assert loader.raw_train_path == train_path
        assert loader.raw_test_path == test_path
        assert loader.test_size == 0.3
        assert loader.random_state == 123

    def test_load_data_success(self, sample_train_data, temp_dir):
        """Test successful data loading and splitting."""
        # Save sample data to temp file
        train_path = temp_dir / "train.csv"
        sample_train_data.to_csv(train_path, index=False)

        # Create loader
        loader = DataLoader(
            raw_train_path=train_path,
            test_size=0.3,
            random_state=42,
        )

        # Load data (this creates processed splits)
        processed_train, processed_test = loader.load_data()

        # Verify files were created
        assert processed_train.exists()
        assert processed_test.exists()

        # Load and verify splits
        train_df = pd.read_csv(processed_train)
        test_df = pd.read_csv(processed_test)

        # Check split sizes
        total = len(sample_train_data)
        expected_test_size = int(total * 0.3)

        assert len(test_df) == expected_test_size
        assert len(train_df) == total - expected_test_size

        # Verify stratification
        assert "Survived" in train_df.columns
        assert "Survived" in test_df.columns

    def test_load_data_file_not_found(self, temp_dir):
        """Test error handling when raw data file doesn't exist."""
        fake_path = temp_dir / "nonexistent.csv"
        loader = DataLoader(raw_train_path=fake_path)

        with pytest.raises(CustomException):
            loader.load_data()

    def test_load_data_preserves_columns(self, sample_train_data, temp_dir):
        """Test that all columns are preserved during loading."""
        train_path = temp_dir / "train.csv"
        sample_train_data.to_csv(train_path, index=False)

        loader = DataLoader(raw_train_path=train_path)
        processed_train, processed_test = loader.load_data()

        train_df = pd.read_csv(processed_train)
        test_df = pd.read_csv(processed_test)

        # Check all original columns are present
        original_cols = set(sample_train_data.columns)
        assert set(train_df.columns) == original_cols
        assert set(test_df.columns) == original_cols


@pytest.mark.unit
class TestLoadKaggleTestData:
    """Test suite for load_kaggle_test_data function."""

    def test_load_kaggle_test_success(self, sample_test_data, temp_dir):
        """Test successful loading of Kaggle test data."""
        test_path = temp_dir / "test.csv"
        sample_test_data.to_csv(test_path, index=False)

        df = load_kaggle_test_data(test_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_test_data)
        assert list(df.columns) == list(sample_test_data.columns)

    def test_load_kaggle_test_file_not_found(self, temp_dir):
        """Test error when Kaggle test file doesn't exist."""
        fake_path = temp_dir / "nonexistent.csv"

        with pytest.raises(CustomException):
            load_kaggle_test_data(fake_path)

    def test_load_kaggle_test_preserves_data_types(self, temp_dir):
        """Test that data types are preserved when loading."""
        # Create test data with specific types
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3],
                "Pclass": [1, 2, 3],
                "Name": ["A", "B", "C"],
                "Age": [25.5, 30.0, None],
                "Fare": [7.25, 10.50, 15.75],
            }
        )

        test_path = temp_dir / "test.csv"
        df.to_csv(test_path, index=False)

        loaded_df = load_kaggle_test_data(test_path)

        assert loaded_df["PassengerId"].dtype == df["PassengerId"].dtype
        assert loaded_df["Name"].dtype == object
        assert loaded_df["Fare"].dtype == float
