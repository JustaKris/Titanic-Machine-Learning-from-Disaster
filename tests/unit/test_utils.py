"""
Unit tests for utility modules (helpers, logger, exception).
"""

import logging
import sys
from pathlib import Path

import dill
import numpy as np
import pytest

from src.utils.exception import CustomException
from src.utils.helpers import load_object, save_object
from src.utils.logger import configure_logging, get_logger


@pytest.mark.unit
class TestHelpers:
    """Test suite for helper functions."""

    def test_save_object_success(self, temp_dir):
        """Test saving object to file."""
        test_obj = {"key": "value", "number": 42}
        file_path = temp_dir / "test_object.pkl"

        save_object(file_path, test_obj)

        # Verify file was created
        assert file_path.exists()

        # Verify content
        with open(file_path, "rb") as f:
            loaded = dill.load(f)
        assert loaded == test_obj

    def test_save_object_creates_directory(self, temp_dir):
        """Test that save_object creates parent directories."""
        file_path = temp_dir / "subdir" / "nested" / "object.pkl"
        test_obj = [1, 2, 3]

        save_object(file_path, test_obj)

        assert file_path.exists()
        assert file_path.parent.exists()

    def test_save_object_complex_types(self, temp_dir):
        """Test saving complex Python objects."""
        import pandas as pd

        test_obj = {
            "array": np.array([1, 2, 3]),
            "dataframe": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "nested": {"list": [1, 2], "tuple": (3, 4)},
        }

        file_path = temp_dir / "complex.pkl"
        save_object(file_path, test_obj)

        loaded = load_object(file_path)

        np.testing.assert_array_equal(loaded["array"], test_obj["array"])
        pd.testing.assert_frame_equal(loaded["dataframe"], test_obj["dataframe"])
        assert loaded["nested"] == test_obj["nested"]

    def test_load_object_success(self, temp_dir):
        """Test loading object from file."""
        test_obj = {"test": "data"}
        file_path = temp_dir / "test.pkl"

        # Save first
        with open(file_path, "wb") as f:
            dill.dump(test_obj, f)

        # Load
        loaded = load_object(file_path)
        assert loaded == test_obj

    def test_load_object_file_not_found(self, temp_dir):
        """Test error when loading non-existent file."""
        fake_path = temp_dir / "nonexistent.pkl"

        with pytest.raises(Exception):  # Could be FileNotFoundError or CustomException
            load_object(fake_path)

    def test_save_and_load_sklearn_model(self, temp_dir):
        """Test saving and loading sklearn model."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(X, y)

        file_path = temp_dir / "model.pkl"
        save_object(file_path, model)

        loaded_model = load_object(file_path)

        # Test predictions are same
        assert (model.predict(X) == loaded_model.predict(X)).all()


@pytest.mark.unit
class TestLogger:
    """Test suite for logging utilities."""

    def test_configure_logging_default(self):
        """Test logger configuration with default settings."""
        configure_logging()
        logger = logging.getLogger()

        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_configure_logging_custom_level(self):
        """Test logger configuration with custom level."""
        configure_logging(level="DEBUG")
        logger = logging.getLogger()

        assert logger.level == logging.DEBUG

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_logger_output(self, caplog, capsys):
        """Test that logger actually logs messages."""
        configure_logging(level="INFO")
        logger = get_logger("test")

        logger.info("Test message")

        # Check either caplog or stdout
        captured = capsys.readouterr()
        assert "Test message" in caplog.text or "Test message" in captured.out

    def test_logger_levels(self, caplog, capsys):
        """Test different logging levels."""
        configure_logging(level="DEBUG")
        logger = get_logger("test")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Check output (may be in caplog or stdout)
        captured = capsys.readouterr()
        output = caplog.text + captured.out

        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output

    def test_logger_structured_logging(self, caplog, capsys):
        """Test structured logging with extra fields."""
        configure_logging(level="INFO")
        logger = get_logger("test")

        logger.info(
            "Prediction made",
            extra={"extra_fields": {"prediction": 1, "probability": 0.85}},
        )

        # Check either caplog or stdout
        captured = capsys.readouterr()
        assert "Prediction made" in caplog.text or "Prediction made" in captured.out


@pytest.mark.unit
class TestCustomException:
    """Test suite for CustomException class."""

    def test_custom_exception_creation(self):
        """Test creating CustomException."""
        error_msg = "Test error"
        original_error = ValueError(error_msg)

        custom_exc = CustomException(original_error, sys)

        assert isinstance(custom_exc, Exception)
        assert error_msg in str(custom_exc)

    def test_custom_exception_preserves_traceback(self):
        """Test that CustomException preserves stack trace info."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            custom_exc = CustomException(e, sys)
            exc_str = str(custom_exc)

            # Should contain file and line information
            assert "test_utils.py" in exc_str or "line" in exc_str.lower()

    def test_custom_exception_wrapping(self):
        """Test wrapping different exception types."""
        errors = [
            ValueError("value error"),
            TypeError("type error"),
            KeyError("key error"),
            AttributeError("attribute error"),
        ]

        for error in errors:
            custom_exc = CustomException(error, sys)
            assert isinstance(custom_exc, Exception)
            assert str(error) in str(custom_exc) or type(error).__name__ in str(custom_exc)

    def test_custom_exception_in_function(self):
        """Test using CustomException in actual function."""

        def failing_function():
            try:
                # Simulate some failing operation
                result = 1 / 0
                return result
            except Exception as e:
                raise CustomException(e, sys)

        with pytest.raises(CustomException) as exc_info:
            failing_function()

        # Exception message should contain useful info
        exc_str = str(exc_info.value)
        assert "division" in exc_str.lower() or "zero" in exc_str.lower()
