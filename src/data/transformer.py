"""
Data transformation module - preprocessing and feature engineering pipelines.
"""

import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import (
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)
from src.utils.exception import CustomException
from src.utils.helpers import save_object
from src.utils.logger import logging

warnings.filterwarnings("ignore")


class DataTransformer:
    """Handles data transformation and preprocessing."""

    def __init__(self, preprocessor_path: Optional[Path] = None):
        """
        Initialize DataTransformer.

        Args:
            preprocessor_path: Path to save preprocessor object
        """
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else PREPROCESSOR_PATH

    def create_preprocessor(
        self, numerical_columns: List[str], categorical_columns: List[str]
    ) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline.

        Args:
            numerical_columns: List of numerical feature names
            categorical_columns: List of categorical feature names

        Returns:
            ColumnTransformer for preprocessing
        """
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Created numerical pipeline")

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Created categorical pipeline")

            # Combined preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error creating preprocessor: {e}")
            raise CustomException(e, sys)

    def transform_data(self, train_path: Path, test_path: Path) -> Tuple:
        """
        Apply feature engineering and preprocessing to train/test data.

        Args:
            train_path: Path to training data
            test_path: Path to test data

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, preprocessor_path)
        """
        try:
            # Validate paths
            train_path = Path(train_path)
            test_path = Path(test_path)

            if not train_path.exists():
                raise FileNotFoundError(f"Train file not found: {train_path}")
            if not test_path.exists():
                raise FileNotFoundError(f"Test file not found: {test_path}")

            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test data for transformation")

            # Apply feature engineering with advanced features
            from src.features.build_features import apply_feature_engineering

            train_df, num_cols, cat_cols = apply_feature_engineering(
                train_df, include_advanced=True, is_training=True
            )
            test_df, _, _ = apply_feature_engineering(
                test_df, include_advanced=True, is_training=True
            )

            # Save augmented data
            augmented_train = PROCESSED_DATA_DIR / "train_augmented.csv"
            augmented_test = PROCESSED_DATA_DIR / "test_augmented.csv"

            train_df.to_csv(augmented_train, index=False)
            test_df.to_csv(augmented_test, index=False)
            logging.info(f"Saved augmented data to {PROCESSED_DATA_DIR}")

            # Create and fit preprocessor
            preprocessor = self.create_preprocessor(num_cols, cat_cols)

            # Transform data
            X_train = preprocessor.fit_transform(train_df.drop(columns=[TARGET_COLUMN]))
            y_train = train_df[TARGET_COLUMN].values

            X_test = preprocessor.transform(test_df.drop(columns=[TARGET_COLUMN]))
            y_test = test_df[TARGET_COLUMN].values

            logging.info("Data transformation completed")

            # Save preprocessor
            save_object(self.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved to {self.preprocessor_path}")

            return X_train, y_train, X_test, y_test, self.preprocessor_path

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.data.loader import DataLoader

    # Load data
    loader = DataLoader()
    train_path, test_path = loader.load_data()

    # Transform data
    transformer = DataTransformer()
    X_train, y_train, X_test, y_test, preprocessor_path = transformer.transform_data(
        train_path, test_path
    )

    from src.utils.logger import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("Running data transformer standalone")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"Preprocessor saved to: {preprocessor_path}")
