"""
Data loading module - handles data ingestion from raw sources.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_ml.config.settings import (
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RANDOM_STATE,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    TEST_SIZE,
)
from titanic_ml.utils.exception import CustomException
from titanic_ml.utils.logger import logging


class DataLoader:
    """Handles loading and initial splitting of raw data."""

    def __init__(
        self,
        raw_train_path: Optional[Path] = None,
        raw_test_path: Optional[Path] = None,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        """
        Initialize DataLoader.

        Args:
            raw_train_path: Path to raw training data
            raw_test_path: Path to raw test data (optional)
            test_size: Proportion of data for test split
            random_state: Random seed for reproducibility
        """
        self.raw_train_path = Path(raw_train_path) if raw_train_path else RAW_TRAIN_PATH
        self.raw_test_path = Path(raw_test_path) if raw_test_path else RAW_TEST_PATH
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> Tuple[Path, Path]:
        """
        Load raw data, split into train/test, and save processed files.

        Returns:
            Tuple of (train_path, test_path) for processed data
        """
        logging.info("Starting data ingestion")

        try:
            # Read raw training data
            if not self.raw_train_path.exists():
                raise FileNotFoundError(f"Raw train data not found: {self.raw_train_path}")

            df = pd.read_csv(self.raw_train_path)
            logging.info(f"Loaded dataset with shape: {df.shape}")

            # Create processed data directory
            PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Train-test split
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df["Survived"] if "Survived" in df.columns else None,
            )

            # Save processed splits
            train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
            test_df.to_csv(PROCESSED_TEST_PATH, index=False)

            logging.info(f"Train data saved: {PROCESSED_TRAIN_PATH} ({len(train_df)} rows)")
            logging.info(f"Test data saved: {PROCESSED_TEST_PATH} ({len(test_df)} rows)")

            return PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(e, sys)
        except pd.errors.EmptyDataError as e:
            logging.error(f"Empty data error: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"Unexpected error in data loading: {e}")
            raise CustomException(e, sys)


def load_kaggle_test_data(test_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Kaggle test dataset (without labels).

    Args:
        test_path: Path to Kaggle test CSV

    Returns:
        DataFrame with test data
    """
    try:
        test_path = Path(test_path) if test_path else RAW_TEST_PATH

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        df = pd.read_csv(test_path)
        logging.info(f"Loaded Kaggle test data: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error loading Kaggle test data: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    from titanic_ml.utils.logger import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("Running data loader standalone")
    loader = DataLoader()
    train_path, test_path = loader.load_data()
    logger.info(f"Train data: {train_path}")
    logger.info(f"Test data: {test_path}")
