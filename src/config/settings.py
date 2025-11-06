"""
Configuration settings for the Titanic ML project.
Production-ready configuration with Pydantic Settings for type-safe,
environment-based configuration suitable for cloud deployments.
"""

import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables or a .env file.
    The .env file takes precedence, making it ideal for Azure/cloud deployments.

    Example .env file:
        ENVIRONMENT=production
        LOG_LEVEL=INFO
        FLASK_HOST=0.0.0.0
        FLASK_PORT=5000
        RANDOM_STATE=42
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )

    # ============================================================================
    # Application Settings
    # ============================================================================

    app_name: str = Field(
        default="Titanic Survival Prediction API",
        description="Application name displayed in logs and API docs",
    )

    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # ============================================================================
    # Data Paths
    # ============================================================================

    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Root data directory",
    )

    artifacts_dir: Path = Field(
        default=PROJECT_ROOT / "artifacts",
        description="Processed data artifacts directory",
    )

    models_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        description="Model artifacts directory",
    )

    logs_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Logs directory",
    )

    # ============================================================================
    # Model Configuration
    # ============================================================================

    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )

    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Test set size for train/test split",
    )

    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of cross-validation folds",
    )

    scoring_metric: str = Field(
        default="f1_weighted",
        description="Scoring metric for model evaluation",
    )

    # ============================================================================
    # Feature Engineering
    # ============================================================================

    numerical_features: List[str] = Field(
        default=["Age", "SibSp", "Parch", "norm_fare"],
        description="Numerical feature columns",
    )

    categorical_features: List[str] = Field(
        default=["Pclass", "Sex", "Embarked", "cabin_multiple", "name_title"],
        description="Categorical feature columns",
    )

    target_column: str = Field(
        default="Survived",
        description="Target column name",
    )

    # ============================================================================
    # Flask API Configuration
    # ============================================================================

    flask_host: str = Field(
        default="0.0.0.0",
        description="Flask host address (0.0.0.0 for containers)",
    )

    flask_port: int = Field(
        default=5000,
        ge=1,
        le=65535,
        description="Flask port number",
    )

    flask_debug: bool = Field(
        default=False,
        description="Enable Flask debug mode (disable in production)",
    )

    # ============================================================================
    # Model File Paths (computed)
    # ============================================================================

    @property
    def model_path(self) -> Path:
        """Path to trained model pickle file."""
        return self.models_dir / "model.pkl"

    @property
    def preprocessor_path(self) -> Path:
        """Path to preprocessor pickle file."""
        return self.models_dir / "preprocessor.pkl"

    @property
    def log_file(self) -> Path:
        """Path to log file."""
        return self.logs_dir / "titanic_ml.log"

    # ============================================================================
    # Data File Paths (computed)
    # ============================================================================

    @property
    def raw_train_path(self) -> Path:
        """Path to raw training data."""
        return self.data_dir / "raw" / "train.csv"

    @property
    def raw_test_path(self) -> Path:
        """Path to raw test data."""
        return self.data_dir / "raw" / "test.csv"

    @property
    def processed_train_path(self) -> Path:
        """Path to processed training data."""
        return self.data_dir / "processed" / "train.csv"

    @property
    def processed_test_path(self) -> Path:
        """Path to processed test data."""
        return self.data_dir / "processed" / "test.csv"

    @property
    def train_augmented_path(self) -> Path:
        """Path to augmented training data."""
        return self.artifacts_dir / "train_augmented.csv"

    @property
    def test_augmented_path(self) -> Path:
        """Path to augmented test data."""
        return self.artifacts_dir / "test_augmented.csv"

    # ============================================================================
    # Validators
    # ============================================================================

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard Python logging levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of: {', '.join(valid_levels)}")
        return v_upper

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "external",
            self.artifacts_dir,
            self.models_dir,
            self.logs_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),
            ],
        )

    def log_startup_info(self, logger: logging.Logger) -> None:
        """Log startup configuration information."""
        logger.info("=" * 60)
        logger.info(f"{self.app_name} v{self.app_version}")
        logger.info("=" * 60)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info(f"Flask Host: {self.flask_host}:{self.flask_port}")
        logger.info(f"Debug Mode: {self.flask_debug}")
        logger.info(f"Random State: {self.random_state}")
        logger.info(f"Models Directory: {self.models_dir}")
        logger.info("=" * 60)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create the global settings instance.

    This ensures we only load settings once and reuse the same instance
    throughout the application lifecycle.

    Returns:
        Settings: The global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.setup_directories()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment/file.

    Useful for testing or dynamic configuration changes.

    Returns:
        Settings: The newly loaded settings instance
    """
    global _settings
    _settings = Settings()
    _settings.setup_directories()
    return _settings


# Legacy compatibility - maintain backward compatibility with old code
settings = get_settings()

# Legacy path constants (computed from settings)
DATA_DIR = settings.data_dir
RAW_DATA_DIR = settings.data_dir / "raw"
PROCESSED_DATA_DIR = settings.data_dir / "processed"
EXTERNAL_DATA_DIR = settings.data_dir / "external"

RAW_TRAIN_PATH = settings.raw_train_path
RAW_TEST_PATH = settings.raw_test_path
PROCESSED_TRAIN_PATH = settings.processed_train_path
PROCESSED_TEST_PATH = settings.processed_test_path
TRAIN_AUGMENTED_PATH = settings.train_augmented_path
TEST_AUGMENTED_PATH = settings.test_augmented_path

MODELS_DIR = settings.models_dir
MODEL_PATH = settings.model_path
PREPROCESSOR_PATH = settings.preprocessor_path

LOGS_DIR = settings.logs_dir
LOG_FILE = settings.log_file

STATIC_DIR = PROJECT_ROOT / "src" / "app" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "src" / "app" / "templates"

TARGET_COLUMN = settings.target_column
RANDOM_STATE = settings.random_state
TEST_SIZE = settings.test_size

NUMERICAL_FEATURES = settings.numerical_features
CATEGORICAL_FEATURES = settings.categorical_features

CV_FOLDS = settings.cv_folds
SCORING_METRIC = settings.scoring_metric

FLASK_HOST = settings.flask_host
FLASK_PORT = settings.flask_port
FLASK_DEBUG = settings.flask_debug
