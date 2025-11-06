"""
Configuration constants for the Titanic Survival Prediction NOTEBOOK.

This module contains all configuration parameters used specifically in the
Jupyter notebook analysis, including paths, model parameters, and reproducibility settings.

For production pipeline configuration, see src/pipeline/ directory.
"""

from pathlib import Path

# ============================================================================
# Project Structure
# ============================================================================
NOTEBOOK_ROOT = Path(__file__).parent.parent  # notebooks/ folder
PROJECT_ROOT = NOTEBOOK_ROOT.parent  # root of the project

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOK_DATA_DIR = NOTEBOOK_ROOT / "data"

# Output directories (relative to notebook)
MODELS_DIR = NOTEBOOK_ROOT / "models"
SUBMISSIONS_DIR = NOTEBOOK_ROOT / "submissions"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# File paths for train/test data
TRAIN_DATA_PATH = NOTEBOOK_DATA_DIR / "train.csv"
TEST_DATA_PATH = NOTEBOOK_DATA_DIR / "test.csv"

# Fallback to raw data if notebook/data doesn't exist
if not TRAIN_DATA_PATH.exists():
    TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
if not TEST_DATA_PATH.exists():
    TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

# ============================================================================
# Reproducibility
# ============================================================================
RANDOM_SEED = 42
"""Random seed for reproducibility across numpy, sklearn, and other libraries."""

# ============================================================================
# Model Training Configuration
# ============================================================================
CV_FOLDS = 5
"""Number of folds for cross-validation."""

TEST_SIZE = 0.2
"""Proportion of dataset to include in the test split."""

N_JOBS = -1
"""Number of parallel jobs for model training (-1 uses all processors)."""

# ============================================================================
# Evaluation Metrics
# ============================================================================
PRIMARY_METRIC = 'accuracy'
"""Primary metric for model evaluation (Kaggle competition metric)."""

SECONDARY_METRIC = 'f1_weighted'
"""Secondary metric for imbalanced class evaluation."""

# ============================================================================
# Feature Engineering
# ============================================================================
NUMERIC_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare']
"""Original numeric features in the dataset."""

CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']
"""Original categorical features used in modeling."""

ENGINEERED_FEATURES = ['cabin_multiple', 'name_title', 'norm_fare']
"""Features created through feature engineering."""

# ============================================================================
# Data Processing
# ============================================================================
AGE_IMPUTE_STRATEGY = 'median'
"""Strategy for imputing missing Age values."""

FARE_IMPUTE_STRATEGY = 'median'
"""Strategy for imputing missing Fare values."""

EMBARKED_IMPUTE_STRATEGY = 'most_frequent'
"""Strategy for imputing missing Embarked values."""

# ============================================================================
# Model Parameters
# ============================================================================
MAX_ITER_LOGISTIC = 2000
"""Maximum iterations for Logistic Regression."""

VERBOSE_TRAINING = 1
"""Verbosity level for model training (0=silent, 1=progress, 2=debug)."""

# Submission file names
SUBMISSION_VOTING_CLF = "01_voting_clf_submission.csv"
SUBMISSION_XGB_TUNED = "02_xgb_tuned_submission.csv"
SUBMISSION_OPTIMIZED = "03_optimised_model_submission.csv"

# ============================================================================
# Visualization
# ============================================================================
FIGURE_DPI = 100
"""DPI for saved figures."""

FIGURE_SIZE = (10, 6)
"""Default figure size for plots."""

PLOT_STYLE = 'seaborn-v0_8-darkgrid'
"""Default matplotlib style for plots."""

COLOR_PALETTE = 'husl'
"""Default seaborn color palette."""
