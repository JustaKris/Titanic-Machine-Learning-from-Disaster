"""
Configuration constants for the Titanic Survival Prediction NOTEBOOK.

This module contains all configuration parameters used specifically in the
Jupyter notebook analysis, including paths, model parameters, and reproducibility settings.

For production pipeline configuration, see src/pipeline/ directory.
"""

import os
from pathlib import Path

# ============================================================================
# Project Structure
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
NOTEBOOK_DATA_DIR = PROJECT_ROOT / "notebook" / "data"

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

VERBOSE_TRAINING = True
"""Whether to print verbose output during model training."""

# ============================================================================
# Visualization
# ============================================================================
FIGURE_DPI = 100
"""DPI for saved figures."""

PLOT_STYLE = 'seaborn-v0_8-darkgrid'
"""Default matplotlib style for plots."""

COLOR_PALETTE = 'husl'
"""Default seaborn color palette."""
