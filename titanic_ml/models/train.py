"""
Model training module - handles model selection and training.
"""

import itertools
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from titanic_ml.config.settings import CV_FOLDS, MODEL_PATH, RANDOM_STATE
from titanic_ml.utils.exception import CustomException
from titanic_ml.utils.helpers import optimize_models, save_object
from titanic_ml.utils.logger import logging

warnings.filterwarnings("ignore")


class ModelTrainer:
    """Handles model training and hyperparameter optimization."""

    def __init__(self, model_path: Optional[Path] = None, cv_folds: int = CV_FOLDS):
        """
        Initialize ModelTrainer.

        Args:
            model_path: Path to save trained model
            cv_folds: Number of cross-validation folds
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.cv_folds = cv_folds

    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.

        Returns:
            Dictionary mapping model names to instances
        """
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "Random Forest Classifier": RandomForestClassifier(random_state=RANDOM_STATE),
            "KNeighbors Classifier": KNeighborsClassifier(),
            "SVC": SVC(probability=True, random_state=RANDOM_STATE),
            "XGB Classifier": XGBClassifier(random_state=RANDOM_STATE, verbosity=0),
            "CatBoost Classifier": CatBoostClassifier(random_state=RANDOM_STATE, silent=True),
        }

    def get_param_grids(self) -> Dict[str, Any]:
        """
        Get hyperparameter grids for each model.

        Returns:
            Dictionary mapping model names to parameter grids
        """
        return {
            "Logistic Regression": {
                "max_iter": [10, 50, 100, 1000],
                "penalty": ["l1", "l2"],
                "C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"],
            },
            "Random Forest Classifier": {
                "n_estimators": [100, 300, 450],
                "criterion": ["gini", "entropy"],
                "bootstrap": [True],
                "max_depth": [10, 15],
                "max_features": ["sqrt", 20, 40],
                "min_samples_leaf": [1, 2],
                "min_samples_split": [0.5, 2],
            },
            "KNeighbors Classifier": {
                "n_neighbors": [3, 5, 7, 9, 12, 15],
                "weights": ["uniform", "distance"],
                "algorithm": ["ball_tree", "kd_tree"],
                "p": [1, 2],
            },
            "SVC": [
                {"kernel": ["rbf"], "gamma": [0.1, 0.5, 1, 2, 5, 10], "C": [0.1, 1, 10]},
                {"kernel": ["linear"], "C": [0.1, 1, 10]},
                {"kernel": ["poly"], "degree": [1, 2, 3], "C": [0.1, 1, 10]},
            ],
            "XGB Classifier": {
                "n_estimators": [500, 550],
                "colsample_bytree": [0.5, 0.6, 0.75],
                "max_depth": [10, None],
                "reg_alpha": [1],
                "reg_lambda": [5, 10, 15],
                "subsample": [0.55, 0.6, 0.65],
                "learning_rate": [0.5],
                "gamma": [0.25, 0.5, 1],
                "min_child_weight": [0.01],
                "sampling_method": ["uniform"],
            },
            "CatBoost Classifier": {
                "iterations": [400, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [2, 4, 6],
                "l2_leaf_reg": [1, 2],
                "border_count": [64, 128],
            },
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        use_voting: bool = True,
    ) -> Tuple[Any, float]:
        """
        Train and optimize models, return best performer.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            use_voting: Whether to create voting classifier ensemble

        Returns:
            Tuple of (best_model, best_score)
        """
        try:
            logging.info("Starting model training")
            logging.info(
                f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features"
            )
            if X_test is not None:
                logging.info(f"Test set size: {X_test.shape[0]} samples")
            logging.info(f"Cross-validation folds: {self.cv_folds}")

            # Get models and parameters
            models = self.get_models()
            params = self.get_param_grids()
            logging.info(f"Training {len(models)} models: {list(models.keys())}")

            # Optimize individual models
            model_report = optimize_models(
                models=models,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_folds=self.cv_folds,
                random_state=RANDOM_STATE,
            )
            logging.info("Model optimization complete")

            # Optional: Create voting classifier ensemble
            if use_voting and len(model_report) > 1:
                logging.info(f"Creating voting classifier ensemble from {len(model_report)} models")

                tuned_model_tuples = [(name, model[0]) for name, model in model_report.items()]
                voting_clf = VotingClassifier(estimators=tuned_model_tuples, voting="soft")

                # Generate weight combinations
                n_models = len(tuned_model_tuples)
                combinations = list(itertools.product([1, 2], repeat=n_models))
                combinations = [  # type: ignore[misc]
                    list(comb)
                    for comb in combinations
                    if len(set(comb)) != 1  # Exclude all same weights
                ]

                voting_params = {
                    "Voting Classifier": {"weights": combinations, "voting": ["soft", "hard"]}
                }

                # Optimize voting classifier
                tuned_voting = optimize_models(
                    models={"Voting Classifier": voting_clf},
                    params=voting_params,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    cv_folds=self.cv_folds,
                    random_state=RANDOM_STATE,
                )

                model_report["Voting Classifier"] = tuned_voting["Voting Classifier"]

            # Sort by score
            model_report = dict(sorted(model_report.items(), key=lambda item: -item[1][1]))

            # Get best model
            best_model_name, best_model_tuple = list(model_report.items())[0]
            best_model, best_model_score = best_model_tuple

            # Validate minimum score threshold
            if best_model_score < 0.6:
                raise CustomException(Exception("No model achieved F1 score above 60%"), sys)

            logging.info(
                f"Best model: {best_model_name} with F1 score of "
                f"{round(best_model_score * 100, 1)}%",
                extra={
                    "extra_fields": {
                        "model_name": best_model_name,
                        "f1_score": best_model_score,
                        "num_features": X_train.shape[1],
                        "train_samples": X_train.shape[0],
                    }
                },
            )

            # Save best model
            save_object(self.model_path, best_model)
            logging.info(f"Model saved to {self.model_path}")

            return best_model, best_model_score

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)


def main():
    """Main entry point for CLI training command."""
    from titanic_ml.data.loader import DataLoader
    from titanic_ml.data.transformer import DataTransformer
    from titanic_ml.utils.logger import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)
    logger.info("Running model trainer standalone")

    # Load and transform data
    loader = DataLoader()
    train_path, test_path = loader.load_data()

    transformer = DataTransformer()
    X_train, y_train, X_test, y_test, _ = transformer.transform_data(train_path, test_path)

    # Train models
    trainer = ModelTrainer()
    best_model, score = trainer.train(X_train, y_train, X_test, y_test)

    logger.info(f"Training complete - Best F1 score: {score:.4f}")
    return score


if __name__ == "__main__":
    main()
