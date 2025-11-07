"""
Utility helper functions for the Titanic ML project.
Includes model training, evaluation, serialization, and Kaggle submission utilities.
"""

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from titanic_ml.utils.exception import CustomException

warnings.filterwarnings("ignore")


def save_object(file_path: Union[str, Path], obj: Any) -> None:
    """
    Saves a given object to a specified file path using dill.

    Args:
        file_path: The path where the object will be saved.
        obj: The object to be saved.
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)


def load_object(file_path: Union[str, Path]) -> Any:
    """
    Loads an object from a specified file path.

    Args:
        file_path: The path from where the object will be loaded.

    Returns:
        The loaded object.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            loaded_obj = dill.load(file_obj)

        logging.info(f"Object loaded from {file_path}")
        return loaded_obj

    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)


def optimize_models(
    models: Dict[str, Any],
    params: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, List]:
    """
    Optimize multiple models using GridSearchCV with Stratified K-Fold CV.

    Args:
        models: Dictionary of model names to model instances
        params: Dictionary of model names to parameter grids
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        cv_folds: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping model names to [best_model, best_score]
    """
    tuned_models = {}

    print("\n------------ Start of model tuning ------------\n")

    # Use Stratified K-Fold for class imbalance
    stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for model_name, model in models.items():

        print(f"{model_name}:")

        f1_scorer = make_scorer(f1_score, average="weighted")

        # Setup GridSearchCV for current model with stratified CV
        tuned_model = GridSearchCV(
            model,
            param_grid=params[model_name],
            cv=stratified_cv,
            scoring=f1_scorer,
            verbose=0,  # Suppress GridSearchCV verbosity
            n_jobs=-1,
        )

        # Fit data
        tuned_model.fit(X_train, y_train)

        # Save best models and best scores
        best_tuned_model = tuned_model.best_estimator_
        tuned_models[model_name] = [best_tuned_model, tuned_model.best_score_]

        # Report model scores and best parameters
        print("- Best Parameters: " + str(tuned_model.best_params_))
        print("- Best F1 Score Train: " + str(round(tuned_model.best_score_ * 100, 1)) + "%")

        # Calculate F1 score on test set if available
        if X_test is not None and y_test is not None:
            y_pred_test = best_tuned_model.predict(X_test)
            f1_test_score = f1_score(y_test, y_pred_test, average="weighted")
            print("- Best F1 Score Test: " + str(round(float(f1_test_score) * 100, 1)) + "%")
            tuned_models[model_name][1] = f1_test_score
        print()

    # Return sorted dictionary
    sorted_tuned_models = dict(sorted(tuned_models.items(), key=lambda item: -item[1][1]))
    return sorted_tuned_models


def evaluate_models_cv(
    models: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    cv_folds: int = 5,
    scoring: str = "accuracy",
    stratified: bool = True,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate multiple models using cross-validation with Stratified K-Fold.

    Args:
        models: Dictionary of model names to model instances
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric to use
        stratified: Whether to use stratified K-fold
        random_state: Random seed

    Returns:
        Dictionary mapping model names to their mean CV scores
    """
    evaluation_report = {}

    cv_strategy = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if stratified
        else cv_folds
    )

    for model_name, model in models.items():
        X_input = (
            X_train.values
            if isinstance(X_train, pd.DataFrame) and model_name in ["KNeighbors Classifier"]
            else X_train
        )

        cv_scores = cross_val_score(
            model, X_input, y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1
        )

        evaluation_report[model_name] = float(cv_scores.mean())

        # Print results
        cv_rounded = [f"{float(score) * 100:.1f}%" for score in cv_scores]
        cv_mean = f"{float(cv_scores.mean()) * 100:.1f}%"

        print(
            f"{model_name}:\n- CV {scoring} scores: {' | '.join(cv_rounded)}\n- CV mean: {cv_mean}\n"
        )

    return evaluation_report


def comprehensive_model_evaluation(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of a trained model.

    Args:
        model: Trained model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)

    Returns:
        Dictionary containing various evaluation metrics
    """
    results = {}

    # Train predictions
    y_train_pred = model.predict(X_train)
    results["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
    results["train_f1"] = float(f1_score(y_train, y_train_pred, average="weighted"))

    print("==================== Train Set Results ====================")
    print(f"Accuracy: {results['train_accuracy']:.4f}")
    print(f"F1 Score: {results['train_f1']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_train_pred))

    # Test predictions if test data provided
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        results["test_accuracy"] = float(accuracy_score(y_test, y_test_pred))
        results["test_f1"] = float(f1_score(y_test, y_test_pred, average="weighted"))

        print("\n==================== Test Set Results ====================")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"F1 Score: {results['test_f1']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))

        if hasattr(model, "predict_proba"):
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
                results["test_roc_auc"] = float(roc_auc_score(y_test, y_test_proba))
                print(f"\nROC-AUC Score: {results['test_roc_auc']:.4f}")
            except Exception:
                pass

    return results


def save_model_pickle(model: Any, file_name: str, directory: Union[str, Path] = "models") -> None:
    """
    Save model to pickle file.

    Args:
        model: Model object to save
        file_name: Name of the file (including .pkl extension)
        directory: Directory to save the model in
    """
    dir_path = Path(directory)
    dir_path.mkdir(exist_ok=True, parents=True)

    file_path = dir_path / file_name

    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"Model saved to {file_path}")
    print(f'File "{file_name}" saved to <./{directory}>')


def load_model_pickle(file_name: str, directory: Union[str, Path] = "models") -> Any:
    """
    Load model from pickle file.

    Args:
        file_name: Name of the file (including .pkl extension)
        directory: Directory containing the model

    Returns:
        Loaded model object
    """
    file_path = Path(directory) / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    logging.info(f"Model loaded from {file_path}")
    print(f'File "{file_name}" loaded')

    return model


def generate_kaggle_submission(
    predictions: np.ndarray,
    passenger_ids: pd.Series,
    file_name: str,
    output_dir: Union[str, Path] = "submissions",
) -> None:
    """
    Generate Kaggle submission CSV file.

    Args:
        predictions: Array of predictions
        passenger_ids: Series of passenger IDs
        file_name: Output file name
        output_dir: Output directory
    """
    submissions_dict = {"PassengerId": passenger_ids, "Survived": predictions.astype(int)}
    submissions_df = pd.DataFrame(data=submissions_dict)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    file_path = output_path / file_name
    submissions_df.to_csv(file_path, index=False)

    print(f'File "{file_name}" saved to <./{output_dir}>')


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    np.random.seed(seed)
    random.seed(seed)

    logging.info(f"Random seeds set to {seed}")


def get_cv_scores_detailed(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv_folds: int = 5,
    scoring: str = "accuracy",
    stratified: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Get detailed cross-validation scores including individual fold results.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv_folds: Number of CV folds
        scoring: Scoring metric
        stratified: Whether to use stratified K-fold
        random_state: Random seed

    Returns:
        Dictionary with detailed CV results
    """
    cv_strategy = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if stratified
        else cv_folds
    )

    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
