"""
Notebook-specific helper functions for the Titanic ML project.
These utilities are optimized for interactive notebook workflows.
"""
import pickle
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    f1_score, 
    make_scorer, 
)


def optimize_models(
    models: Dict[str, Any], 
    params: Dict[str, Any], 
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    cv_folds: int = 5,
    random_state: int = 42
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

    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION - GRIDSEARCH CV")
    print("="*80)
    
    # Use Stratified K-Fold for class imbalance
    stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for model_name, model in models.items():
        
        print(f"\n{model_name}:")
        
        f1_scorer = make_scorer(f1_score, average='weighted')
        
        # Setup GridSearchCV for current model with stratified CV
        tuned_model = GridSearchCV(
            model,
            param_grid=params[model_name],
            cv=stratified_cv,
            scoring=f1_scorer,
            verbose=0,  # Suppress GridSearchCV verbosity
            n_jobs=-1
        )
        
        # Fit data
        tuned_model.fit(X_train, y_train)

        # Save best models and best scores
        best_tuned_model = tuned_model.best_estimator_
        tuned_models[model_name] = [best_tuned_model, tuned_model.best_score_]

        # Report model scores and best parameters
        print(f'- Best Parameters: {tuned_model.best_params_}')
        print(f'- Best F1 Score Train: {round(tuned_model.best_score_ * 100, 2)}%')

        # Calculate F1 score on test set if available
        if X_test is not None and y_test is not None:
            y_pred_test = best_tuned_model.predict(X_test)
            f1_test_score = f1_score(y_test, y_pred_test, average='weighted')
            print(f'- Best F1 Score Test: {round(float(f1_test_score) * 100, 2)}%')
            tuned_models[model_name][1] = f1_test_score

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")

    # Return sorted dictionary
    sorted_tuned_models = dict(sorted(tuned_models.items(), key=lambda item: -item[1][1]))
    return sorted_tuned_models


def evaluate_models_cv(
    models: Dict[str, Any], 
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray],
    cv_folds: int = 5,
    scoring: str = 'accuracy',
    stratified: bool = True,
    random_state: int = 42
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
    
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds, 
        shuffle=True, 
        random_state=random_state
    ) if stratified else cv_folds
    
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION EVALUATION ({scoring.upper()})")
    print("="*80 + "\n")
    
    for model_name, model in models.items():
        X_input = X_train.values if isinstance(X_train, pd.DataFrame) and model_name in ['KNeighbors Classifier'] else X_train
        
        cv_scores = cross_val_score(
            model, 
            X_input, 
            y_train, 
            cv=cv_strategy, 
            scoring=scoring, 
            n_jobs=-1
        )
        
        evaluation_report[model_name] = float(cv_scores.mean())
        
        # Print results
        cv_rounded = ' | '.join([f"{float(score) * 100:.2f}%" for score in cv_scores])
        cv_mean = f"{float(cv_scores.mean()) * 100:.2f}%"
        
        print(f"{model_name}:")
        print(f"  CV {scoring} scores: {cv_rounded}")
        print(f"  CV mean: {cv_mean}\n")
    
    print("="*80 + "\n")
    
    return evaluation_report


def get_cv_scores_detailed(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv_folds: int = 5,
    scoring: str = 'accuracy',
    stratified: bool = True,
    random_state: int = 42
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
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds, 
        shuffle=True, 
        random_state=random_state
    ) if stratified else cv_folds
    
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    
    return {
        'scores': scores,
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'min': float(scores.min()),
        'max': float(scores.max())
    }


def save_model_pickle(model: Any, file_name: str, directory: Union[str, Path] = '../../models') -> None:
    """
    Save model to pickle file.
    
    Args:
        model: Model object to save
        file_name: Name of the file (including .pkl extension)
        directory: Directory to save the model in (default: project root models/)
    """
    dir_path = Path(directory)
    dir_path.mkdir(exist_ok=True, parents=True)
    
    file_path = dir_path / file_name
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Model saved to {file_path}")
    print(f'File "{file_name}" saved to <./{directory}>')


def load_model_pickle(file_name: str, directory: Union[str, Path] = '../../models') -> Any:
    """
    Load model from pickle file.
    
    Args:
        file_name: Name of the file (including .pkl extension)
        directory: Directory containing the model (default: project root models/)
        
    Returns:
        Loaded model object
    """
    file_path = Path(directory) / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    logging.info(f"Model loaded from {file_path}")
    print(f'File "{file_name}" loaded')
    
    return model


def generate_kaggle_submission(
    predictions: np.ndarray,
    passenger_ids: pd.Series,
    file_name: str,
    output_dir: Union[str, Path] = 'submissions'
) -> None:
    """
    Generate Kaggle submission CSV file.
    
    Args:
        predictions: Array of predictions
        passenger_ids: Series of passenger IDs
        file_name: Output file name
        output_dir: Output directory
    """
    submissions_dict = {
        'PassengerId': passenger_ids,
        'Survived': predictions.astype(int)
    }
    submissions_df = pd.DataFrame(data=submissions_dict)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    file_path = output_path / file_name
    submissions_df.to_csv(file_path, index=False)
    
    print(f'File "{file_name}" saved to <./{output_dir}>')
