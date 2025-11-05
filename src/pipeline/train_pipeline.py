"""
Production Training Pipeline for Titanic Survival Prediction.

This module implements a scikit-learn Pipeline for production use,
combining preprocessing and model training in a single, reproducible workflow.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.utils import save_model_pickle, load_model_pickle


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Titanic-specific feature engineering.
    
    Creates:
        - cabin_multiple: Number of cabins per passenger
        - name_title: Title extracted from passenger name
        - norm_fare: Log-normalized fare
    """
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        X = X.copy()
        
        # Cabin multiple
        X['cabin_multiple'] = X['Cabin'].apply(
            lambda x: 0 if pd.isna(x) else len(str(x).split(' '))
        )
        
        # Name title
        X['name_title'] = X['Name'].apply(
            lambda x: x.split(',')[1].split('.')[0].strip()
        )
        
        # Normalized fare (log transform with shift to handle zeros)
        X['norm_fare'] = np.log(X['Fare'] + 1)
        
        return X


class TitanicPreprocessor:
    """
    Complete preprocessing pipeline for Titanic dataset.
    
    Handles:
        - Feature engineering
        - Missing value imputation
        - Categorical encoding
        - Numeric scaling
    """
    
    def __init__(self):
        """Initialize preprocessing pipeline."""
        # Define features
        self.numeric_features = ['Age', 'SibSp', 'Parch', 'norm_fare']
        self.categorical_features = ['Pclass', 'Sex', 'Embarked', 'cabin_multiple', 'name_title']
        
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop columns not specified
        )
    
    def get_preprocessor(self) -> ColumnTransformer:
        """Get the preprocessing pipeline."""
        return self.preprocessor


class TitanicTrainingPipeline:
    """
    Complete training pipeline for Titanic survival prediction.
    
    Combines feature engineering, preprocessing, and model training
    into a single scikit-learn Pipeline for production use.
    """
    
    def __init__(
        self,
        model=None,
        random_state: int = 42
    ):
        """
        Initialize training pipeline.
        
        Args:
            model: Scikit-learn compatible model (default: XGBClassifier)
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Default to XGBoost with optimized parameters
        if model is None:
            model = XGBClassifier(
                n_estimators=500,
                colsample_bytree=0.75,
                max_depth=10,
                reg_lambda=10,
                subsample=0.6,
                learning_rate=0.5,
                gamma=0.5,
                min_child_weight=0.01,
                random_state=random_state,
                eval_metric='logloss'
            )
        
        # Build complete pipeline
        self.pipeline = Pipeline(steps=[
            ('feature_engineering', FeatureEngineer()),
            ('preprocessing', TitanicPreprocessor().get_preprocessor()),
            ('model', model)
        ])
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the pipeline and return results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        print("Training pipeline...")
        print(f"Training set size: {len(X_train)} samples")
        
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Cross-validation scores using Stratified K-Fold for class imbalance
        stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.pipeline,
            X_train,
            y_train,
            cv=stratified_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': self.pipeline
        }
        
        print(f"\n✓ Training complete!")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained pipeline.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predictions
        """
        predictions = self.pipeline.predict(X)
        # Ensure we return ndarray, not tuple
        if isinstance(predictions, tuple):
            return predictions[0]
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of prediction probabilities
        """
        return self.pipeline.predict_proba(X)
    
    def save(self, filepath: str = 'titanic_pipeline.pkl', directory: str = 'models') -> None:
        """
        Save trained pipeline to disk.
        
        Args:
            filepath: Name of the file
            directory: Directory to save in
        """
        save_model_pickle(self.pipeline, filepath, directory)
        print(f"Pipeline saved to {directory}/{filepath}")
    
    @classmethod
    def load(cls, filepath: str = 'titanic_pipeline.pkl', directory: str = 'models'):
        """
        Load trained pipeline from disk.
        
        Args:
            filepath: Name of the file
            directory: Directory to load from
            
        Returns:
            TitanicTrainingPipeline instance with loaded pipeline
        """
        instance = cls()
        instance.pipeline = load_model_pickle(filepath, directory)
        print(f"Pipeline loaded from {directory}/{filepath}")
        return instance


def train_and_evaluate_pipeline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    save_model: bool = True
) -> Tuple[TitanicTrainingPipeline, Dict[str, Any]]:
    """
    Train and evaluate the complete Titanic pipeline.
    
    Args:
        train_data: Training dataframe (must include 'Survived' column)
        test_data: Test dataframe for predictions
        save_model: Whether to save the trained pipeline
        
    Returns:
        Tuple of (trained pipeline, results dictionary)
    """
    # Separate features and target
    y_train = train_data['Survived']
    X_train = train_data.drop('Survived', axis=1)
    X_test = test_data.copy()
    
    # Initialize and train pipeline
    pipeline = TitanicTrainingPipeline(random_state=42)
    results = pipeline.train(X_train, y_train, cv_folds=5)
    
    # Generate predictions
    predictions = pipeline.predict(X_test)
    results['test_predictions'] = predictions
    
    # Save if requested
    if save_model:
        pipeline.save()
    
    return pipeline, results


if __name__ == '__main__':
    """Example usage of the training pipeline."""
    
    # Load data
    train = pd.read_csv('notebook/data/train.csv')
    test = pd.read_csv('notebook/data/test.csv')
    
    # Train and evaluate
    pipeline, results = train_and_evaluate_pipeline(train, test, save_model=True)
    
    print("\n" + "="*60)
    print("Pipeline training complete!")
    print("="*60)
    print(f"Cross-validation accuracy: {results['cv_mean']:.4f}")
    print(f"Test predictions generated: {len(results['test_predictions'])} samples")
