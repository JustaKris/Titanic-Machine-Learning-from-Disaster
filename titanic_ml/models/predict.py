"""
Model prediction module - handles inference pipeline.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from titanic_ml.config.settings import MODEL_PATH, PREPROCESSOR_PATH
from titanic_ml.features.build_features import TITLE_MAPPING, infer_fare_from_class
from titanic_ml.utils.exception import CustomException
from titanic_ml.utils.helpers import load_object
from titanic_ml.utils.logger import logging


class PredictPipeline:
    """Handles model predictions on new data."""

    def __init__(self, model_path: Optional[Path] = None, preprocessor_path: Optional[Path] = None):
        """
        Initialize prediction pipeline.

        Args:
            model_path: Path to trained model
            preprocessor_path: Path to preprocessor object
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else PREPROCESSOR_PATH

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input features.

        Args:
            features: DataFrame with input features

        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            logging.info("Starting prediction pipeline")
            logging.info(f"Input shape: {features.shape}")

            # Load model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            model_type = type(model).__name__
            logging.info(f"Loaded model: {model_type}")
            logging.info(f"Loaded preprocessor from {self.preprocessor_path}")

            # Transform features
            data_scaled = preprocessor.transform(features)
            logging.info(f"Transformed features shape: {data_scaled.shape}")

            # Make predictions
            predictions = model.predict(data_scaled)

            # Get probabilities
            # For VotingClassifier with hard voting, manually average individual probabilities
            if hasattr(model, "estimators_") and not hasattr(model, "predict_proba"):
                logging.info(
                    "VotingClassifier with hard voting detected - averaging individual model probabilities"
                )
                individual_probas = np.array(
                    [estimator.predict_proba(data_scaled)[:, 1] for estimator in model.estimators_]
                )
                probabilities = individual_probas.mean(axis=0)
                avg_probability = probabilities.mean()
                logging.info(
                    f"Average prediction probability (from {len(model.estimators_)} models): {avg_probability:.2%}"
                )
            elif hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data_scaled)[:, 1]
                avg_probability = probabilities.mean()
                logging.info(f"Average prediction probability: {avg_probability:.2%}")
            else:
                # Fallback: use predictions as probabilities (0.0 or 1.0)
                probabilities = predictions.astype(float)
                logging.warning(
                    "Model has no predict_proba - using hard predictions as probabilities"
                )

            survival_rate = predictions.mean()
            logging.info(
                f"Made {len(predictions)} predictions - "
                f"Predicted survival rate: {survival_rate:.1%}",
                extra={
                    "extra_fields": {
                        "num_predictions": len(predictions),
                        "survival_rate": float(survival_rate),
                        "model_type": model_type,
                    }
                },
            )

            return predictions, probabilities

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)


class CustomData:
    """
    Represents custom input data for single prediction.
    Used primarily for web app predictions.
    """

    def __init__(
        self,
        age: float,
        sex: str,
        name_title: str,
        sibsp: int,
        pclass: str,
        embarked: str,
        cabin_multiple: int,
        parch: int = 0,
    ):
        """
        Initialize custom data instance.

        Args:
            age: Passenger age
            sex: Passenger gender ('male' or 'female')
            name_title: Title from name (Mr, Mrs, Miss, etc.)
            sibsp: Number of siblings/spouses
            pclass: Passenger class (1, 2, or 3)
            embarked: Port of embarkation (C, Q, or S)
            cabin_multiple: Number of cabins
            parch: Number of parents/children
        """
        self.age = age
        self.sex = sex
        self.name_title = name_title
        self.sibsp = sibsp
        self.pclass = pclass
        self.embarked = embarked
        self.cabin_multiple = cabin_multiple
        self.parch = parch

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert custom data to DataFrame format for prediction.
        Uses class-based fare inference for better accuracy.

        Returns:
            DataFrame with single row of features
        """
        try:
            # Calculate family size
            family_size = self.sibsp + self.parch + 1

            # Infer fare based on passenger class and family size
            inferred_fare = infer_fare_from_class(str(self.pclass), family_size)
            norm_fare = np.log(inferred_fare + 1)

            logging.info(
                f"Inferred fare for Pclass {self.pclass}, family_size {family_size}: "
                f"£{inferred_fare:.2f} (normalized: {norm_fare:.4f})"
            )

            # Map title to grouped category
            title_grouped = TITLE_MAPPING.get(self.name_title, "Rare")

            # Create base features
            custom_data_dict = {
                "Pclass": [str(self.pclass)],
                "Sex": [self.sex],
                "Age": [float(self.age)],
                "SibSp": [int(self.sibsp)],
                "Parch": [int(self.parch)],
                "Embarked": [self.embarked],
                "cabin_multiple": [int(self.cabin_multiple)],
                "name_title": [self.name_title],
                "norm_fare": [norm_fare],
                # Advanced features
                "family_size": [family_size],
                "is_alone": [1 if family_size == 1 else 0],
                "title_grouped": [title_grouped],
                "age_missing": [0],  # User provided age, so not missing
                "fare_per_person": [norm_fare / family_size],
                "cabin_known": [1 if self.cabin_multiple > 0 else 0],
                "cabin_deck": ["Unknown"],  # Default since we don't have full cabin info
                "sex_pclass": [f"{self.sex}_{self.pclass}"],
                "age_class": [float(self.age) * int(self.pclass)],
            }

            # Age group
            if self.age <= 12:
                age_group = "child"
            elif self.age <= 18:
                age_group = "teen"
            elif self.age <= 35:
                age_group = "adult"
            elif self.age <= 60:
                age_group = "middle_age"
            else:
                age_group = "senior"

            custom_data_dict["age_group"] = [age_group]

            # Fare group (approximate quartiles)
            if inferred_fare < 7.9:
                fare_group = "low"
            elif inferred_fare < 14.5:
                fare_group = "medium"
            elif inferred_fare < 31:
                fare_group = "high"
            else:
                fare_group = "very_high"

            custom_data_dict["fare_group"] = [fare_group]

            # Ticket features (defaults since we don't collect ticket info)
            custom_data_dict["ticket_prefix"] = ["NONE"]
            custom_data_dict["ticket_has_letters"] = [0]

            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            logging.error(f"Error creating dataframe from custom data: {e}")
            raise CustomException(e, sys)


def main():
    """Main entry point for CLI prediction command."""
    from titanic_ml.utils.logger import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("Running prediction pipeline standalone")

    # Test prediction pipeline
    custom_data = CustomData(
        age=25,
        sex="female",
        name_title="Miss",
        sibsp=1,
        pclass="3",
        embarked="S",
        cabin_multiple=0,
        parch=0,
    )

    df = custom_data.get_data_as_dataframe()
    logger.info(f"Input data: {df.to_dict('records')[0]}")

    pipeline = PredictPipeline()
    predictions, probabilities = pipeline.predict(df)

    logger.info(f"Prediction: {'Survived' if predictions[0] == 1 else 'Did not survive'}")
    logger.info(f"Probability: {probabilities[0]:.2%}")


if __name__ == "__main__":
    main()
