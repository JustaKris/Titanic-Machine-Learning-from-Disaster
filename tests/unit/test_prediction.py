"""
Unit tests for prediction pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from titanic_ml.models.predict import CustomData, PredictPipeline
from titanic_ml.utils.exception import CustomException


@pytest.mark.unit
class TestCustomData:
    """Test suite for CustomData class."""

    def test_initialization(self, sample_custom_data_dict):
        """Test CustomData initialization."""
        data = CustomData(**sample_custom_data_dict)

        assert data.age == 42.0
        assert data.sex == "female"
        assert data.name_title == "Mrs"
        assert data.sibsp == 1
        assert data.pclass == "1"
        assert data.embarked == "C"
        assert data.cabin_multiple == 2
        assert data.parch == 0

    def test_get_data_as_dataframe_structure(self, sample_custom_data_dict):
        """Test that get_data_as_dataframe returns correct structure."""
        data = CustomData(**sample_custom_data_dict)
        df = data.get_data_as_dataframe()

        # Check it's a DataFrame with one row
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

        # Check essential columns exist
        essential_cols = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Embarked",
            "cabin_multiple",
            "name_title",
            "norm_fare",
        ]

        for col in essential_cols:
            assert col in df.columns

    def test_get_data_as_dataframe_advanced_features(self, sample_custom_data_dict):
        """Test that advanced features are included."""
        data = CustomData(**sample_custom_data_dict)
        df = data.get_data_as_dataframe()

        # Check advanced features
        advanced_features = [
            "family_size",
            "is_alone",
            "title_grouped",
            "age_group",
            "age_missing",
            "fare_per_person",
            "fare_group",
            "cabin_known",
            "cabin_deck",
            "sex_pclass",
            "age_class",
            "ticket_prefix",
            "ticket_has_letters",
        ]

        for feature in advanced_features:
            assert feature in df.columns

    def test_family_size_calculation(self):
        """Test family size calculation."""
        data = CustomData(
            age=30,
            sex="male",
            name_title="Mr",
            sibsp=2,
            pclass="3",
            embarked="S",
            cabin_multiple=0,
            parch=1,
        )

        df = data.get_data_as_dataframe()

        # family_size = sibsp + parch + 1 = 2 + 1 + 1 = 4
        assert df["family_size"].iloc[0] == 4

    def test_is_alone_solo_traveler(self):
        """Test is_alone flag for solo traveler."""
        data = CustomData(
            age=25,
            sex="male",
            name_title="Mr",
            sibsp=0,
            pclass="3",
            embarked="S",
            cabin_multiple=0,
            parch=0,
        )

        df = data.get_data_as_dataframe()

        # Solo traveler (family_size=1) should have is_alone=1
        assert df["is_alone"].iloc[0] == 1

    def test_is_alone_with_family(self):
        """Test is_alone flag for traveler with family."""
        data = CustomData(
            age=30,
            sex="female",
            name_title="Mrs",
            sibsp=1,
            pclass="2",
            embarked="C",
            cabin_multiple=1,
            parch=0,
        )

        df = data.get_data_as_dataframe()

        # Traveler with family (family_size=2) should have is_alone=0
        assert df["is_alone"].iloc[0] == 0

    def test_fare_inference_first_class(self):
        """Test fare inference for first class passenger."""
        data = CustomData(
            age=35,
            sex="female",
            name_title="Miss",
            sibsp=0,
            pclass="1",
            embarked="C",
            cabin_multiple=1,
            parch=0,
        )

        df = data.get_data_as_dataframe()

        # First class solo should have fare around 84.15
        # norm_fare = log(fare + 1)
        assert df["norm_fare"].iloc[0] > 0

    def test_age_group_child(self):
        """Test age group classification for child."""
        data = CustomData(
            age=10,
            sex="male",
            name_title="Master",
            sibsp=1,
            pclass="3",
            embarked="S",
            cabin_multiple=0,
            parch=1,
        )

        df = data.get_data_as_dataframe()
        assert df["age_group"].iloc[0] == "child"

    def test_age_group_adult(self):
        """Test age group classification for adult."""
        data = CustomData(
            age=30,
            sex="male",
            name_title="Mr",
            sibsp=0,
            pclass="2",
            embarked="S",
            cabin_multiple=0,
            parch=0,
        )

        df = data.get_data_as_dataframe()
        assert df["age_group"].iloc[0] == "adult"

    def test_age_group_senior(self):
        """Test age group classification for senior."""
        data = CustomData(
            age=70,
            sex="male",
            name_title="Mr",
            sibsp=0,
            pclass="3",
            embarked="S",
            cabin_multiple=0,
            parch=0,
        )

        df = data.get_data_as_dataframe()
        assert df["age_group"].iloc[0] == "senior"

    def test_cabin_known_flag(self):
        """Test cabin_known flag."""
        # With cabin
        data_with_cabin = CustomData(
            age=40,
            sex="female",
            name_title="Mrs",
            sibsp=1,
            pclass="1",
            embarked="C",
            cabin_multiple=2,
            parch=0,
        )
        df1 = data_with_cabin.get_data_as_dataframe()
        assert df1["cabin_known"].iloc[0] == 1

        # Without cabin
        data_no_cabin = CustomData(
            age=40,
            sex="male",
            name_title="Mr",
            sibsp=0,
            pclass="3",
            embarked="S",
            cabin_multiple=0,
            parch=0,
        )
        df2 = data_no_cabin.get_data_as_dataframe()
        assert df2["cabin_known"].iloc[0] == 0

    def test_sex_pclass_interaction(self):
        """Test sex_pclass interaction feature."""
        data = CustomData(
            age=25,
            sex="female",
            name_title="Miss",
            sibsp=0,
            pclass="2",
            embarked="S",
            cabin_multiple=0,
            parch=0,
        )

        df = data.get_data_as_dataframe()
        assert df["sex_pclass"].iloc[0] == "female_2"


@pytest.mark.unit
@pytest.mark.requires_model
class TestPredictPipeline:
    """Test suite for PredictPipeline class."""

    def test_initialization_default_paths(self):
        """Test PredictPipeline initialization with default paths."""
        pipeline = PredictPipeline()

        assert pipeline.model_path is not None
        assert pipeline.preprocessor_path is not None

    def test_initialization_custom_paths(self, temp_dir):
        """Test PredictPipeline initialization with custom paths."""
        model_path = temp_dir / "custom_model.pkl"
        preprocessor_path = temp_dir / "custom_preprocessor.pkl"

        pipeline = PredictPipeline(model_path=model_path, preprocessor_path=preprocessor_path)

        assert pipeline.model_path == model_path
        assert pipeline.preprocessor_path == preprocessor_path

    def test_predict_with_mock_model(
        self, mock_model, mock_preprocessor, sample_prediction_features, monkeypatch
    ):
        """Test prediction with mock model and preprocessor."""
        from titanic_ml.config import settings

        # Monkeypatch settings to use mock paths
        monkeypatch.setattr(settings, "MODEL_PATH", mock_model)
        monkeypatch.setattr(settings, "PREPROCESSOR_PATH", mock_preprocessor)

        pipeline = PredictPipeline(model_path=mock_model, preprocessor_path=mock_preprocessor)

        # Use only numeric features for the mock (it was trained on 2D data)
        simple_features = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )

        predictions, probabilities = pipeline.predict(simple_features)

        # Check output types and shapes
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        assert len(predictions) == len(simple_features)
        assert len(probabilities) == len(simple_features)

        # Check predictions are binary
        assert all(p in [0, 1] for p in predictions)

    def test_predict_output_structure(self, mock_model, mock_preprocessor, monkeypatch):
        """Test that predict returns correct structure."""
        from titanic_ml.config import settings

        monkeypatch.setattr(settings, "MODEL_PATH", mock_model)
        monkeypatch.setattr(settings, "PREPROCESSOR_PATH", mock_preprocessor)

        pipeline = PredictPipeline(model_path=mock_model, preprocessor_path=mock_preprocessor)

        features = pd.DataFrame(
            {
                "feature1": [1.0],
                "feature2": [2.0],
            }
        )

        predictions, probabilities = pipeline.predict(features)

        # Should be tuple of two arrays
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        assert predictions.shape[0] == 1
        assert probabilities.shape[0] == 1

    def test_predict_error_handling(self, temp_dir):
        """Test error handling when model file doesn't exist."""
        fake_model = temp_dir / "nonexistent_model.pkl"
        fake_preprocessor = temp_dir / "nonexistent_preprocessor.pkl"

        pipeline = PredictPipeline(model_path=fake_model, preprocessor_path=fake_preprocessor)

        features = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(CustomException):
            pipeline.predict(features)
