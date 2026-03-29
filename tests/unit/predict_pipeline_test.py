"""Tests for the prediction pipeline module."""

import pandas as pd
import pytest

from titanic_ml.models.predict import CustomData, PredictPipeline


class TestCustomData:
    """Tests for CustomData input handling."""

    def test_get_data_as_data_frame(self):
        custom_data = CustomData(
            age=42,
            sex="female",
            name_title="Mrs",
            sibsp=0,
            pclass="1",
            embarked="C",
            cabin_multiple=1,
        )

        df = custom_data.get_data_as_dataframe()

        # Check that df is a DataFrame and has expected columns
        # Note: get_data_as_dataframe() now includes advanced features
        assert isinstance(df, pd.DataFrame)

        # Check essential columns exist
        essential_columns = [
            "Age",
            "Sex",
            "SibSp",
            "name_title",
            "Pclass",
            "Embarked",
            "cabin_multiple",
            "Parch",
            "norm_fare",
        ]

        for col in essential_columns:
            assert col in df.columns

        assert df.Age.tolist() == [float(custom_data.age)]


@pytest.mark.requires_model
class TestPredictPipeline:
    """Tests for PredictPipeline inference (requires saved model on disk)."""

    def test_predict_success(self):
        # Use CustomData to generate features with proper engineering
        custom_data = CustomData(
            age=42,
            sex="female",
            name_title="Mrs",
            sibsp=0,
            pclass="1",
            embarked="C",
            cabin_multiple=1,
        )

        features = custom_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(features)

        assert prediction[0] == 1
        assert probability[0] > 0.5
