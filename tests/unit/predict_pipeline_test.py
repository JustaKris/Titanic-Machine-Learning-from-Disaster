from unittest import TestCase, main

import pandas as pd

from titanic_ml.models.predict import CustomData, PredictPipeline


class TestCustomData(TestCase):
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
        self.assertIsInstance(df, pd.DataFrame)

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
            self.assertIn(col, df.columns)

        self.assertListEqual(df.Age.tolist(), [float(custom_data.age)])


class TestPredictPipeline(TestCase):
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

        self.assertEqual(prediction[0], 1)
        self.assertGreater(probability[0], 0.5)


if __name__ == "__main__":
    main()
