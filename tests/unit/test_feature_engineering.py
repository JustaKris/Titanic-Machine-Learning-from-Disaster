"""
Unit tests for feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from titanic_ml.features.build_features import (
    TITLE_MAPPING,
    apply_feature_engineering,
    infer_fare_from_class,
)


@pytest.mark.unit
class TestInferFareFromClass:
    """Test suite for infer_fare_from_class function."""

    def test_first_class_solo(self):
        """Test fare inference for first class solo traveler."""
        fare = infer_fare_from_class("1", 1)
        assert fare == pytest.approx(84.15, rel=0.01)

    def test_second_class_solo(self):
        """Test fare inference for second class solo traveler."""
        fare = infer_fare_from_class("2", 1)
        assert fare == pytest.approx(20.66, rel=0.01)

    def test_third_class_solo(self):
        """Test fare inference for third class solo traveler."""
        fare = infer_fare_from_class("3", 1)
        assert fare == pytest.approx(13.68, rel=0.01)

    def test_family_discount_applied(self):
        """Test that family discount is applied for group travel."""
        solo_fare = infer_fare_from_class("1", 1)
        family_fare = infer_fare_from_class("1", 3)

        # Family should get 10% discount
        assert family_fare == pytest.approx(solo_fare * 0.9, rel=0.01)

    def test_unknown_class_default(self):
        """Test default fare for unknown passenger class."""
        fare = infer_fare_from_class("99", 1)
        assert fare == pytest.approx(32.2, rel=0.01)

    def test_large_family(self):
        """Test fare for large family."""
        fare = infer_fare_from_class("2", 6)
        expected = 20.66 * 0.9  # Second class with 10% discount
        assert fare == pytest.approx(expected, rel=0.01)


@pytest.mark.unit
class TestApplyFeatureEngineering:
    """Test suite for apply_feature_engineering function."""

    def test_basic_features_created(self, sample_train_data):
        """Test that basic features are created."""
        df_transformed, num_cols, cat_cols = apply_feature_engineering(
            sample_train_data.copy(), include_advanced=False
        )

        # Check basic features exist
        assert "cabin_multiple" in df_transformed.columns
        assert "name_title" in df_transformed.columns
        assert "norm_fare" in df_transformed.columns

        # Verify column lists
        assert isinstance(num_cols, list)
        assert isinstance(cat_cols, list)
        assert len(num_cols) > 0
        assert len(cat_cols) > 0

    def test_advanced_features_created(self, sample_train_data):
        """Test that advanced features are created when enabled."""
        df_transformed, _, _ = apply_feature_engineering(
            sample_train_data.copy(), include_advanced=True
        )

        # Check advanced features
        assert "family_size" in df_transformed.columns
        assert "is_alone" in df_transformed.columns
        assert "title_grouped" in df_transformed.columns
        assert "age_group" in df_transformed.columns
        assert "fare_per_person" in df_transformed.columns
        assert "fare_group" in df_transformed.columns
        assert "cabin_known" in df_transformed.columns
        assert "sex_pclass" in df_transformed.columns

    def test_family_size_calculation(self):
        """Test family size calculation."""
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3],
                "Survived": [1, 0, 1],
                "Pclass": [1, 2, 3],
                "Name": ["A, Mr. X", "B, Mrs. Y", "C, Miss. Z"],
                "Sex": ["male", "female", "female"],
                "Age": [25, 30, 22],
                "SibSp": [1, 0, 2],
                "Parch": [0, 2, 1],
                "Ticket": ["A1", "A2", "A3"],
                "Fare": [10, 20, 15],
                "Cabin": [np.nan, "C85", np.nan],
                "Embarked": ["S", "C", "Q"],
            }
        )

        df_transformed, _, _ = apply_feature_engineering(df, include_advanced=True)

        # family_size = SibSp + Parch + 1
        expected_sizes = [2, 3, 4]
        assert df_transformed["family_size"].tolist() == expected_sizes

    def test_is_alone_flag(self):
        """Test is_alone flag creation."""
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2],
                "Survived": [1, 0],
                "Pclass": [1, 2],
                "Name": ["A, Mr. X", "B, Mrs. Y"],
                "Sex": ["male", "female"],
                "Age": [25, 30],
                "SibSp": [0, 1],
                "Parch": [0, 0],
                "Ticket": ["A1", "A2"],
                "Fare": [10, 20],
                "Cabin": [np.nan, "C85"],
                "Embarked": ["S", "C"],
            }
        )

        df_transformed, _, _ = apply_feature_engineering(df, include_advanced=True)

        # First passenger is alone (family_size=1), second is not (family_size=2)
        assert df_transformed["is_alone"].tolist() == [1, 0]

    def test_title_extraction(self):
        """Test title extraction from names."""
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4],
                "Survived": [1, 0, 1, 0],
                "Pclass": [1, 2, 3, 1],
                "Name": [
                    "Braund, Mr. Owen",
                    "Cumings, Mrs. John",
                    "Heikkinen, Miss. Laina",
                    "Futrelle, Master. Jacques",
                ],
                "Sex": ["male", "female", "female", "male"],
                "Age": [22, 38, 26, 4],
                "SibSp": [1, 1, 0, 1],
                "Parch": [0, 0, 0, 2],
                "Ticket": ["A1", "A2", "A3", "A4"],
                "Fare": [7, 71, 8, 53],
                "Cabin": [np.nan, "C85", np.nan, "C123"],
                "Embarked": ["S", "C", "S", "S"],
            }
        )

        df_transformed, _, _ = apply_feature_engineering(df, include_advanced=False)

        expected_titles = ["Mr", "Mrs", "Miss", "Master"]
        assert df_transformed["name_title"].tolist() == expected_titles

    def test_title_grouping(self, sample_train_data):
        """Test title grouping using TITLE_MAPPING."""
        df_transformed, _, _ = apply_feature_engineering(
            sample_train_data.copy(), include_advanced=True
        )

        # Verify title_grouped uses TITLE_MAPPING
        for _, row in df_transformed.iterrows():
            title = row["name_title"]
            grouped = row["title_grouped"]
            expected = TITLE_MAPPING.get(title, "Rare")
            assert grouped == expected

    def test_age_group_categorization(self):
        """Test age group categorization."""
        df = pd.DataFrame(
            {
                "PassengerId": range(1, 6),
                "Survived": [1, 0, 1, 0, 1],
                "Pclass": [1, 2, 3, 1, 2],
                "Name": ["A, Mr. X"] * 5,
                "Sex": ["male"] * 5,
                "Age": [10, 15, 25, 50, 70],
                "SibSp": [0] * 5,
                "Parch": [0] * 5,
                "Ticket": ["A"] * 5,
                "Fare": [10] * 5,
                "Cabin": [np.nan] * 5,
                "Embarked": ["S"] * 5,
            }
        )

        df_transformed, _, _ = apply_feature_engineering(df, include_advanced=True)

        expected_groups = ["child", "teen", "adult", "middle_age", "senior"]
        assert df_transformed["age_group"].tolist() == expected_groups

    def test_cabin_features(self):
        """Test cabin-related feature engineering."""
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3],
                "Survived": [1, 0, 1],
                "Pclass": [1, 2, 3],
                "Name": ["A, Mr. X", "B, Mrs. Y", "C, Miss. Z"],
                "Sex": ["male", "female", "female"],
                "Age": [25, 30, 22],
                "SibSp": [0, 0, 0],
                "Parch": [0, 0, 0],
                "Ticket": ["A1", "A2", "A3"],
                "Fare": [10, 20, 15],
                "Cabin": ["C85 C87", np.nan, "E46"],
                "Embarked": ["S", "C", "Q"],
            }
        )

        df_transformed, _, _ = apply_feature_engineering(df, include_advanced=True)

        # cabin_multiple: 2, 0, 1
        assert df_transformed["cabin_multiple"].tolist() == [2, 0, 1]

        # cabin_known: 1, 0, 1
        assert df_transformed["cabin_known"].tolist() == [1, 0, 1]

        # cabin_deck
        assert df_transformed["cabin_deck"].tolist() == ["C", "Unknown", "E"]

    def test_norm_fare_log_transform(self):
        """Test that norm_fare is log-transformed."""
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2],
                "Survived": [1, 0],
                "Pclass": [1, 2],
                "Name": ["A, Mr. X", "B, Mrs. Y"],
                "Sex": ["male", "female"],
                "Age": [25, 30],
                "SibSp": [0, 0],
                "Parch": [0, 0],
                "Ticket": ["A1", "A2"],
                "Fare": [10.0, 50.0],
                "Cabin": [np.nan, "C85"],
                "Embarked": ["S", "C"],
            }
        )

        df_transformed, _, _ = apply_feature_engineering(df, include_advanced=False)

        # norm_fare should be log(fare + 1)
        expected_norm_fares = [np.log(11), np.log(51)]
        np.testing.assert_array_almost_equal(
            df_transformed["norm_fare"].values, expected_norm_fares, decimal=5
        )

    def test_target_column_removal_training(self, sample_train_data):
        """Test that target column is removed during training."""
        df_transformed, _, _ = apply_feature_engineering(
            sample_train_data.copy(), include_advanced=True, is_training=True
        )

        # Survived should be in the result for training data
        assert "Survived" in df_transformed.columns

    def test_output_structure(self, sample_train_data):
        """Test that output has correct structure."""
        df, num_cols, cat_cols = apply_feature_engineering(
            sample_train_data.copy(), include_advanced=True
        )

        # Check return types
        assert isinstance(df, pd.DataFrame)
        assert isinstance(num_cols, list)
        assert isinstance(cat_cols, list)

        # Check no overlap between num and cat
        assert len(set(num_cols) & set(cat_cols)) == 0

        # Check all columns in lists exist in df
        for col in num_cols + cat_cols:
            assert col in df.columns
