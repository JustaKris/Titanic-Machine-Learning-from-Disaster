"""
Pytest configuration and shared fixtures.
"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from titanic_ml.app.routes import create_app


@pytest.fixture(scope="session")
def sample_train_data() -> pd.DataFrame:
    """Create sample training data for testing."""
    return pd.DataFrame(
        {
            "PassengerId": range(1, 11),
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
                "Allen, Mr. William Henry",
                "Moran, Mr. James",
                "McCarthy, Mr. Timothy J",
                "Palsson, Master. Gosta Leonard",
                "Johnson, Mrs. Oscar W",
                "Nasser, Mrs. Nicholas",
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
            ],
            "Age": [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14],
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            "Ticket": [
                "A/5 21171",
                "PC 17599",
                "STON/O2. 3101282",
                "113803",
                "373450",
                "330877",
                "17463",
                "349909",
                "347742",
                "237736",
            ],
            "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07],
            "Cabin": [np.nan, "C85", np.nan, "C123", np.nan, np.nan, "E46", np.nan, np.nan, np.nan],
            "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"],
        }
    )


@pytest.fixture(scope="session")
def sample_test_data() -> pd.DataFrame:
    """Create sample test data for testing."""
    return pd.DataFrame(
        {
            "PassengerId": range(1, 6),
            "Survived": [0, 1, 0, 1, 1],
            "Pclass": [3, 1, 3, 2, 1],
            "Name": [
                "Kelly, Mr. James",
                "Wilkes, Mrs. James",
                "Myles, Mr. Thomas Francis",
                "Wirz, Mrs. Albert",
                "Hirvonen, Mrs. Alexander",
            ],
            "Sex": ["male", "female", "male", "female", "female"],
            "Age": [34.5, 47.0, 62.0, 22.0, np.nan],
            "SibSp": [0, 1, 0, 0, 1],
            "Parch": [0, 0, 0, 2, 1],
            "Ticket": ["330911", "363272", "240276", "315154", "3101298"],
            "Fare": [7.83, 52.00, 9.69, 12.35, 12.65],
            "Cabin": [np.nan, "D28", np.nan, np.nan, np.nan],
            "Embarked": ["Q", "C", "Q", "S", "S"],
        }
    )


@pytest.fixture(scope="session")
def sample_prediction_features() -> pd.DataFrame:
    """Create sample features for prediction testing."""
    return pd.DataFrame(
        {
            "Pclass": ["1", "3", "2"],
            "Sex": ["female", "male", "female"],
            "Age": [38.0, 25.0, 30.0],
            "SibSp": [1, 0, 1],
            "Parch": [0, 0, 1],
            "Embarked": ["C", "S", "Q"],
            "cabin_multiple": [1, 0, 0],
            "name_title": ["Mrs", "Mr", "Miss"],
            "norm_fare": [4.27, 2.08, 2.56],
            "family_size": [2, 1, 3],
            "is_alone": [0, 1, 0],
            "title_grouped": ["Mrs", "Mr", "Miss"],
            "age_group": ["adult", "adult", "adult"],
            "age_missing": [0, 0, 0],
            "fare_per_person": [2.13, 2.08, 0.85],
            "fare_group": ["very_high", "medium", "medium"],
            "cabin_known": [1, 0, 0],
            "cabin_deck": ["C", "Unknown", "Unknown"],
            "sex_pclass": ["female_1", "male_3", "female_2"],
            "age_class": [38.0, 75.0, 60.0],
            "ticket_prefix": ["PC", "NONE", "STON"],
            "ticket_has_letters": [1, 0, 1],
        }
    )


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary CSV file."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )
    df.to_csv(csv_file, index=False)
    yield csv_file


@pytest.fixture
def mock_model(tmp_path: Path) -> Path:
    """Create a mock trained model."""
    import dill

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Create dummy training data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(model, f)

    return model_path


@pytest.fixture
def mock_preprocessor(tmp_path: Path) -> Path:
    """Create a mock preprocessor."""
    import dill

    preprocessor = StandardScaler()
    # Fit with dummy data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    preprocessor.fit(X)

    preprocessor_path = tmp_path / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        dill.dump(preprocessor, f)

    return preprocessor_path


@pytest.fixture
def flask_app():
    """Create Flask test application."""
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
        }
    )
    yield app


@pytest.fixture
def client(flask_app):
    """Create Flask test client."""
    return flask_app.test_client()


@pytest.fixture
def runner(flask_app):
    """Create Flask CLI runner."""
    return flask_app.test_cli_runner()


@pytest.fixture(scope="session")
def sample_custom_data_dict() -> dict:
    """Sample input data for CustomData class."""
    return {
        "age": 42.0,
        "sex": "female",
        "name_title": "Mrs",
        "sibsp": 1,
        "pclass": "1",
        "embarked": "C",
        "cabin_multiple": 2,
        "parch": 0,
    }


@pytest.fixture(scope="session")
def sample_api_request() -> dict:
    """Sample API request payload."""
    return {
        "age": 25,
        "sex": "male",
        "name_title": "Mr",
        "sibsp": 0,
        "pclass": 3,
        "embarked": "S",
        "cabin_multiple": 0,
        "parch": 0,
    }


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for testing."""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger state between tests."""
    import logging

    # Clear all handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    yield

    # Clean up after test
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
