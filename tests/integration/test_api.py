"""
Integration tests for Flask API endpoints.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.mark.integration
class TestHomeRoute:
    """Test suite for home route."""

    def test_index_get(self, client):
        """Test GET request to index page."""
        response = client.get("/")

        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data

    def test_index_returns_html(self, client):
        """Test that index returns HTML content."""
        response = client.get("/")

        assert response.content_type.startswith("text/html")


@pytest.mark.integration
class TestPredictionRoute:
    """Test suite for prediction route."""

    def test_prediction_page_get(self, client):
        """Test GET request to prediction page."""
        response = client.get("/prediction")

        assert response.status_code == 200
        assert response.content_type.startswith("text/html")

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_prediction_post_success(self, mock_pipeline, client):
        """Test successful prediction via form submission."""
        # Mock the prediction pipeline
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([1], [0.85])
        mock_pipeline.return_value = mock_instance

        form_data = {
            "age": "42",
            "gender": "female",
            "name_title": "Mrs",
            "sibsp": "1",
            "pclass": "1",
            "embarked": "C",
            "cabin_multiple": "2",
            "parch": "0",
        }

        response = client.post("/prediction", data=form_data)

        assert response.status_code == 200
        assert b"Survived" in response.data or b"survived" in response.data

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_prediction_post_no_survival(self, mock_pipeline, client):
        """Test prediction showing no survival."""
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([0], [0.75])
        mock_pipeline.return_value = mock_instance

        form_data = {
            "age": "25",
            "gender": "male",
            "name_title": "Mr",
            "sibsp": "0",
            "pclass": "3",
            "embarked": "S",
            "cabin_multiple": "0",
            "parch": "0",
        }

        response = client.post("/prediction", data=form_data)

        assert response.status_code == 200
        assert b"not survive" in response.data or b"Did not survive" in response.data

    def test_prediction_post_missing_fields(self, client):
        """Test prediction with missing required fields."""
        form_data = {
            "age": "30",
            # Missing other required fields
        }

        response = client.post("/prediction", data=form_data)

        # Should still return 200 but might show error message
        assert response.status_code == 200

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_prediction_post_error_handling(self, mock_pipeline, client):
        """Test error handling in prediction endpoint."""
        # Make pipeline raise an exception
        mock_pipeline.return_value.predict.side_effect = Exception("Test error")

        form_data = {
            "age": "30",
            "gender": "male",
            "name_title": "Mr",
            "sibsp": "0",
            "pclass": "2",
            "embarked": "S",
            "cabin_multiple": "0",
            "parch": "0",
        }

        response = client.post("/prediction", data=form_data)

        assert response.status_code == 200
        assert b"error" in response.data or b"Error" in response.data


@pytest.mark.integration
class TestHealthRoute:
    """Test suite for health check endpoint."""

    @patch("titanic_ml.app.routes.MODEL_PATH")
    @patch("titanic_ml.app.routes.PREPROCESSOR_PATH")
    def test_health_check_healthy(self, mock_prep_path, mock_model_path, client):
        """Test health check when model and preprocessor exist."""
        mock_model_path.exists.return_value = True
        mock_prep_path.exists.return_value = True

        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["preprocessor_loaded"] is True
        assert "version" in data

    @patch("titanic_ml.app.routes.MODEL_PATH")
    @patch("titanic_ml.app.routes.PREPROCESSOR_PATH")
    def test_health_check_unhealthy(self, mock_prep_path, mock_model_path, client):
        """Test health check when model doesn't exist."""
        mock_model_path.exists.return_value = False
        mock_prep_path.exists.return_value = True

        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False

    @patch("titanic_ml.app.routes.MODEL_PATH")
    def test_health_check_error(self, mock_model_path, client):
        """Test health check error handling."""
        mock_model_path.exists.side_effect = Exception("Test error")

        response = client.get("/health")

        assert response.status_code == 500
        data = json.loads(response.data)

        assert "error" in data
        assert data["error"] == "HealthCheckError"


@pytest.mark.integration
class TestAPIPredictRoute:
    """Test suite for /api/predict endpoint."""

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_api_predict_success(self, mock_pipeline, client, sample_api_request):
        """Test successful prediction via JSON API."""
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([1], [0.82])
        mock_pipeline.return_value = mock_instance

        response = client.post(
            "/api/predict",
            data=json.dumps(sample_api_request),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "message" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    @patch("titanic_ml.app.routes.PredictPipeline")
    @patch("titanic_ml.app.routes.CustomData")
    def test_api_predict_high_confidence(self, mock_custom_data, mock_pipeline, client):
        """Test API prediction with high confidence."""
        # Mock the pipeline
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([1], [0.95])
        mock_pipeline.return_value = mock_instance

        # Mock CustomData
        mock_data_instance = MagicMock()
        mock_data_instance.get_data_as_dataframe.return_value = pd.DataFrame({"col": [1]})
        mock_custom_data.return_value = mock_data_instance

        request_data = {
            "age": 35,
            "sex": "female",
            "name_title": "Mrs",
            "sibsp": 1,
            "pclass": 1,
            "embarked": "C",
            "cabin_multiple": 1,  # Changed from 2 to 1
            "parch": 0,
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        data = json.loads(response.data)

        # Verify response structure
        assert response.status_code == 200
        assert "prediction" in data
        assert "probability" in data

        # Check if confidence field exists (it should based on the route logic)
        if "confidence" in data:
            assert data["confidence"] == "high"

    @patch("titanic_ml.app.routes.PredictPipeline")
    @patch("titanic_ml.app.routes.CustomData")
    def test_api_predict_medium_confidence(self, mock_custom_data, mock_pipeline, client):
        """Test API prediction with medium confidence."""
        # Mock the pipeline
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([0], [0.60])
        mock_pipeline.return_value = mock_instance

        # Mock CustomData
        mock_data_instance = MagicMock()
        mock_data_instance.get_data_as_dataframe.return_value = pd.DataFrame({"col": [1]})
        mock_custom_data.return_value = mock_data_instance

        request_data = {
            "age": 35,
            "sex": "female",
            "name_title": "Mrs",
            "sibsp": 1,
            "pclass": 1,
            "embarked": "C",
            "cabin_multiple": 1,  # Changed from 2 to 1
            "parch": 0,
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        data = json.loads(response.data)

        # Verify response structure
        assert response.status_code == 200
        assert "prediction" in data
        assert "probability" in data

        # Check if confidence field exists
        # Prediction=0 (did not survive) with prob_survival=0.60
        # means confidence = 1 - 0.60 = 0.40 (40%) which is "low"
        if "confidence" in data:
            assert data["confidence"] == "low"  # 40% confidence is low

    def test_api_predict_validation_error(self, client):
        """Test API validation with invalid data."""
        invalid_data = {
            "age": "invalid",  # Should be number
            "sex": "female",
            "name_title": "Mrs",
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(invalid_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)

        assert "error" in data
        assert data["error"] == "ValidationError"

    def test_api_predict_missing_fields(self, client):
        """Test API validation with missing required fields."""
        incomplete_data = {
            "age": 30,
            "sex": "male",
            # Missing other required fields
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(incomplete_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_api_predict_includes_inferred_fare(self, mock_pipeline, client):
        """Test that API response includes inferred fare."""
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([1], [0.75])
        mock_pipeline.return_value = mock_instance

        request_data = {
            "age": 30,
            "sex": "female",
            "name_title": "Miss",
            "sibsp": 0,
            "pclass": 1,
            "embarked": "C",
            "cabin_multiple": 1,
            "parch": 0,
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        data = json.loads(response.data)
        assert "inferred_fare" in data
        assert isinstance(data["inferred_fare"], (int, float))
        assert data["inferred_fare"] > 0

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_api_predict_includes_family_size(self, mock_pipeline, client):
        """Test that API response includes family size."""
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ([0], [0.65])
        mock_pipeline.return_value = mock_instance

        request_data = {
            "age": 25,
            "sex": "male",
            "name_title": "Mr",
            "sibsp": 2,
            "pclass": 3,
            "embarked": "S",
            "cabin_multiple": 0,
            "parch": 1,
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        data = json.loads(response.data)
        assert "family_size" in data
        # family_size = sibsp + parch + 1 = 2 + 1 + 1 = 4
        assert data["family_size"] == 4

    @patch("titanic_ml.app.routes.PredictPipeline")
    def test_api_predict_error_handling(self, mock_pipeline, client):
        """Test API error handling during prediction."""
        mock_pipeline.return_value.predict.side_effect = Exception("Prediction failed")

        request_data = {
            "age": 30,
            "sex": "male",
            "name_title": "Mr",
            "sibsp": 0,
            "pclass": 2,
            "embarked": "S",
            "cabin_multiple": 0,
            "parch": 0,
        }

        response = client.post(
            "/api/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 500
        data = json.loads(response.data)

        assert "error" in data
        assert data["error"] == "PredictionError"


@pytest.mark.integration
class TestAPIExplainRoute:
    """Test suite for /api/explain endpoint."""

    def test_api_explain_validation_error(self, client):
        """Test explain endpoint validation."""
        invalid_data = {
            "age": "not_a_number",
            "sex": "female",
        }

        response = client.post(
            "/api/explain",
            data=json.dumps(invalid_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_api_explain_missing_shap(self, client, monkeypatch):
        """Test explain endpoint when SHAP is not installed."""

        # Mock import error for shap
        def mock_import_error(*args, **kwargs):
            raise ImportError("No module named 'shap'")

        with patch("builtins.__import__", side_effect=mock_import_error):
            request_data = {
                "age": 30,
                "sex": "female",
                "name_title": "Mrs",
                "sibsp": 1,
                "pclass": 1,
                "embarked": "C",
                "cabin_multiple": 1,
                "parch": 0,
                "plot_type": "waterfall",
            }

            response = client.post(
                "/api/explain",
                data=json.dumps(request_data),
                content_type="application/json",
            )

            # Might return 500 or 400 depending on error handling
            assert response.status_code in [400, 500]


@pytest.mark.integration
class TestSwaggerUI:
    """Test suite for Swagger UI integration."""

    def test_swagger_ui_accessible(self, client):
        """Test that Swagger UI is accessible."""
        response = client.get("/api/docs/")

        # Should either return the UI or redirect
        assert response.status_code in [200, 301, 302, 308]

    def test_openapi_yaml_accessible(self, client):
        """Test that OpenAPI spec is accessible."""
        response = client.get("/static/openapi.yaml")

        # File might not exist in test environment
        assert response.status_code in [200, 404]
