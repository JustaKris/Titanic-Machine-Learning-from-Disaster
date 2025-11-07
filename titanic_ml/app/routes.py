"""
Flask application routes and views.
"""

from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_swagger_ui import get_swaggerui_blueprint
from pydantic import ValidationError

from titanic_ml.config.settings import (
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    STATIC_DIR,
    TEMPLATES_DIR,
)
from titanic_ml.features.build_features import infer_fare_from_class
from titanic_ml.models.predict import CustomData, PredictPipeline
from titanic_ml.models.schemas import (
    ErrorResponse,
    ExplanationRequest,
    ExplanationResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from titanic_ml.utils.logger import configure_logging, get_logger

# Initialize Flask app
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# Configure Azure-ready logging
configure_logging(level="INFO")
logger = get_logger(__name__)

# Swagger UI configuration
SWAGGER_URL = "/api/docs"
API_URL = "/static/openapi.yaml"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "Titanic Survival Prediction API",
        "docExpansion": "list",
        "defaultModelsExpandDepth": 2,
        "deepLinking": True,
    },
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route("/")
def index():
    """Home page route."""
    logger.info(f"Index page accessed from {request.remote_addr}")
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def predict_datapoint():
    """Prediction page route - handles form submission and prediction."""
    if request.method == "GET":
        logger.info(f"Prediction page accessed from {request.remote_addr}")
        return render_template("home.html")

    try:
        # Extract form data with proper None handling
        data = CustomData(
            age=float(request.form.get("age", 0)),
            sex=request.form.get("gender", "").lower(),
            name_title=request.form.get("name_title", ""),
            sibsp=int(request.form.get("sibsp", 0)),
            pclass=str(request.form.get("pclass", "3")),
            embarked=request.form.get("embarked", "S"),
            cabin_multiple=int(request.form.get("cabin_multiple", 0)),
            parch=int(request.form.get("parch", 0)),
        )

        # Convert to dataframe
        pred_df = data.get_data_as_dataframe()
        input_data = pred_df.to_dict("records")[0]

        # Log prediction request with structured data
        logger.info(
            "Prediction request received",
            extra={"extra_fields": {"user_ip": request.remote_addr, "input_data": input_data}},
        )

        # Make prediction
        predict_pipeline = PredictPipeline()
        prediction, probability = predict_pipeline.predict(pred_df)

        # Format results
        # probability[0] is the probability of survival (class 1)
        # For did not survive (class 0), we need 1 - probability
        survived = bool(prediction[0] == 1)
        prob_survival = float(probability[0])

        if survived:
            proba_pct = round(prob_survival * 100, 1)
            results = f"✅ Survived with {proba_pct}% confidence"
        else:
            proba_pct = round((1 - prob_survival) * 100, 1)
            results = f"❌ Did not survive with {proba_pct}% confidence"

        # Log prediction result with structured data
        logger.info(
            f"Prediction completed: {'Survived' if survived else 'Did not survive'}",
            extra={
                "extra_fields": {
                    "prediction": int(prediction[0]),
                    "probability": prob_survival,
                    "confidence_pct": proba_pct,
                    "user_ip": request.remote_addr,
                }
            },
        )

        return render_template("home.html", results=results)

    except Exception as e:
        logger.error(
            f"Prediction error: {str(e)}",
            exc_info=True,
            extra={
                "extra_fields": {"user_ip": request.remote_addr, "form_data": dict(request.form)}
            },
        )
        error_msg = "⚠️ An error occurred during prediction. Please check your inputs and try again."
        return render_template("home.html", results=error_msg)


@app.route("/health")
def health_check():
    """Health check endpoint with model status."""
    try:
        model_exists = MODEL_PATH.exists()
        preprocessor_exists = PREPROCESSOR_PATH.exists()

        response = HealthResponse(
            status="healthy" if (model_exists and preprocessor_exists) else "unhealthy",
            model_loaded=model_exists,
            preprocessor_loaded=preprocessor_exists,
            version="1.0.0",
        )

        logger.debug(f"Health check from {request.remote_addr}: {response.status}")
        return jsonify(response.model_dump()), 200

    except Exception as e:
        logger.error(f"Health check error: {e}")
        error_response = ErrorResponse(
            error="HealthCheckError",
            message="Failed to check application health",
            details={"exception": str(e)},
        )
        return jsonify(error_response.model_dump()), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for predictions with Pydantic validation."""
    try:
        # Validate request using Pydantic
        req_data = PredictionRequest(**request.json)  # type: ignore

        logger.info(
            "API prediction request received",
            extra={
                "extra_fields": {
                    "user_ip": request.remote_addr,
                    "input_data": req_data.model_dump(),
                }
            },
        )

        # Create CustomData instance
        data = CustomData(
            age=req_data.age,
            sex=req_data.sex,
            name_title=req_data.name_title,
            sibsp=req_data.sibsp,
            pclass=str(req_data.pclass),
            embarked=req_data.embarked,
            cabin_multiple=req_data.cabin_multiple,
            parch=req_data.parch,
        )

        # Make prediction
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction, probability = predict_pipeline.predict(pred_df)

        # Calculate family size and inferred fare
        family_size = req_data.sibsp + req_data.parch + 1
        inferred_fare = infer_fare_from_class(str(req_data.pclass), family_size)

        # Get prediction and calculate confidence
        # probability[0] is the probability of survival (class 1)
        survived = int(prediction[0])
        prob_survival = float(probability[0])

        # For confidence, use the probability of the predicted class
        if survived == 1:
            prob_val = prob_survival  # Confidence in survival
        else:
            prob_val = 1 - prob_survival  # Confidence in non-survival

        # Determine confidence level based on how certain the model is
        if prob_val >= 0.7:
            confidence = "high"
        elif prob_val >= 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        # Create response
        message = (
            f"Passenger {'likely survived' if survived else 'likely did not survive'} "
            f"with {prob_val * 100:.1f}% confidence"
        )

        response = PredictionResponse(
            prediction=survived,  # type: ignore
            probability=prob_val,
            confidence=confidence,  # type: ignore
            inferred_fare=round(inferred_fare, 2),
            family_size=family_size,
            message=message,
        )

        logger.info(
            f"API prediction completed: {message}",
            extra={
                "extra_fields": {
                    "prediction": survived,
                    "probability": prob_val,
                    "user_ip": request.remote_addr,
                }
            },
        )

        return jsonify(response.model_dump()), 200

    except ValidationError as e:
        logger.warning(f"Validation error in API request: {e}")
        error_response = ErrorResponse(
            error="ValidationError", message="Invalid input data", details={"errors": e.errors()}
        )
        return jsonify(error_response.model_dump()), 400

    except Exception as e:
        logger.error(f"API prediction error: {e}", exc_info=True)
        error_response = ErrorResponse(
            error="PredictionError",
            message="An error occurred during prediction",
            details={"exception": str(e)},
        )
        return jsonify(error_response.model_dump()), 500


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """SHAP explainability endpoint with waterfall plot."""
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        from titanic_ml.utils.helpers import load_object

        # Validate request
        req_data = ExplanationRequest(**request.json)  # type: ignore

        logger.info(
            "SHAP explanation request received",
            extra={
                "extra_fields": {"user_ip": request.remote_addr, "plot_type": req_data.plot_type}
            },
        )

        # Create CustomData and get features
        data = CustomData(
            age=req_data.age,
            sex=req_data.sex,
            name_title=req_data.name_title,
            sibsp=req_data.sibsp,
            pclass=str(req_data.pclass),
            embarked=req_data.embarked,
            cabin_multiple=req_data.cabin_multiple,
            parch=req_data.parch,
        )

        pred_df = data.get_data_as_dataframe()

        # Load model and preprocessor
        model = load_object(MODEL_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)

        # Transform features
        features_transformed = preprocessor.transform(pred_df)

        # Make prediction
        prediction = model.predict(features_transformed)[0]
        probability = model.predict_proba(features_transformed)[0]

        # Generate SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(features_transformed)

        # Create plot
        plt.figure(figsize=(10, 6))

        if req_data.plot_type == "waterfall":
            shap.plots.waterfall(shap_values[0], show=False)
        elif req_data.plot_type == "force":
            shap.plots.force(shap_values[0], show=False, matplotlib=True)
        else:  # bar plot
            shap.plots.bar(shap_values[0], show=False)

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"shap_explanation_{timestamp}.png"
        plot_path = STATIC_DIR / "images" / plot_filename
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Get top contributing features
        feature_names = preprocessor.get_feature_names_out()
        shap_vals = shap_values.values[0]

        # Get top 5 absolute SHAP values
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[-5:][::-1]

        top_features = {feature_names[idx]: float(shap_vals[idx]) for idx in top_indices}

        # Handle expected_value (can be scalar or array)
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        elif not isinstance(base_val, (int, float, np.number)):
            base_val = 0.0

        response = ExplanationResponse(
            prediction=int(prediction),  # type: ignore
            probability=float(probability[1]),
            explanation_image_url=f"/static/images/{plot_filename}",
            top_features=top_features,
            base_value=float(base_val),
        )

        logger.info(
            f"SHAP explanation generated: {plot_filename}",
            extra={"extra_fields": {"user_ip": request.remote_addr, "prediction": int(prediction)}},
        )

        return jsonify(response.model_dump()), 200

    except ValidationError as e:
        logger.warning(f"Validation error in explain request: {e}")
        error_response = ErrorResponse(
            error="ValidationError", message="Invalid input data", details={"errors": e.errors()}
        )
        return jsonify(error_response.model_dump()), 400

    except ImportError as e:
        logger.error(f"SHAP not installed: {e}")
        error_response = ErrorResponse(
            error="DependencyError",
            message="SHAP library not installed. Install with: pip install shap",
            details={"exception": str(e)},
        )
        return jsonify(error_response.model_dump()), 500

    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        error_response = ErrorResponse(
            error="ExplanationError",
            message="An error occurred generating explanation",
            details={"exception": str(e)},
        )
        return jsonify(error_response.model_dump()), 500


def create_app():
    """Application factory pattern."""
    return app


def main():
    """Main entry point for CLI command."""
    logger.info(f"Starting Flask application on {FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Debug mode: {FLASK_DEBUG}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)


if __name__ == "__main__":
    main()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
