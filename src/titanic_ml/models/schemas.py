"""Pydantic models for API request/response validation."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """Request schema for survival prediction."""

    # Core passenger features
    pclass: Literal[1, 2, 3] = Field(
        ..., description="Passenger class (1=1st, 2=2nd, 3=3rd)", examples=[1]
    )
    sex: Literal["male", "female"] = Field(..., description="Passenger gender", examples=["female"])
    age: float = Field(..., ge=0.0, le=100.0, description="Passenger age in years", examples=[29.0])
    sibsp: int = Field(
        ..., ge=0, le=10, description="Number of siblings/spouses aboard", examples=[0]
    )
    parch: int = Field(
        ..., ge=0, le=10, description="Number of parents/children aboard", examples=[0]
    )
    embarked: Literal["S", "C", "Q"] = Field(
        ...,
        description="Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)",
        examples=["S"],
    )

    # Optional features
    cabin_multiple: int = Field(
        default=0, ge=0, le=1, description="Has multiple cabins (0=No, 1=Yes)", examples=[0]
    )
    name_title: str = Field(
        default="Mr", description="Title extracted from passenger name", examples=["Mrs"]
    )

    @field_validator("name_title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is one of expected values."""
        valid_titles = {
            "Mr",
            "Miss",
            "Mrs",
            "Master",
            "Dr",
            "Rev",
            "Col",
            "Major",
            "Mlle",
            "Countess",
            "Ms",
            "Lady",
            "Jonkheer",
            "Don",
            "Mme",
            "Capt",
            "Sir",
            "Dona",
        }
        if v not in valid_titles:
            return "Mr"  # Default fallback
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "embarked": "S",
                "cabin_multiple": 0,
                "name_title": "Mrs",
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for survival prediction."""

    prediction: Literal[0, 1] = Field(
        ..., description="Predicted survival (0=Did not survive, 1=Survived)", examples=[1]
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of survival (0.0 to 1.0)", examples=[0.87]
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence level based on probability", examples=["high"]
    )
    inferred_fare: float = Field(
        ..., description="Fare inferred from passenger class and family size", examples=[84.15]
    )
    family_size: int = Field(..., description="Total family size (sibsp + parch + 1)", examples=[1])
    message: str = Field(
        ...,
        description="Human-readable prediction message",
        examples=["Passenger likely survived with 87% confidence"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "probability": 0.87,
                "confidence": "high",
                "inferred_fare": 84.15,
                "family_size": 1,
                "message": "Passenger likely survived with 87% confidence",
            }
        }
    )


class ExplanationRequest(PredictionRequest):
    """Request schema for SHAP explanation."""

    # Additional explanation options
    plot_type: Literal["waterfall", "force", "bar"] = Field(
        default="waterfall", description="Type of SHAP plot to generate", examples=["waterfall"]
    )


class ExplanationResponse(BaseModel):
    """Response schema for SHAP explanation."""

    prediction: Literal[0, 1] = Field(..., description="Predicted survival")
    probability: float = Field(..., ge=0.0, le=1.0, description="Survival probability")
    explanation_image_url: str = Field(
        ...,
        description="URL path to generated SHAP explanation plot",
        examples=["/static/images/shap_explanation.png"],
    )
    top_features: Dict[str, float] = Field(
        ...,
        description="Top contributing features with SHAP values",
        examples=[{"Sex_female": 0.45, "Pclass_1": 0.32, "Age": -0.12}],
    )
    base_value: float = Field(
        ..., description="SHAP base value (average model output)", examples=[0.38]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "probability": 0.87,
                "explanation_image_url": "/static/images/shap_explanation.png",
                "top_features": {"Sex_female": 0.45, "Pclass_1": 0.32, "Age": -0.12},
                "base_value": 0.38,
            }
        }
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: Literal["healthy", "unhealthy"] = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether model is loaded in memory")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    version: str = Field(..., description="API version", examples=["1.0.0"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "preprocessor_loaded": True,
                "version": "1.0.0",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {"field": "age", "error": "Value must be between 0 and 100"},
            }
        }
    )
