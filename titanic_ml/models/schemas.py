"""
Pydantic models for API request/response validation.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request schema for survival prediction."""

    # Core passenger features
    pclass: Literal[1, 2, 3] = Field(
        ..., description="Passenger class (1=1st, 2=2nd, 3=3rd)", example=1
    )
    sex: Literal["male", "female"] = Field(..., description="Passenger gender", example="female")
    age: float = Field(..., ge=0.0, le=100.0, description="Passenger age in years", example=29.0)
    sibsp: int = Field(..., ge=0, le=10, description="Number of siblings/spouses aboard", example=0)
    parch: int = Field(..., ge=0, le=10, description="Number of parents/children aboard", example=0)
    embarked: Literal["S", "C", "Q"] = Field(
        ...,
        description="Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)",
        example="S",
    )

    # Optional features
    cabin_multiple: int = Field(
        default=0, ge=0, le=1, description="Has multiple cabins (0=No, 1=Yes)", example=0
    )
    name_title: str = Field(
        default="Mr", description="Title extracted from passenger name", example="Mrs"
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

    class Config:
        """Pydantic config."""

        json_schema_extra = {
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


class PredictionResponse(BaseModel):
    """Response schema for survival prediction."""

    prediction: Literal[0, 1] = Field(
        ..., description="Predicted survival (0=Did not survive, 1=Survived)", example=1
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of survival (0.0 to 1.0)", example=0.87
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence level based on probability", example="high"
    )
    inferred_fare: float = Field(
        ..., description="Fare inferred from passenger class and family size", example=84.15
    )
    family_size: int = Field(..., description="Total family size (sibsp + parch + 1)", example=1)
    message: str = Field(
        ...,
        description="Human-readable prediction message",
        example="Passenger likely survived with 87% confidence",
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.87,
                "confidence": "high",
                "inferred_fare": 84.15,
                "family_size": 1,
                "message": "Passenger likely survived with 87% confidence",
            }
        }


class ExplanationRequest(BaseModel):
    """Request schema for SHAP explanation."""

    # Inherits all fields from PredictionRequest
    pclass: Literal[1, 2, 3] = Field(..., example=1)
    sex: Literal["male", "female"] = Field(..., example="female")
    age: float = Field(..., ge=0.0, le=100.0, example=29.0)
    sibsp: int = Field(..., ge=0, le=10, example=0)
    parch: int = Field(..., ge=0, le=10, example=0)
    embarked: Literal["S", "C", "Q"] = Field(..., example="S")
    cabin_multiple: int = Field(default=0, ge=0, le=1, example=0)
    name_title: str = Field(default="Mr", example="Mrs")

    # Additional explanation options
    plot_type: Literal["waterfall", "force", "bar"] = Field(
        default="waterfall", description="Type of SHAP plot to generate", example="waterfall"
    )

    @field_validator("name_title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title."""
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
        return v if v in valid_titles else "Mr"


class ExplanationResponse(BaseModel):
    """Response schema for SHAP explanation."""

    prediction: Literal[0, 1] = Field(..., description="Predicted survival")
    probability: float = Field(..., ge=0.0, le=1.0, description="Survival probability")
    explanation_image_url: str = Field(
        ...,
        description="URL path to generated SHAP explanation plot",
        example="/static/images/shap_explanation.png",
    )
    top_features: Dict[str, float] = Field(
        ...,
        description="Top contributing features with SHAP values",
        example={"Sex_female": 0.45, "Pclass_1": 0.32, "Age": -0.12},
    )
    base_value: float = Field(
        ..., description="SHAP base value (average model output)", example=0.38
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.87,
                "explanation_image_url": "/static/images/shap_explanation.png",
                "top_features": {"Sex_female": 0.45, "Pclass_1": 0.32, "Age": -0.12},
                "base_value": 0.38,
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: Literal["healthy", "unhealthy"] = Field(..., example="healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded in memory")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    version: str = Field(..., description="API version", example="1.0.0")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "preprocessor_loaded": True,
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {"field": "age", "error": "Value must be between 0 and 100"},
            }
        }
