# API Reference

Complete API documentation for the Titanic Survival Prediction service.

## Base URL

- **Development**: `http://localhost:8080`
- **Production**: `https://titanic-ml-api.azurewebsites.net` (if deployed)

## Interactive Documentation

Access Swagger UI at: **`/api/docs`**

The interactive documentation allows you to test all API endpoints directly in your browser.

## Authentication

Currently no authentication required. For production deployment, consider adding:
- API keys
- OAuth 2.0
- Rate limiting

## Endpoints

### 1. Health Check

**`GET /health`**

Check API and model health status.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "version": "1.0.0"
}
```

**cURL Example**:
```bash
curl -X GET http://localhost:8080/health
```

---

### 2. Prediction

**`POST /api/predict`**

Get survival prediction with confidence score.

**Request Body**:
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29.0,
  "sibsp": 0,
  "parch": 0,
  "embarked": "S",
  "cabin_multiple": 0,
  "name_title": "Mrs"
}
```

**Parameters**:

| Field | Type | Required | Range/Options | Description |
|-------|------|----------|---------------|-------------|
| `pclass` | int | ✅ | 1, 2, 3 | Passenger class |
| `sex` | string | ✅ | "male", "female" | Gender |
| `age` | float | ✅ | 0.0 - 100.0 | Age in years |
| `sibsp` | int | ✅ | 0 - 10 | Siblings/spouses aboard |
| `parch` | int | ✅ | 0 - 10 | Parents/children aboard |
| `embarked` | string | ✅ | "S", "C", "Q" | Embarkation port |
| `cabin_multiple` | int | ❌ | 0, 1 | Has multiple cabins |
| `name_title` | string | ❌ | "Mr", "Mrs", "Miss", etc. | Title from name |

**Response** (200 OK):
```json
{
  "prediction": 1,
  "probability": 0.87,
  "confidence": "high",
  "inferred_fare": 84.15,
  "family_size": 1,
  "message": "Passenger likely survived with 87% confidence"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | int | 0 = Did not survive, 1 = Survived |
| `probability` | float | Survival probability (0.0-1.0) |
| `confidence` | string | "low", "medium", "high" |
| `inferred_fare` | float | Fare inferred from class + family size |
| `family_size` | int | Total family members (sibsp + parch + 1) |
| `message` | string | Human-readable result |

**Confidence Levels**:
- **High**: probability ≥ 0.7 or ≤ 0.3
- **Medium**: probability ≥ 0.55 or ≤ 0.45
- **Low**: probability between 0.45-0.55

**cURL Example**:
```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "embarked": "S",
    "cabin_multiple": 0,
    "name_title": "Mrs"
  }'
```

**Python Example**:
```python
import requests

response = requests.post(
    'http://localhost:8080/api/predict',
    json={
        'pclass': 1,
        'sex': 'female',
        'age': 29.0,
        'sibsp': 0,
        'parch': 0,
        'embarked': 'S',
        'cabin_multiple': 0,
        'name_title': 'Mrs'
    }
)

result = response.json()
print(f"Survived: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Message: {result['message']}")
```

**Error Responses**:

400 Bad Request:
```json
{
  "error": "ValidationError",
  "message": "Invalid input data",
  "details": {
    "errors": [
      {
        "loc": ["age"],
        "msg": "ensure this value is less than or equal to 100.0",
        "type": "value_error.number.not_le"
      }
    ]
  }
}
```

500 Internal Server Error:
```json
{
  "error": "PredictionError",
  "message": "An error occurred during prediction",
  "details": {
    "exception": "Model file not found"
  }
}
```

---

### 3. SHAP Explainability

**`POST /api/explain`**

Generate SHAP explanation plot for model prediction.

**Request Body**:
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29.0,
  "sibsp": 0,
  "parch": 0,
  "embarked": "S",
  "cabin_multiple": 0,
  "name_title": "Mrs",
  "plot_type": "waterfall"
}
```

**Additional Parameter**:

| Field | Type | Required | Options | Description |
|-------|------|----------|---------|-------------|
| `plot_type` | string | ❌ | "waterfall", "force", "bar" | SHAP plot type |

**Plot Types**:
- **waterfall**: Shows cumulative contribution of each feature
- **force**: Interactive force plot (horizontal bar)
- **bar**: Feature importance bar chart

**Response** (200 OK):
```json
{
  "prediction": 1,
  "probability": 0.87,
  "explanation_image_url": "/static/images/shap_explanation_20240315_143022.png",
  "top_features": {
    "Sex_female": 0.45,
    "Pclass_1": 0.32,
    "Age": -0.12,
    "family_size": 0.08,
    "fare_per_person": 0.05
  },
  "base_value": 0.38
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `explanation_image_url` | string | Path to generated SHAP plot |
| `top_features` | object | Top 5 features with SHAP values |
| `base_value` | float | Model's average prediction |

**SHAP Value Interpretation**:
- **Positive**: Feature increases survival probability
- **Negative**: Feature decreases survival probability
- **Magnitude**: Strength of feature contribution

**cURL Example**:
```bash
curl -X POST http://localhost:8080/api/explain \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "embarked": "S",
    "plot_type": "waterfall"
  }'
```

**Python Example with Image Display**:
```python
import requests
from PIL import Image
from io import BytesIO

# Get explanation
response = requests.post(
    'http://localhost:8080/api/explain',
    json={
        'pclass': 1,
        'sex': 'female',
        'age': 29.0,
        'sibsp': 0,
        'parch': 0,
        'embarked': 'S',
        'plot_type': 'waterfall'
    }
)

result = response.json()

# Download and display image
image_url = result['explanation_image_url']
img_response = requests.get(f'http://localhost:8080{image_url}')
img = Image.open(BytesIO(img_response.content))
img.show()

# Print top features
print("Top Contributing Features:")
for feature, value in result['top_features'].items():
    direction = "increases" if value > 0 else "decreases"
    print(f"  {feature}: {value:.3f} ({direction} survival)")
```

**Error Response** (500 - SHAP not installed):
```json
{
  "error": "DependencyError",
  "message": "SHAP library not installed. Install with: pip install shap",
  "details": {
    "exception": "No module named 'shap'"
  }
}
```

---

## Web UI Endpoints

### Home Page

**`GET /`**

Renders landing page with project information.

### Prediction Form

**`GET /prediction`**

Displays HTML form for manual prediction input.

**`POST /prediction`**

Submits form data (same fields as `/api/predict`) and shows results on page.

---

## Rate Limiting

**Not currently implemented.** For production:

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def api_predict():
    ...
```

---

## CORS

**Not currently enabled.** To enable cross-origin requests:

```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

---

## Monitoring

**Logging**: All requests logged with structured data including:
- Timestamp
- User IP
- Request payload
- Prediction results
- Errors (with stack traces)

**Metrics** (not implemented):
- Request count
- Response times
- Error rates
- Model performance drift

---

## Testing

Test endpoints with pytest:

```bash
pytest tests/unit/predict_pipeline_test.py -v
```

Manual testing:
```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_passenger.json

# Explanation
curl -X POST http://localhost:8080/api/explain \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_passenger.json
```

---

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
- **File**: `/static/openapi.yaml`
- **Swagger UI**: `/api/docs`

Import into Postman, Insomnia, or other API clients for easy testing.
