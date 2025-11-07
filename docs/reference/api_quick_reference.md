# API Quick Reference

Quick reference for the Titanic ML API endpoints.

## Base URL
```
http://localhost:8080
```

## Endpoints

### 🏥 Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "version": "1.0.0"
}
```

---

### 🎯 Prediction
```bash
POST /api/predict
Content-Type: application/json
```

**Request**:
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

**Response**:
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

---

### 📊 SHAP Explanation
```bash
POST /api/explain
Content-Type: application/json
```

**Request**:
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29.0,
  "sibsp": 0,
  "parch": 0,
  "embarked": "S",
  "plot_type": "waterfall"
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.87,
  "explanation_image_url": "/static/images/shap_explanation_20240315_143022.png",
  "top_features": {
    "Sex_female": 0.45,
    "Pclass_1": 0.32,
    "Age": -0.12
  },
  "base_value": 0.38
}
```

---

### 📚 API Documentation
```
GET /api/docs
```

Opens Swagger UI with interactive documentation.

---

## Field Reference

### Required Fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `pclass` | int | 1, 2, 3 | Passenger class |
| `sex` | string | "male", "female" | Gender |
| `age` | float | 0.0-100.0 | Age in years |
| `sibsp` | int | 0-10 | Siblings/spouses |
| `parch` | int | 0-10 | Parents/children |
| `embarked` | string | "S", "C", "Q" | Embarkation port |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cabin_multiple` | int | 0 | Multiple cabins (0/1) |
| `name_title` | string | "Mr" | Passenger title |
| `plot_type` | string | "waterfall" | SHAP plot type |

---

## Embarkation Codes

- **S**: Southampton (most common)
- **C**: Cherbourg
- **Q**: Queenstown

---

## Plot Types (SHAP)

- **waterfall**: Cumulative feature contributions
- **force**: Horizontal bar (red=positive, blue=negative)
- **bar**: Feature importance ranking

---

## Confidence Levels

- **high**: probability ≥ 0.7 or ≤ 0.3
- **medium**: probability ≥ 0.55 or ≤ 0.45
- **low**: probability 0.45-0.55

---

## Example Passengers

### High Survival (>80%)
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29,
  "sibsp": 0,
  "parch": 0,
  "embarked": "S"
}
```

### Low Survival (<20%)
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "embarked": "S"
}
```

### Uncertain (40-60%)
```json
{
  "pclass": 2,
  "sex": "male",
  "age": 35,
  "sibsp": 1,
  "parch": 2,
  "embarked": "C"
}
```

---

## Python Client

```python
import requests

# Prediction
response = requests.post(
    'http://localhost:8080/api/predict',
    json={
        'pclass': 1,
        'sex': 'female',
        'age': 29.0,
        'sibsp': 0,
        'parch': 0,
        'embarked': 'S'
    }
)
print(response.json())

# Explanation
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
print(f"Image URL: {result['explanation_image_url']}")
```

---

## Error Codes

- **200**: Success
- **400**: Invalid request (validation error)
- **500**: Server error (model/SHAP issue)

---

## See Also

- Full documentation: `docs/guides/api.md`
- OpenAPI spec: `static/openapi.yaml`
- Swagger UI: http://localhost:8080/api/docs
