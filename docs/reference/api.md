# API Reference

Complete API documentation for the Titanic Survival Prediction system.

---

## Overview

The project provides both a **web interface** and **programmatic API** for making predictions.

### Web Interface

- **Home Page**: `http://localhost:5000/`
- **Prediction Form**: `http://localhost:5000/prediction`
- **Health Check**: `http://localhost:5000/health`

### Python API

```python
from src.models.predict import PredictPipeline, CustomData

# Create input
passenger = CustomData(age=25, sex='female', pclass='1', sibsp=0, parch=0, embarked='S', name_title='Miss', cabin_multiple=0)

# Make prediction
pipeline = PredictPipeline()
predictions, probabilities = pipeline.predict(passenger.get_data_as_dataframe())
```

---

## Flask Routes

### GET /

Home page with project information.

**Response**: HTML page

---

### GET/POST /prediction

Prediction form and API endpoint.

**GET**: Returns prediction form  
**POST**: Makes prediction

**Form Parameters**:

- `age` (float): Passenger age
- `gender` (string): 'male' or 'female'
- `pclass` (string): '1', '2', or '3'
- `sibsp` (int): Number of siblings/spouses
- `parch` (int): Number of parents/children
- `embarked` (string): 'C', 'Q', or 'S'
- `name_title` (string): Title from name
- `cabin_multiple` (int): Number of cabins

**Response**: HTML with prediction result

---

### GET /health

Health check endpoint for monitoring.

**Response**:

```json
{
  "status": "healthy",
  "service": "titanic-survival-predictor-api"
}
```

---

## Python Modules

The following modules are auto-documented using mkdocstrings.

### Prediction Pipeline

::: src.models.predict
    options:
      show_root_heading: true
      show_source: false
      members:
        - PredictPipeline
        - CustomData

### Model Training

::: src.models.train
    options:
      show_root_heading: true
      show_source: false
      members:
        - ModelTrainer

### Data Loading

::: src.data.loader
    options:
      show_root_heading: true
      show_source: false
      members:
        - DataLoader

### Data Transformation

::: src.data.transformer
    options:
      show_root_heading: true
      show_source: false
      members:
        - DataTransformer

### Feature Engineering

::: src.features.build_features
    options:
      show_root_heading: true
      show_source: false
      members:
        - apply_feature_engineering

### Utilities

::: src.utils.helpers
    options:
      show_root_heading: true
      show_source: false
