# Configuration Guide

This project uses **Pydantic Settings** for type-safe, environment-based configuration - the modern standard for Python applications deployed to cloud platforms.

## Quick Start

### 1. Create `.env` File

```bash
cp .env.example .env
```

### 2. Configure Settings

Edit `.env` for local development or set environment variables in Azure:

```env
# Application
ENVIRONMENT="development"
LOG_LEVEL="INFO"

# Flask API
FLASK_HOST="0.0.0.0"
FLASK_PORT=5000
FLASK_DEBUG=false

# Model Training
RANDOM_STATE=42
CV_FOLDS=5
```

### 3. Run the Application

```bash
uv run python app.py
```

Settings automatically load from `.env` or environment variables.

## Configuration Options

### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | "Titanic Survival Prediction API" | Application name |
| `ENVIRONMENT` | `development` | Environment: development/staging/production |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Flask API

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Host address (0.0.0.0 for containers) |
| `FLASK_PORT` | `5000` | Port number |
| `FLASK_DEBUG` | `false` | Enable debug mode (disable in production) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOM_STATE` | `42` | Random seed for reproducibility |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `SCORING_METRIC` | `f1_weighted` | Model evaluation metric |

## Using Configuration in Code

### Type-Safe Access

```python
from src.config.settings import get_settings

settings = get_settings()

# Type-safe access
print(settings.flask_host)      # str
print(settings.random_state)    # int
print(settings.flask_debug)     # bool

# Computed paths
model_path = settings.model_path
data_dir = settings.data_dir
```

### Legacy Constants

For backward compatibility, legacy constants are still available:

```python
from src.config.settings import MODEL_PATH, FLASK_HOST, RANDOM_STATE
```

## Cloud Deployment

### Azure App Service

Configure via Azure Portal or CLI:

```bash
az webapp config appsettings set \
  --name myapp --resource-group mygroup \
  --settings LOG_LEVEL="INFO" ENVIRONMENT="production"
```

### Docker

Using .env file:

```bash
docker run --env-file .env -p 5000:5000 titanic-ml
```

Using environment variables:

```bash
docker run -e LOG_LEVEL="INFO" -e FLASK_DEBUG="false" -p 5000:5000 titanic-ml
```

## Validation

Pydantic automatically validates configuration at startup:

```python
# Invalid log level
LOG_LEVEL="INVALID"
# ❌ Error: Invalid log level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Invalid port number
FLASK_PORT="not_a_number"
# ❌ Error: Input should be a valid integer
```

## Benefits

- ✅ **Type-safe** - Automatic validation and type checking
- ✅ **Environment-aware** - Different configs for dev/staging/prod
- ✅ **Cloud-friendly** - Works with Azure, AWS, GCP
- ✅ **Fail-fast** - Invalid configs caught at startup
- ✅ **Self-documenting** - Type hints and field descriptions

See `.env.example` for all available options.
