# Logging Enhancements

**Status:** Complete

Project-wide structured logging configuration for local development and cloud deployments.

## Quick Start

```python
from src.logger import get_logger, configure_logging

configure_logging()
logger = get_logger(__name__)
logger.info("Application started")
```

## Configuration

The logger auto-detects the deployment environment:

```python
configure_logging()
configure_logging(level="INFO", use_json=True)
configure_logging(level="DEBUG", use_json=False)
```

## Environment Auto-Detection

The logger checks for cloud hosting indicators:

| Environment | Detection Variable |
|---|---|
| Azure App Service | `WEBSITE_INSTANCE_ID` |
| Azure Functions | `AZURE_FUNCTIONS_ENVIRONMENT` |
| AWS Lambda | `AWS_EXECUTION_ENV` |
| Kubernetes | `KUBERNETES_SERVICE_HOST` |

## Structured Logging

Add queryable custom fields:

```python
logger.info(
    "Model training complete",
    extra={'extra_fields': {
        'model_name': 'XGBClassifier',
        'f1_score': 0.843
    }}
)
```

## Logging Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational events
- **WARNING**: Something unexpected but not critical
- **ERROR**: Serious problem; application continues
- **CRITICAL**: Application failure

## Modules Enhanced

| Module | Purpose |
|---|---|
| src/data/loader.py | Data loading logging |
| src/data/transformer.py | Transformation logging |
| src/features/build_features.py | Feature engineering logging |
| src/models/train.py | Training metrics with structured data |
| src/models/predict.py | Prediction tracking with metrics |
| src/app/routes.py | API request/response logging |

## Best Practices

1. Use structured logging for metrics (avoid free-form text)
2. Log at appropriate levels (INFO for operations, DEBUG for diagnostics)
3. Include context (model names, feature counts, data shapes)
4. Avoid logging sensitive data (PII, credentials, passwords)
5. Log summaries not entire objects

---

For Azure-specific configuration, see [Azure Integration](azure-logging.md).
