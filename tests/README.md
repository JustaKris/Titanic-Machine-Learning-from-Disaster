# Tests

Comprehensive test suite for the Titanic Survival Prediction project.

## Quick Start

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest -m unit

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                     # Unit tests (fast, isolated)
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_prediction.py
│   └── test_utils.py
├── integration/              # Integration tests (API)
│   └── test_api.py
└── ui/                       # UI tests (Playwright)
    └── home.test.js
```

## Test Types

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual functions and classes:

- **test_data_loader.py** - Data loading, splitting, and file operations
- **test_feature_engineering.py** - Feature creation, transformations, and mappings
- **test_prediction.py** - CustomData class and PredictPipeline
- **test_utils.py** - Helper functions, logging, and exception handling

**Run unit tests:**
```bash
uv run pytest -m unit -v
```

### Integration Tests (`tests/integration/`)

Tests for API endpoints and component integration:

- **test_api.py** - All Flask routes, validation, error handling

Covers:
- Home and prediction pages
- Health check endpoint
- `/api/predict` JSON endpoint
- `/api/explain` SHAP endpoint
- Input validation and error responses

**Run integration tests:**
```bash
uv run pytest -m integration -v
```

### UI Tests (`tests/ui/`)

End-to-end browser tests using Playwright:

- **home.test.js** - Form interactions, predictions, navigation

Covers:
- Form field presence
- Different passenger scenarios
- Multiple passenger classes and embarkation ports
- Input validation
- Result display

**Run UI tests:**
```bash
npx playwright test tests/ui/
```

**Note:** Requires running Flask app:
```bash
# Terminal 1
uv run python -m src.app.routes

# Terminal 2
npx playwright test
```

## Test Fixtures

Shared fixtures in `conftest.py`:

- `sample_train_data` - Sample Titanic training dataset (10 rows)
- `sample_test_data` - Sample Titanic test dataset (5 rows)
- `sample_prediction_features` - Pre-engineered features
- `mock_model` - Mock RandomForest model
- `mock_preprocessor` - Mock StandardScaler
- `flask_app` - Flask app in test mode
- `client` - Flask test client
- `temp_dir` - Temporary directory for files

## Coverage

Current coverage target: **40%+**

Generate coverage report:
```bash
uv run pytest --cov=src --cov-report=html
start reports/coverage/html/index.html  # Windows
open reports/coverage/html/index.html   # macOS
```

## Writing Tests

### Unit Test Example

```python
import pytest

@pytest.mark.unit
def test_fare_inference():
    """Test fare inference for first class."""
    from src.features.build_features import infer_fare_from_class
    
    fare = infer_fare_from_class("1", 1)
    assert fare == pytest.approx(84.15, rel=0.01)
```

### Integration Test Example

```python
import json
import pytest

@pytest.mark.integration
def test_api_predict(client):
    """Test prediction API endpoint."""
    data = {
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
        data=json.dumps(data),
        content_type="application/json",
    )
    
    assert response.status_code == 200
    result = json.loads(response.data)
    assert "prediction" in result
```

### UI Test Example

```javascript
test('should predict survival', async ({ page }) => {
    await page.goto('http://localhost:5000/prediction');
    
    await page.fill('input[name="age"]', '35');
    await page.selectOption('select[name="gender"]', 'female');
    await page.click('input[type="submit"]');
    
    const content = await page.content();
    expect(content.toLowerCase()).toMatch(/surviv/);
});
```

## Test Markers

Use markers to categorize tests:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - API integration tests
- `@pytest.mark.slow` - Time-consuming tests
- `@pytest.mark.requires_model` - Needs pre-trained model

Run specific markers:
```bash
uv run pytest -m unit        # Unit tests only
uv run pytest -m integration # Integration tests only
uv run pytest -m "not slow"  # Skip slow tests
```

## Continuous Integration

Tests run automatically in GitHub Actions on:
- Every push
- Every pull request

CI pipeline runs:
1. Code formatting checks (Black, isort)
2. Linting (flake8)
3. Type checking (mypy)
4. Unit tests
5. Integration tests
6. Coverage reporting

## Documentation

For detailed testing guide, see: [docs/guides/testing.md](../docs/guides/testing.md)

## Contributing

When adding new features:
1. Write unit tests for new functions
2. Add integration tests for new endpoints
3. Update UI tests if forms change
4. Ensure coverage stays above 40%
5. Run all tests before committing:
   ```bash
   uv run pytest -v
   ```
