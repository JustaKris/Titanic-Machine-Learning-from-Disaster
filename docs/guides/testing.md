# Testing Guide

Comprehensive guide for running tests, code quality checks, and CI/CD validation in the Titanic Survival Prediction project.

---

## Quick Command Reference

```bash
# Run all tests
uv run pytest

# Run specific test types
uv run pytest -m unit              # Unit tests only
uv run pytest -m integration       # Integration tests only

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run code quality checks
uv run black --check titanic-ml/       # Formatter check
uv run isort --check-only titanic-ml/  # Import sorter check
uv run flake8 titanic-ml/              # Linting
uv run mypy titanic-ml/                # Type checking
```

---

## Test Suite Overview

The project includes three types of tests:

1. **Unit Tests** (`tests/unit/`) - Fast, isolated tests for individual functions and classes
2. **Integration Tests** (`tests/integration/`) - API endpoint and component integration tests
3. **UI Tests** (`tests/ui/`) - End-to-end browser tests using Playwright

### Test Coverage

Current test modules:

- **Data Loading** (`test_data_loader.py`) - Tests for data ingestion and splitting
- **Feature Engineering** (`test_feature_engineering.py`) - Tests for feature creation and transformations
- **Prediction Pipeline** (`test_prediction.py`) - Tests for CustomData and PredictPipeline classes
- **Utilities** (`test_utils.py`) - Tests for helpers, logging, and exception handling
- **API Endpoints** (`test_api.py`) - Integration tests for all Flask routes
- **UI** (`home.test.js`) - Browser-based end-to-end tests

---

## Testing with pytest

### Prerequisites

Ensure development dependencies are installed:

```bash
uv sync --group dev
```

### Run All Tests

```bash
# Run all tests with verbose output
uv run pytest -v

# Run only unit tests (fast)
uv run pytest -m unit -v

# Run only integration tests
uv run pytest -m integration -v

# Run specific test file
uv run pytest tests/unit/test_prediction.py -v

# Run specific test class
uv run pytest tests/unit/test_feature_engineering.py::TestApplyFeatureEngineering -v

# Run specific test function
uv run pytest tests/unit/test_prediction.py::TestCustomData::test_initialization -v

# Run tests matching a pattern
uv run pytest -k "test_predict" -v

# Run tests and stop at first failure
uv run pytest -x

# Run tests with detailed output
uv run pytest -vv --tb=short
```

### Coverage Reports

```bash
# Generate coverage report in terminal
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html:reports/coverage/html

# Generate XML coverage report (for CI)
uv run pytest --cov=src --cov-report=xml:reports/coverage/coverage.xml

# View HTML report
start reports/coverage/html/index.html  # Windows
open reports/coverage/html/index.html   # macOS
```

### Test Markers

Tests are organized with markers for selective execution:

```bash
# Unit tests (fast)
uv run pytest -m unit

# Integration tests (API endpoints)
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"

# Run only specific markers
uv run pytest -m "unit and not slow"
```

**Available markers:**
- `@pytest.mark.unit` - Fast unit tests (< 1s each)
- `@pytest.mark.integration` - API integration tests (Flask app)
- `@pytest.mark.slow` - Time-consuming tests (model training, etc.)
- `@pytest.mark.requires_model` - Tests requiring pre-trained model files

### Test Fixtures

Shared test fixtures are defined in `tests/conftest.py`:

- `sample_train_data` - Sample Titanic training dataset
- `sample_test_data` - Sample Titanic test dataset
- `sample_prediction_features` - Pre-engineered features for testing
- `sample_custom_data_dict` - Dictionary for CustomData initialization
- `sample_api_request` - JSON payload for API testing
- `mock_model` - Mock trained RandomForest model
- `mock_preprocessor` - Mock StandardScaler preprocessor
- `flask_app` - Flask application in test mode
- `client` - Flask test client for HTTP requests
- `temp_dir` - Temporary directory for file operations

**Using fixtures in tests:**

```python
@pytest.mark.unit
def test_data_loader(sample_train_data, temp_dir):
    """Test uses sample data and temp directory."""
    train_path = temp_dir / "train.csv"
    sample_train_data.to_csv(train_path, index=False)
    # ... test code
```

### Pytest Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests for individual functions",
    "integration: Integration tests for API endpoints",
    "slow: Tests that take longer to run",
]
addopts = [
    "--verbose",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-fail-under=40",
]
```

---

## UI Testing with Playwright

### Prerequisites

Install Playwright and browsers:

```bash
# Install Playwright
npm install -D @playwright/test

# Install browsers
npx playwright install
```

### Running UI Tests

```bash
# Run all UI tests
npx playwright test tests/ui/

# Run with UI mode (interactive)
npx playwright test --ui

# Run in headed mode (see browser)
npx playwright test --headed

# Run specific test file
npx playwright test tests/ui/home.test.js

# Run tests in specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox  
npx playwright test --project=webkit

# Generate HTML report
npx playwright test --reporter=html
npx playwright show-report
```

### UI Test Coverage

The UI tests (`tests/ui/home.test.js`) cover:

- ✅ Page loading and form field presence
- ✅ Prediction for different passenger profiles
- ✅ Different passenger classes (1st, 2nd, 3rd)
- ✅ Different embarkation ports (C, Q, S)
- ✅ Input validation
- ✅ Confidence level display
- ✅ Multiple form submissions

**Important:** UI tests require the Flask app to be running:

```bash
# Terminal 1: Start the app
uv run python -m src.app.routes

# Terminal 2: Run UI tests
npx playwright test
```

---

## Test Structure and Organization

```
tests/
├── conftest.py                 # Shared fixtures
├── __init__.py
│
├── unit/                        # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_data_loader.py     # Data loading tests
│   ├── test_feature_engineering.py  # Feature tests
│   ├── test_prediction.py      # Prediction pipeline tests
│   └── test_utils.py           # Utility function tests
│
├── integration/                 # Integration tests
│   ├── __init__.py
│   └── test_api.py             # API endpoint tests
│
└── ui/                          # UI/E2E tests
    └── home.test.js            # Playwright browser tests
```

### Writing Unit Tests

```python
import pytest
from src.features.build_features import infer_fare_from_class

@pytest.mark.unit
class TestFareInference:
    """Test fare inference logic."""
    
    def test_first_class_solo(self):
        """Test fare for first class solo traveler."""
        fare = infer_fare_from_class("1", 1)
        assert fare == pytest.approx(84.15, rel=0.01)
    
    def test_family_discount(self):
        """Test family discount is applied."""
        solo_fare = infer_fare_from_class("1", 1)
        family_fare = infer_fare_from_class("1", 3)
        assert family_fare == pytest.approx(solo_fare * 0.9, rel=0.01)
```

### Writing Integration Tests

```python
import pytest
import json

@pytest.mark.integration
class TestAPIPredictRoute:
    """Test /api/predict endpoint."""
    
    def test_api_predict_success(self, client):
        """Test successful prediction via API."""
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
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "prediction" in data
        assert 0 <= data["probability"] <= 1
```

### Using Test Fixtures

```python
@pytest.mark.unit
def test_with_fixtures(sample_train_data, temp_dir):
    """Test using shared fixtures."""
    # sample_train_data is a pre-loaded DataFrame
    assert len(sample_train_data) > 0
    
    # temp_dir is a Path to temporary directory
    test_file = temp_dir / "output.csv"
    sample_train_data.to_csv(test_file, index=False)
    assert test_file.exists()
```

### Mocking in Tests

```python
from unittest.mock import patch, MagicMock

@pytest.mark.integration
@patch("src.app.routes.PredictPipeline")
def test_with_mock(mock_pipeline, client):
    """Test with mocked prediction pipeline."""
    # Configure mock
    mock_instance = MagicMock()
    mock_instance.predict.return_value = ([1], [0.85])
    mock_pipeline.return_value = mock_instance
    
    # Make request
    response = client.post(
        "/prediction",
        data={"age": "30", "gender": "female", ...}
    )
    
    # Verify
    assert response.status_code == 200
    assert mock_instance.predict.called
```

---

## Code Quality Checks

### Black - Code Formatter

Ensures consistent code style across the project:

```bash
# Check formatting (no changes)
uv run black --check titanic-ml/

# Apply formatting
uv run black titanic-ml/

# Check specific directory
uv run black --check titanic-ml/models/
```

**Configuration** (`pyproject.toml`):
- Line length: 100 characters
- Target Python versions: 3.11, 3.12, 3.13
- Preview mode: Enabled for modern features

---

### isort - Import Sorter

Organizes imports in a consistent manner:

```bash
# Check import sorting (no changes)
uv run isort --check-only titanic-ml/

# Sort imports
uv run isort titanic-ml/

# Check specific directory
uv run isort --check-only titanic-ml/models/
```

**Configuration** (`pyproject.toml`):
- Profile: black (compatible with Black formatter)
- Line length: 100 characters
- First-party imports: `titanic-ml/*`

---

### flake8 - Linter

Checks code quality, PEP 8 compliance, and naming conventions:

```bash
# Run linting with statistics
uv run flake8 titanic-ml/

# Detailed output with source
uv run flake8 titanic-ml/ --count --statistics --show-source

# Check specific directory
uv run flake8 titanic-ml/models/
```

**Configuration** (`pyproject.toml`):
- Max line length: 100 characters
- Max complexity: 15 (reasonable for ML pipelines)
- Selected checks: E, W, F, C, N (Error, Warning, Flake, Complexity, Naming)

**Ignored errors:**
- E203: Whitespace before ':'
- E266: Too many leading '#' for block comment
- E501: Line too long (handled by Black)
- W503/W504: Line break operators (deprecated)
- F401: Imported but unused (checked per-file)

---

### mypy - Type Checker

Validates type hints and catches type-related bugs:

```bash
# Run type checking
uv run mypy titanic-ml/

# With detailed output
uv run mypy titanic-ml/ --show-error-codes --show-error-context

# Check specific file
uv run mypy titanic-ml/models/predict.py
```

**Configuration** (`pyproject.toml`):
- Python version: 3.11
- Lenient settings for gradual type adoption
- Ignore missing type stubs: `ignore_missing_imports = true`

!!! info "Expected mypy errors"
    Some Pydantic Field overloads generate false positives. These are acceptable and don't affect runtime behavior.

---

## Running All Quality Checks

### Sequential Execution

```bash
# Run all checks one after another
uv run black titanic-ml/ && uv run isort titanic-ml/ && uv run flake8 titanic-ml/ && uv run mypy titanic-ml/

# On Windows PowerShell
uv run black titanic-ml/ ; uv run isort titanic-ml/ ; uv run flake8 titanic-ml/ ; uv run mypy titanic-ml/
```

### Quality Check Script

Create a shell script (`.scripts/quality.sh` or `.scripts/quality.ps1`) for convenience:

```bash
#!/bin/bash
echo "Running code quality checks..."
echo "================================"

echo -e "\n1. Black (formatter)..."
uv run black --check titanic-ml/ || exit 1

echo -e "\n2. isort (imports)..."
uv run isort --check-only titanic-ml/ || exit 1

echo -e "\n3. flake8 (linting)..."
uv run flake8 titanic-ml/ || exit 1

echo -e "\n4. mypy (type checking)..."
uv run mypy titanic-ml/ || true  # Allow mypy to fail

echo -e "\n✓ All quality checks passed!"
```

---

## Continuous Integration (CI/CD)

### GitHub Actions Workflow

The project uses GitHub Actions to automatically run checks on every push and pull request. Configuration is in `.github/workflows/ci.yml`:

**Test Jobs:**
- Python 3.11, 3.12, 3.13
- Unit and integration tests
- Coverage reporting

**Quality Jobs:**
- Black formatting check
- isort import sorting check
- flake8 linting
- mypy type checking

### Local CI Simulation

To test locally before pushing:

```bash
# Run tests (as CI would)
uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=40

# Run all formatters/linters
uv run black --check titanic-ml/
uv run isort --check-only titanic-ml/
uv run flake8 titanic-ml/ --count --statistics
uv run mypy titanic-ml/ --ignore-missing-imports
```

---

## Writing Tests

### Test Structure

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   └── test_predict_pipeline.py
└── integration/
    ├── __init__.py
    └── test_api.py
```

### Unit Test Example

```python
import pytest
from src.models.predict import PredictPipeline


@pytest.mark.unit
def test_predict_pipeline_initialization():
    """Test that PredictPipeline initializes correctly."""
    pipeline = PredictPipeline()
    assert pipeline is not None
    assert hasattr(pipeline, 'predict')


@pytest.mark.unit
def test_predict_output_format():
    """Test prediction output format."""
    pipeline = PredictPipeline()
    # Assertions would go here
    assert True  # Placeholder
```

### Integration Test Example

```python
import pytest
from flask.testing import FlaskClient


@pytest.mark.integration
def test_health_endpoint(client: FlaskClient):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200


@pytest.mark.integration
def test_predict_endpoint_validation():
    """Test prediction endpoint validates input."""
    # Test invalid input handling
    assert True  # Placeholder
```

### Test Fixtures

Define reusable fixtures in `tests/conftest.py`:

```python
import pytest
from flask import Flask


@pytest.fixture
def client():
    """Create Flask test client."""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
```

---

## Coverage Goals

- **Target**: 40%+ overall coverage (configurable in `pyproject.toml`)
- **Focus**: Core logic in `titanic-ml/models/`, `titanic-ml/data/`, and `titanic-ml/features/`
- **Exclude**: ML model internals, visualization code, Flask templates

To check coverage:

```bash
# Terminal report
uv run pytest --cov=src --cov-report=term-missing

# HTML report for detailed view
uv run pytest --cov=src --cov-report=html
open reports/coverage/html/index.html  # macOS
start reports/coverage/html/index.html  # Windows
```

### Current Test Coverage

**Unit Tests:**
- ✅ Data loading and CSV operations
- ✅ Feature engineering and transformations
- ✅ Fare inference logic
- ✅ Title extraction and grouping
- ✅ Age group categorization
- ✅ Cabin feature extraction
- ✅ CustomData class and DataFrame conversion
- ✅ PredictPipeline initialization
- ✅ Utility functions (save/load objects)
- ✅ Logger configuration
- ✅ Custom exception handling

**Integration Tests:**
- ✅ Home page route (`/`)
- ✅ Prediction page route (`/prediction`)
- ✅ Health check endpoint (`/health`)
- ✅ API prediction endpoint (`/api/predict`)
- ✅ API explanation endpoint (`/api/explain`)
- ✅ Input validation
- ✅ Error handling
- ✅ Response format validation

**UI Tests:**
- ✅ Form field presence
- ✅ Different passenger scenarios
- ✅ Navigation and page loading
- ✅ Prediction result display

---

## Running Complete Test Suite

### Full Test Run

```bash
# Run all Python tests with coverage
uv run pytest -v --cov=src --cov-report=html --cov-report=term

# Run all UI tests (app must be running)
npx playwright test

# Run code quality checks
uv run black --check titanic-ml/
uv run isort --check-only titanic-ml/
uv run flake8 titanic-ml/
uv run mypy titanic-ml/
```

### Pre-Commit Checklist

Before committing code, run:

```bash
# 1. Format code
uv run black titanic-ml/
uv run isort titanic-ml/

# 2. Run unit and integration tests
uv run pytest -m "unit or integration" -v

# 3. Check linting
uv run flake8 titanic-ml/

# 4. Check types
uv run mypy titanic-ml/

# 5. Verify coverage
uv run pytest --cov=src --cov-fail-under=40
```

### CI/CD Pipeline Simulation

Simulate what runs in GitHub Actions:

```bash
# Install dependencies
uv sync --all-groups

# Run formatters in check mode
uv run black --check titanic-ml/
uv run isort --check-only titanic-ml/

# Run linters
uv run flake8 titanic-ml/ --count --statistics

# Run type checker
uv run mypy titanic-ml/

# Run tests with coverage
uv run pytest -v --cov=src --cov-report=xml --cov-fail-under=40

# Check for security vulnerabilities
uv run pip-audit
```

---

## Troubleshooting

### Tests Won't Run

```bash
# Ensure dev dependencies are installed
uv sync --group dev

# Verify pytest is available
uv run pytest --version
```

### Import Errors in Tests

```bash
# Reinstall dependencies
uv sync --all-groups

# Check Python path
uv run python -c "import sys; print(sys.path)"
```

### Coverage Report Not Generated

```bash
# Install coverage explicitly
uv run pip install coverage pytest-cov

# Generate with explicit output
uv run pytest --cov=src --cov-report=html:reports/coverage/html --cov-report=term
```

### Slow Tests

```bash
# Skip slow tests during development
uv run pytest -m "not slow"

# Run only fast tests
uv run pytest -m "unit"

# Show slowest tests
uv run pytest --durations=10
```

---

## Best Practices

1. **Write tests as you code** - Test-driven development catches bugs early
2. **Use meaningful test names** - Names should describe what is being tested
3. **Keep tests isolated** - Use fixtures for setup/teardown
4. **Test edge cases** - Test boundary conditions and error handling
5. **Maintain test coverage** - Aim for high coverage on critical paths
6. **Run tests locally before pushing** - Catch failures early

---

## Related Documentation

- **Installation**: [Installation Guide](installation.md) - Set up the project
- **Quick Start**: [Quick Start Guide](quickstart.md) - Run the application
- **Development**: [Architecture Guide](architecture.md) - Understand the codebase
- **Configuration**: [Configuration Guide](configuration.md) - Customize settings

---

## Need Help?

- Review existing tests in `tests/` directory
- Check pytest documentation: [pytest.org](https://pytest.org)
- Open an issue on [GitHub](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/issues)
- Email: k.s.bonev@gmail.com
