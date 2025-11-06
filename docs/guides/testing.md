# Testing Guide

Comprehensive guide for running tests, code quality checks, and CI/CD validation in the Titanic Survival Prediction project.

---

## Quick Command Reference

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run code quality checks
uv run black --check src/       # Formatter check
uv run isort --check-only src/  # Import sorter check
uv run flake8 src/              # Linting
uv run mypy src/                # Type checking
```

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

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run specific test file
uv run pytest tests/unit/test_predict_pipeline.py

# Run tests matching a pattern
uv run pytest -k "test_predict" -v
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
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - API integration tests
- `@pytest.mark.slow` - Time-consuming tests
- `@pytest.mark.requires_model` - Tests requiring pre-trained models

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

## Code Quality Checks

### Black - Code Formatter

Ensures consistent code style across the project:

```bash
# Check formatting (no changes)
uv run black --check src/

# Apply formatting
uv run black src/

# Check specific directory
uv run black --check src/models/
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
uv run isort --check-only src/

# Sort imports
uv run isort src/

# Check specific directory
uv run isort --check-only src/models/
```

**Configuration** (`pyproject.toml`):
- Profile: black (compatible with Black formatter)
- Line length: 100 characters
- First-party imports: `src/*`

---

### flake8 - Linter

Checks code quality, PEP 8 compliance, and naming conventions:

```bash
# Run linting with statistics
uv run flake8 src/

# Detailed output with source
uv run flake8 src/ --count --statistics --show-source

# Check specific directory
uv run flake8 src/models/
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
uv run mypy src/

# With detailed output
uv run mypy src/ --show-error-codes --show-error-context

# Check specific file
uv run mypy src/models/predict.py
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
uv run black src/ && uv run isort src/ && uv run flake8 src/ && uv run mypy src/

# On Windows PowerShell
uv run black src/ ; uv run isort src/ ; uv run flake8 src/ ; uv run mypy src/
```

### Quality Check Script

Create a shell script (`.scripts/quality.sh` or `.scripts/quality.ps1`) for convenience:

```bash
#!/bin/bash
echo "Running code quality checks..."
echo "================================"

echo -e "\n1. Black (formatter)..."
uv run black --check src/ || exit 1

echo -e "\n2. isort (imports)..."
uv run isort --check-only src/ || exit 1

echo -e "\n3. flake8 (linting)..."
uv run flake8 src/ || exit 1

echo -e "\n4. mypy (type checking)..."
uv run mypy src/ || true  # Allow mypy to fail

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
uv run black --check src/
uv run isort --check-only src/
uv run flake8 src/ --count --statistics
uv run mypy src/ --ignore-missing-imports
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
- **Focus**: Core logic in `src/models/` and `src/data/`
- **Exclude**: ML model internals, visualization code, Flask templates

To check coverage:

```bash
# Terminal report
uv run pytest --cov=src --cov-report=term-missing

# HTML report for detailed view
uv run pytest --cov=src --cov-report=html
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
