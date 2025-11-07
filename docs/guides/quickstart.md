# Quick Start Guide

Get the Titanic Survival Prediction system running locally in under 5 minutes.

## Prerequisites

- **Python 3.11 or higher** ([download](https://www.python.org/downloads/))
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Git** ([download](https://git-scm.com/downloads))

!!! tip "Why uv?"
    `uv` is a blazing-fast Python package manager that replaces pip. It's 10-100x faster and handles dependencies more reliably. [Learn more](https://docs.astral.sh/uv/)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster
```

### 2. Install Dependencies

```bash
# Install all production dependencies
uv sync

# Or install with optional groups
uv sync --group dev        # Include development tools
uv sync --group notebooks  # Include Jupyter support
uv sync --all-groups       # Include everything
```

!!! info "First-time setup"
    First run will download ~500MB of ML models and dependencies. Subsequent runs are instant.

### 3. Verify Installation

```bash
# Check Python environment
uv run python --version

# Verify key packages
uv run python -c "import pandas, sklearn, xgboost, catboost; print('✓ All packages imported successfully')"
```

---

## Running the Application

### Option 1: Web Interface (Recommended for Demo)

```bash
python app.py
```

Then open your browser to **http://localhost:5000**

You'll see a form to enter passenger details:
- Age
- Gender
- Passenger Class (1st, 2nd, 3rd)
- Number of siblings/spouses
- Number of parents/children
- Port of embarkation
- And more...

Click **"Predict Survival"** to get instant predictions with confidence scores!

### Option 2: Jupyter Notebook (Recommended for Analysis)

```bash
# Install notebook dependencies (if not already installed)
uv sync --group notebooks

# Launch JupyterLab
uv run jupyter lab notebooks/Titanic-Machine-Learning-from-Disaster.ipynb
```

The notebook contains:
- Complete exploratory data analysis (EDA)
- Feature engineering walkthrough
- Model comparison and tuning
- SHAP explainability analysis
- Visualizations and insights

### Option 3: Python API (Recommended for Integration)

```python
from src.models.predict import PredictPipeline, CustomData

# Create input data
passenger = CustomData(
    age=25,
    sex='female',
    name_title='Miss',
    sibsp=1,
    pclass='1',
    embarked='C',
    cabin_multiple=0,
    parch=0
)

# Make prediction
pipeline = PredictPipeline()
predictions, probabilities = pipeline.predict(passenger.get_data_as_dataframe())

print(f"Survived: {bool(predictions[0])}")
print(f"Probability: {probabilities[0]:.2%}")
```

---

## Training Your Own Model

### Run the Full Training Pipeline

```bash
# Option 1: Using the scripts
python scripts/run_training.py

# Option 2: Step by step
python titanic_ml/data/loader.py           # Load and split data
python titanic_ml/features/build_features.py # Engineer features
python titanic_ml/data/transformer.py       # Create preprocessor
python titanic_ml/models/train.py           # Train models
```

Trained models are saved to `models/` directory:
- `model.pkl` - Best performing model
- `preprocessor.pkl` - Feature transformer

### Customize Training

Edit `titanic_ml/config/settings.py` to modify:

```python
# Model training settings
CV_FOLDS = 5              # Cross-validation folds
RANDOM_STATE = 42         # Random seed for reproducibility

# Feature engineering
NUMERICAL_FEATURES = ['Age', 'SibSp', 'Parch', 'norm_fare', 'cabin_multiple']
CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked', 'name_title']
```

---

## Project Structure

```
Titanic-Machine-Learning-from-Disaster/
│
├── notebooks/                      # Jupyter notebooks
│   ├── Titanic-Machine-Learning-from-Disaster.ipynb
│   └── utils/                      # Notebook utilities
│
├── titanic_ml/                            # Source code
│   ├── config/                     # Configuration
│   ├── data/                       # Data loading
│   ├── features/                   # Feature engineering
│   ├── models/                     # Training & prediction
│   ├── app/                        # Flask web app
│   └── utils/                      # Utilities
│
├── tests/                          # Test suite
├── models/                         # Trained models
├── artifacts/                      # Processed data
├── static/                         # Web assets
├── templates/                      # HTML templates
└── docs/                           # Documentation
```

---

## Testing

For comprehensive testing information, see the [Testing Guide](testing.md), which covers:
- Running unit and integration tests
- Code quality checks (Black, isort, flake8, mypy)
- Coverage reporting
- CI/CD validation

Quick test command:
```bash
uv run pytest --cov=src --cov-report=html
```

---

## Using with Docker

### Build the Image

```bash
docker build -t titanic-survival-predictor .
```

### Run the Container

```bash
# Basic run
docker run -p 5000:5000 titanic-survival-predictor

# Run with volume for models
docker run -p 5000:5000 -v $(pwd)/models:/app/models titanic-survival-predictor

# Run in background
docker run -d -p 5000:5000 --name titanic-predictor titanic-survival-predictor
```

### Stop and Remove

```bash
docker stop titanic-predictor
docker rm titanic-predictor
```

See the [Deployment Guide](deployment.md) for production deployment.

---

## Common Issues

### `uv` not found

Install uv first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### ModuleNotFoundError

Ensure you're in the project directory and have installed dependencies:

```bash
cd Titanic-Machine-Learning-from-Disaster
uv sync
```

### Port 5000 Already in Use

Change the port in `app.py`:

```python
# Change this line
app.run(host='0.0.0.0', port=5000, debug=True)

# To this
app.run(host='0.0.0.0', port=8000, debug=True)
```

### Model Files Not Found

Download pre-trained models or train your own:

```bash
# Train models
python scripts/run_training.py

# Models will be saved to models/ directory
```

---

## Next Steps

<div class="grid cards" markdown>

- :material-rocket:{ .lg .middle } **Deploy to Cloud**

    ---

    Deploy to Render, Azure, or AWS

    [:octicons-arrow-right-24: Deployment Guide](deployment.md)

- :material-book-open-variant:{ .lg .middle } **Explore the Notebook**

    ---

    Dive deep into the analysis with the Jupyter notebook

    [:octicons-arrow-right-24: Open notebooks/Titanic-Machine-Learning-from-Disaster.ipynb](../notebooks/)

- :material-code-braces:{ .lg .middle } **API Integration**

    ---

    Integrate predictions into your app

    [:octicons-arrow-right-24: API Reference](../reference/api.md)

- :material-cog:{ .lg .middle } **Customize & Extend**

    ---

    Add features or improve models

    [:octicons-arrow-right-24: Architecture Guide](architecture.md)

</div>

---

## Need Help?

- Read the [Architecture Guide](architecture.md)
- Open an issue on [GitHub](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/issues)
- Email: k.s.bonev@gmail.com
