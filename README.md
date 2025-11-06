# 🚢 Titanic Survival Prediction

[![CI](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/ci.yml/badge.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/ci.yml)
[![Docker Build](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/build-deploy.yml/badge.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/build-deploy.yml)
[![Security Scan](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/security.yml/badge.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/security.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/titanic)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)

> A comprehensive machine learning solution for predicting Titanic passenger survival, featuring advanced feature engineering, ensemble methods, and production-ready deployment architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Results](#results)
- [Web Application](#web-application)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🎯 Overview

This project tackles the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) with a two-pronged approach:

1. **Research Notebook**: Comprehensive exploratory data analysis, feature engineering, and model optimization
2. **Production Application**: Scalable Flask web app with modular ML pipeline architecture

### Jupyter Notebook

The primary research and analysis is documented in an interactive notebook with:
- Complete exploratory data analysis (EDA)
- Advanced feature engineering techniques  
- Comparison of 8+ machine learning algorithms
- Hyperparameter tuning with GridSearchCV
- Ensemble methods and voting classifiers
- Comprehensive visualizations and model interpretability

📓 **[View Research Notebook](./notebook/Titanic-Machine-Learning-from-Disaster.ipynb)**

## 📊 Project Structure

```
Titanic-Machine-Learning-from-Disaster/
├── notebook/
│   ├── Titanic-Machine-Learning-from-Disaster.ipynb  # Main research notebook
│   └── data/                                          # Raw Kaggle datasets
├── src/
│   ├── notebook_config.py           # Notebook-specific configuration
│   ├── utils.py                     # Utility functions (evaluation, model persistence)
│   ├── visualization.py             # Plotting utilities (SHAP, CV distributions, etc.)
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading pipeline
│   │   ├── data_transformation.py  # Feature engineering pipeline
│   │   └── model_trainer.py        # Model training pipeline
│   └── pipeline/
│       ├── train_pipeline.py       # Production sklearn Pipeline
│       └── predict_pipeline.py     # Inference pipeline
├── artifacts/                       # Processed datasets
├── models/                          # Saved model artifacts
├── submissions/                     # Kaggle submission files
├── static/ & templates/             # Flask web app assets
└── app.py                           # Flask application entry point
```

## ✨ Key Features

### Research & Analysis

- 🔍 **Comprehensive EDA**: Multi-dimensional analysis with 20+ visualizations
- 🛠️ **Advanced Feature Engineering**: Title extraction, cabin analysis, fare normalization
- 🤖 **Model Comparison**: Evaluation of 8 algorithms (RF, XGBoost, CatBoost, SVM, etc.)
- 📈 **Hyperparameter Tuning**: GridSearchCV optimization with cross-validation
- 🎭 **Ensemble Methods**: Voting classifiers with soft/hard voting strategies
- � **SHAP Analysis**: Model explainability with summary plots, waterfall diagrams, and dependence plots
- �📊 **Model Interpretability**: Feature importance analysis across multiple models
- 🎯 **Reproducibility**: Fixed random seeds, documented configurations
- 📉 **CV Analysis**: Detailed cross-validation score distributions and stability metrics

### Production Architecture

- 🏗️ **Sklearn Pipelines**: Production-ready ML pipeline with FeatureUnion and ColumnTransformer
- 🔧 **Modular Components**: Separate ingestion, transformation, and training modules
- 💾 **Model Persistence**: Serialized pipelines and preprocessors  
- 🌐 **Web Interface**: Flask application for real-time predictions
- ☁️ **Cloud Deployment**: Render-ready with Docker support
- ✅ **Code Quality**: Type hints, docstrings, Black formatting, pytest coverage

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (tested with 3.11, 3.12, 3.13)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster

# Install dependencies with uv (recommended)
uv sync

# Or install globally
uv sync --no-dev

# Alternative: Use pip with pyproject.toml
pip install -e .
```

### Running the Notebook

```bash
# Launch Jupyter Lab
jupyter lab notebook/Titanic-Machine-Learning-from-Disaster.ipynb
```

### Running the Web App

```bash
# Start Flask application
uv run python -m src.app.routes

# Or using app.py entrypoint
python app.py

# Navigate to http://localhost:5000
```

### Command-Line Scripts

```bash
# Train the model
uv run python scripts/run_training.py

# Generate predictions (batch inference)
uv run python scripts/run_inference.py --input artifacts/test.csv --output submission.csv

# Start API server
uv run python scripts/run_api.py --host 0.0.0.0 --port 5000
```

### Docker Deployment

```bash
# Build the image
docker build -t titanic-ml .

# Run the container
docker run -p 5000:5000 titanic-ml

# Access at http://localhost:5000
```

## 🔬 Methodology

### Data Overview

**Goal**: Predict passenger survival (binary classification)

**Features** (11 total):

- `PassengerId`: Unique identifier
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Sex`: Passenger gender
- `Age`: Passenger age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Ticket cost
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C/Q/S)

**Target**: `Survived` (0 = No, 1 = Yes)

[Kaggle Dataset →](https://www.kaggle.com/competitions/titanic/data)

### Pipeline Stages

1. **Data Ingestion** (`src/components/data_ingestion.py`)
   - Loads raw Kaggle datasets
   - Splits training data for validation
   - Saves processed artifacts

2. **Data Transformation** (`src/components/data_transformation.py`)
   - Feature engineering (cabin_multiple, name_title, norm_fare)
   - Missing value imputation (median strategy for Age/Fare)
   - StandardScaler normalization
   - OneHotEncoding for categorical variables
   - Saves preprocessor pipeline

3. **Model Training** (`src/components/model_trainer.py`)
   - Trains multiple model candidates
   - Hyperparameter tuning with GridSearchCV
   - Ensemble methods (VotingClassifier)
   - Saves best performing model

4. **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`)
   - Loads trained model and preprocessor
   - Real-time inference with confidence scores
   - Input validation and error handling

## 📈 Results

### Model Performance

| Model | Baseline Accuracy | Tuned Accuracy | Improvement |
|-------|------------------|----------------|-------------|
| Logistic Regression | 82.1% | 82.6% | +0.5% |
| K-Nearest Neighbors | 80.5% | 83.0% | +2.5% |
| Random Forest | 80.6% | 83.6% | +3.0% |
| **XGBoost** | 81.8% | **85.3%** | **+3.5%** |
| CatBoost | - | 84.2% | - |
| Voting Ensemble | - | 85.1% | - |

### Key Insights

- **Most Important Features**: Sex, Fare (normalized), Age, Pclass, Title
- **Feature Engineering Impact**: +2-3% accuracy improvement
- **Ensemble Benefit**: Voting classifier provides stable predictions
- **Cross-Validation**: 5-fold CV ensures generalization

## 🌐 Web Application

### Live Demo

The predictor is deployed on Render with a Flask interface:

🔗 **[Live Demo](https://titanic-ml-kaggle.onrender.com)** *(may take ~60s to wake from sleep)*

### Features

- **Real-time Predictions**: Enter passenger details for instant survival probability
- **Confidence Scores**: Model certainty displayed with predictions
- **Responsive UI**: Clean, mobile-friendly interface
- **API Endpoint**: JSON API available for integration

### Local Deployment

```bash
# Run Flask application
python app.py

# Access at http://localhost:5000
```

## 📚 Documentation

Detailed documentation is available in the `/docs` folder (built with MkDocs):

- **Methodology**: Comprehensive explanation of approach
- **API Reference**: Function and class documentation
- **Deployment Guide**: Production setup instructions
- **Contributing Guide**: Development guidelines

Build docs locally:

```bash
mkdocs serve
# Navigate to http://localhost:8000
```

## 🛠️ Development

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run flake8 src/ tests/

# Type checking
uv run mypy src/

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Security scans
uv run bandit -r src/ --skip B104
uv run pip-audit
```

### CI/CD Workflows

This project uses GitHub Actions for continuous integration and deployment:

- **CI** (`ci.yml`): Runs tests, linting, and coverage on Python 3.11-3.13
- **Docker Build** (`build-deploy.yml`): Builds and pushes multi-platform images to Docker Hub
- **Security Scan** (`security.yml`): Runs bandit and pip-audit for vulnerability detection
- **Deploy Docs** (`deploy-docs.yml`): Deploys MkDocs documentation to GitHub Pages
- **Train Model** (`train-model.yml`): Scheduled/manual model training with artifact uploads

### Project Configuration

Key configuration in `pyproject.toml`:

- Package dependencies managed with uv
- Development tools: pytest, black, flake8, mypy
- Project metadata and entry points
- Testing and coverage settings

## 🏗️ Architecture & Deployment

### System Architecture

```
┌─────────────────┐
│  Data Ingestion │ → Loads raw Kaggle datasets
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Transformation  │ → Feature engineering, scaling, encoding
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Model Training  │ → Train/tune models, save artifacts
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Prediction API  │ → Flask REST API for inference
└─────────────────┘
```

### Deployment Options

1. **Local Development**
   - Run Flask app directly: `python app.py`
   - Use scripts for training and inference

2. **Docker Container**
   - Multi-stage build with uv for minimal image size
   - Health checks and unprivileged user for security
   - Optimized layer caching for fast rebuilds

3. **Cloud Platforms**
   - **Render**: Direct deployment from GitHub (current live demo)
   - **Azure Web Apps**: Container deployment with Application Insights
   - **Docker Hub**: Automated multi-platform builds (amd64/arm64)

4. **CI/CD Pipeline**
   - Automated testing on every push
   - Docker images built and pushed on main branch updates
   - Scheduled weekly model retraining
   - Security scanning with bandit and pip-audit

### Production Best Practices

- ✅ Package management with uv (faster than pip)
- ✅ Multi-stage Docker builds for minimal image size
- ✅ Health checks and readiness probes
- ✅ Unprivileged container user for security
- ✅ Automated testing with 40% coverage threshold
- ✅ Security scanning in CI/CD pipeline
- ✅ Model artifact versioning and tracking
- ✅ Comprehensive logging with Azure integration

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

All pull requests must:
- Pass CI tests (pytest, coverage ≥40%)
- Pass security scans (bandit, pip-audit)
- Follow code style (black, flake8)
- Include tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle**: For hosting the competition and providing the dataset
- **Scikit-learn**: Core ML library
- **XGBoost & CatBoost**: Gradient boosting implementations
- **Flask**: Web framework for deployment

## 📧 Contact

**Kristiyan Bonev**

- GitHub: [@JustaKris](https://github.com/JustaKris)
- Email: k.s.bonev@gmail.com
- Project Link: [Titanic-Machine-Learning-from-Disaster](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
