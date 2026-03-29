# 🚢 Titanic Survival Prediction

[![CI](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/ci.yml/badge.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-82%20passed-success.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions)
[![Coverage](https://img.shields.io/badge/coverage-66%25-yellowgreen.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade ML pipeline demonstrating end-to-end machine learning engineering:** from exploratory research to deployed web application with REST API, comprehensive testing, and CI/CD automation.

---

## 🌐 Live Demo

**🚀 [Try the Live Application](https://titanic-survival-predictor.azurewebsites.net)** | **📊 [View Research Notebook](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/blob/main/notebooks/Titanic-Machine-Learning-from-Disaster.ipynb)** | **🚂 [Trigger a Training Run](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/train-model.yml)** | **📚 [Read Documentation](https://justakris.github.io/Titanic-Machine-Learning-from-Disaster/)**

---

## 📖 Table of Contents

- [Project Highlights](#-project-highlights)
- [Live Application](#-live-application)
- [Research Notebook](#-research-notebook)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [Development](#-development)
- [Deployment](#-deployment)
- [Contact](#-contact)

---

## ✨ Project Highlights

This project showcases **professional ML engineering practices** through two complementary components:

### 🔬 **Research Component**

Comprehensive Jupyter notebook featuring:

- **Exploratory Data Analysis (EDA)** with 20+ visualizations
- **Advanced Feature Engineering**: Title extraction, cabin analysis, fare normalization
- **Model Comparison**: Evaluated 8 algorithms (Random Forest, XGBoost, CatBoost, SVM, etc.)
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
- **Model Interpretability**: SHAP analysis, feature importance, force plots

### 🚀 **Production Component**

Enterprise-ready web application with:

- **RESTful API** with Swagger/OpenAPI documentation
- **Flask Web Interface** for real-time predictions with confidence scores
- **Modular ML Pipeline**: Separate data ingestion, transformation, and training modules
- **Production Best Practices**: Type hints, logging, error handling, comprehensive testing
- **CI/CD Pipeline**: Automated testing, security scanning, Docker builds
- **Cloud Deployment**: Azure App Service with container deployment

**Key Differentiator:** Unlike typical Kaggle projects, this demonstrates the complete ML lifecycle from research to production deployment.

---

## 🤖 Automated Model Training

[![Train Model](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/train-model.yml/badge.svg)](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/train-model.yml)

Experience a **production-grade ML training pipeline** running entirely in GitHub Actions. Watch the complete model training process from data download to performance reporting.

### 🚀 Try It Yourself

1. **[View Past Training Runs](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/train-model.yml)** - See detailed logs and metrics from previous executions
2. **[Trigger a Training Run](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/actions/workflows/train-model.yml)** - Click "Run workflow" to start a new training session
3. **Download Artifacts** - Get trained models, predictions, and detailed performance reports

### ⚡ What Happens During Training

The automated pipeline executes a complete ML workflow (~15 minutes total):

- **📥 Data Acquisition** (10s) - Downloads Titanic dataset from Kaggle API or fallback source
- **✅ Data Validation** (5s) - Verifies data integrity and structure
- **🧪 Pre-Training Tests** (15s) - Runs unit tests to ensure code quality
- **🚂 Model Training** (~12 minutes) - Trains ensemble of 6 models with hyperparameter optimization
- **📊 Performance Reporting** (30s) - Generates comprehensive metrics and visualizations
- **🎯 Kaggle Submission** (20s) - Creates competition-ready prediction file

### 📦 Training Outputs

Each run produces downloadable artifacts:

- **Trained Models** (.pkl) - Serialized model objects ready for deployment
- **Performance Metrics** (JSON) - Accuracy, precision, recall, F1 scores
- **Training Report** (Markdown) - Detailed analysis with metric tables
- **Kaggle Submission** (CSV) - Competition-ready predictions
- **Training Logs** (TXT) - Complete execution trace

### 🎯 Professional Features

- **Automatic Data Download** - No data files committed to repo
- **Comprehensive Logging** - Track every step with detailed status indicators
- **Error Handling** - Graceful fallbacks for missing data sources
- **Performance Tracking** - Timing and metrics for each pipeline stage
- **Artifact Management** - 90-day retention for models, 30-day for submissions
- **Job Summaries** - Beautiful GitHub Actions dashboard with key metrics
- **Multiple Triggers** - Manual, scheduled (weekly), or on code changes

> 💡 **Note:** Training takes approximately 12-15 minutes to complete. The model training phase itself requires ~12 minutes due to extensive hyperparameter optimization across multiple algorithms.

---

## 🎯 Live Application

### Features

🔮 **Real-Time Predictions**

- Enter passenger details through intuitive web form
- Receive instant survival prediction with confidence percentage
- Confidence scores range from realistic probabilities (not just 0% or 100%)

🎨 **User Interface**

- Clean, responsive design optimized for mobile and desktop
- Clear visualization of prediction results
- Direct links to research notebook and project repository

🔌 **REST API**

- `/api/predict` - Get prediction with JSON input
- `/api/health` - Health check endpoint
- `/api/docs` - Interactive Swagger UI documentation
- Full CORS support for integration

📊 **Under the Hood**

- VotingClassifier ensemble (6 models)
- Automatic feature engineering (family size, title extraction, fare inference)
- Confidence calculated by averaging individual model probabilities
- Input validation with Pydantic schemas

### Try It Live

👉 **[Launch Application](https://titanic-survival-predictor.azurewebsites.net)**

**Example API Request:**
```bash
curl -X POST https://titanic-survival-predictor.azurewebsites.net/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22,
    "sex": "female",
    "pclass": "1",
    "sibsp": 0,
    "parch": 0,
    "embarked": "C",
    "name_title": "Miss",
    "cabin_multiple": 1
  }'
```

**Response:**
```json
{
  "prediction": "survived",
  "confidence": "high",
  "probability": 0.968,
  "message": "Passenger likely survived with 96.8% confidence",
  "features": {
    "family_size": 1,
    "inferred_fare": 84.15
  }
}
```

---

## 📓 Research Notebook

### Overview

The Jupyter notebook contains the complete data science workflow with reproducible results and detailed analysis.

**📊 [Open Notebook on GitHub](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/blob/main/notebooks/Titanic-Machine-Learning-from-Disaster.ipynb)**

### Notebook Structure

#### 1. **Data Exploration & Visualization**

- Dataset overview and statistics
- Missing value analysis (Age: 19.9%, Cabin: 77.1%, Embarked: 0.2%)
- Distribution analysis with histograms and box plots
- Correlation heatmaps and pair plots
- Survival rate analysis by features (Sex, Pclass, Age groups, etc.)

#### 2. **Feature Engineering**

Advanced feature creation demonstrating domain knowledge:

- **Title Extraction**: From Name field (Mr, Mrs, Miss, Master, Rare titles)
- **Family Features**: `family_size`, `is_alone` flags
- **Cabin Analysis**: Deck extraction, cabin multiplicity
- **Fare Engineering**: Log transformation, per-person fare, fare groups
- **Interaction Features**: `sex_pclass`, `age_class`
- **Age Imputation**: Median by title and class

#### 3. **Model Training & Evaluation**

Systematic comparison of multiple algorithms:

| Model | Baseline Accuracy | Tuned Accuracy | Improvement |
| ------- | ------------------ | ---------------- | ------------- |
| Logistic Regression | 80.2% | 82.1% | +1.9% |
| K-Nearest Neighbors | 78.5% | 81.3% | +2.8% |
| Support Vector Machine | 81.5% | 83.7% | +2.2% |
| Random Forest | 79.8% | 84.2% | +4.4% |
| **XGBoost** | 82.1% | **85.9%** | **+3.8%** |
| CatBoost | 81.9% | 85.1% | +3.2% |
| **Voting Ensemble** | - | **86.2%** | **Best** |

#### 4. **Model Interpretability**

- **SHAP Analysis**: Waterfall plots, force plots, summary plots
- **Feature Importance**: Random Forest, XGBoost importances
- **Permutation Importance**: Model-agnostic feature ranking
- **Partial Dependence Plots**: Effect of individual features

#### 5. **Results & Insights**

**Most Predictive Features:**

1. **Sex** (female = 74% survival, male = 19% survival)
2. **Passenger Class** (1st = 63%, 2nd = 47%, 3rd = 24%)
3. **Fare** (log-transformed, normalized)
4. **Title** (grouped: Mr, Mrs, Miss, Master, Rare)
5. **Age** (children < 12 had higher survival)

**Key Findings:**

- Feature engineering improved accuracy by **3-5%**
- Ensemble methods outperformed individual models
- Cross-validation showed stable performance (σ < 2%)
- Model achieves **86.2% accuracy** on validation set

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (tested on 3.11, 3.12, 3.13)
- **[uv](https://github.com/astral-sh/uv)** package manager (recommended)
- **Docker** (optional, for containerized deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster

# Install with uv (recommended - 10-100x faster than pip)
uv sync

# Or install with pip
pip install -e .
```

### Running Locally

**Option 1: Web Application**
```bash
# Start Flask server
uv run titanic-api

# Navigate to http://localhost:5000
```

**Option 2: Command-Line Inference**
```bash
# Single prediction
uv run titanic-predict \
  --age 22 \
  --sex female \
  --pclass 1 \
  --name-title Miss \
  --embarked C

# Batch predictions from CSV
uv run titanic-predict --input data/test.csv --output predictions.csv
```

**Option 3: Python API**
```python
from titanic_ml.models.predict import CustomData, PredictPipeline

# Create passenger data
passenger = CustomData(
    age=22,
    sex="female",
    name_title="Miss",
    sibsp=0,
    pclass="1",
    embarked="C",
    cabin_multiple=1
)

# Get prediction
pipeline = PredictPipeline()
df = passenger.get_data_as_dataframe()
prediction, probability = pipeline.predict(df)

print(f"Survived: {bool(prediction[0])}")
print(f"Confidence: {probability[0]:.1%}")
```

### Running the Notebook

```bash
# Install notebook dependencies
uv sync --group notebooks

# Launch Jupyter Lab
jupyter lab notebooks/Titanic-Machine-Learning-from-Disaster.ipynb
```

### Docker Deployment

```bash
# Build image
docker build -t titanic-ml .

# Run container
docker run -p 5000:5000 titanic-ml

# Access at http://localhost:5000
```

---

## 🏗️ Architecture

### Project Structure

```
Titanic-Machine-Learning-from-Disaster/
├── src/
│   └── titanic_ml/                # Main package (src layout)
│       ├── config/
│       │   └── settings.py        # Centralized configuration with Pydantic
│       ├── data/
│       │   ├── loader.py          # Data ingestion from CSV
│       │   └── transformer.py     # Feature engineering pipeline
│       ├── features/
│       │   └── build_features.py  # Advanced feature creation
│       ├── models/
│       │   ├── train.py           # Model training with hyperparameter tuning
│       │   ├── predict.py         # Inference pipeline
│       │   └── schemas.py         # Pydantic validation schemas
│       ├── app/
│       │   ├── routes.py          # Flask application with API endpoints
│       │   ├── templates/         # HTML templates
│       │   └── static/            # CSS, JS, images
│       └── utils/
│           ├── logger.py          # Structured logging
│           ├── helpers.py         # Model persistence utilities
│           └── exception.py       # Custom exception handling
├── notebooks/
│   └── Titanic-Machine-Learning-from-Disaster.ipynb
├── tests/
│   ├── unit/                      # Unit tests (60 tests)
│   └── integration/               # API integration tests (22 tests)
├── data/
│   ├── raw/                       # Original Kaggle datasets
│   └── processed/                 # Engineered features
├── saved_models/                  # Saved model artifacts
├── scripts/                       # CLI entry points
├── docs/                          # MkDocs documentation
├── pyproject.toml                 # Modern Python packaging
└── Dockerfile                     # Multi-stage production build
```

### ML Pipeline Flow

```mermaid
graph LR
    A[Raw Data] --> B[Data Loader]
    B --> C[Feature Engineering]
    C --> D[Preprocessing]
    D --> E[Model Training]
    E --> F[Evaluation]
    F --> G{Deploy?}
    G -->|Yes| H[Save Model]
    H --> I[Flask API]
    I --> J[Predictions]
```

### Key Design Decisions

1. **Package Structure**: Uses `src/titanic_ml/` layout for proper Python packaging
2. **Configuration Management**: Centralized settings with Pydantic for type safety
3. **Pipeline Architecture**: sklearn Pipeline with FeatureUnion for reproducibility
4. **Testing Strategy**: 82 tests (66% coverage) with unit + integration tests
5. **CI/CD**: GitHub Actions for automated testing, security scanning, Docker builds
6. **Logging**: Structured JSON logging with different levels for dev/prod

---

## 📊 Model Performance

### Final Model: VotingClassifier Ensemble

**Composition**: 6 base estimators with hard voting

- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression
- CatBoost Classifier
- Support Vector Classifier
- K-Neighbors Classifier

### Metrics

| Metric | Training | Validation |
| ------- | ---------- | ------------ |
| **Accuracy** | 88.5% | 86.2% |
| **Precision** | 87.3% | 84.7% |
| **Recall** | 82.1% | 79.8% |
| **F1 Score** | 84.6% | 82.2% |
| **ROC AUC** | 0.923 | 0.901 |

### Cross-Validation Results

5-Fold stratified cross-validation:

- **Mean Accuracy**: 85.1%
- **Std Deviation**: 1.8%
- **Min**: 82.9%
- **Max**: 87.4%

**Interpretation**: Low variance indicates stable, generalizable model.

### Feature Importance (Top 10)

1. **Sex** - 28.3%
2. **Title (Grouped)** - 15.7%
3. **Fare (Normalized)** - 12.4%
4. **Age** - 9.8%
5. **Pclass** - 8.6%
6. **Family Size** - 6.2%
7. **Cabin Known** - 5.1%
8. **Embarked** - 4.3%
9. **Sex × Pclass** - 3.9%
10. **Is Alone** - 2.8%

### Why This Approach Works

✅ **Ensemble Diversity**: Combines tree-based (RF, XGB, CB) with linear (LR) and distance-based (KNN) models  
✅ **Feature Engineering**: Domain knowledge improves signal extraction  
✅ **Proper Validation**: Stratified CV prevents data leakage  
✅ **Hyperparameter Tuning**: Grid search optimizes each estimator  
✅ **Probability Calibration**: Averaging individual model probabilities gives realistic confidence scores

---

## 🛠️ Technology Stack

### Core ML & Data Science

- **pandas** `2.2+` - Data manipulation and analysis
- **numpy** `1.26+` - Numerical computing
- **scikit-learn** `1.5+` - ML algorithms and pipelines
- **XGBoost** `2.1+` - Gradient boosting
- **CatBoost** `1.2+` - Categorical boosting
- **SHAP** `0.46+` - Model explainability

### Web Framework & API

- **Flask** `3.0+` - Web application framework
- **flask-swagger-ui** `4.11+` - API documentation
- **Pydantic** `2.10+` - Data validation
- **pydantic-settings** `2.6+` - Configuration management

### Development Tools

- **pytest** `8.3+` - Testing framework (82 tests, 66% coverage)
- **pytest-cov** - Code coverage reporting
- **ruff** `0.6+` - Fast linting and formatting (replaces black, flake8, isort)
- **mypy** `1.13+` - Static type checking
- **bandit** `1.8+` - Security scanning
- **pymarkdownlnt** - Markdown linting

### DevOps & Deployment

- **uv** - Fast Python package manager (10-100x faster than pip)
- **Docker** - Containerization
- **GitHub Actions** - CI/CD automation
- **Azure App Service** - Cloud hosting

### Documentation

- **Jupyter Lab** `4.4+` - Interactive notebooks
- **MkDocs Material** `9.5+` - Documentation site
- **matplotlib** `3.9+` - Visualizations
- **seaborn** `0.13+` - Statistical plots

---

## 🧪 Development

### Running Tests

```bash
# Run all tests with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific test categories
uv run pytest tests/unit -v          # Unit tests only
uv run pytest tests/integration -v   # Integration tests only

# Run with specific markers
uv run pytest -m "not slow" -v       # Skip slow tests
```

**Test Coverage**: 82 tests, 66% coverage (exceeds 40% threshold)

### Code Quality

```bash
# Lint code
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Type checking
uv run mypy src/ tests/

# Security scan
uv run bandit -r src/ -c pyproject.toml
uv run pip-audit

# Markdown linting
uv run pymarkdown --config pyproject.toml scan docs/ README.md
```

### Training Models

```bash
# Train with default settings
uv run titanic-train

# Train with custom data paths
uv run python scripts/run_training.py \
  --train-path data/raw/train.csv \
  --test-path data/raw/test.csv \
  --output-dir saved_models/
```

### CLI Commands

The project includes three CLI entry points:

```bash
# Train models
uv run titanic-train

# Make predictions
uv run titanic-predict --age 22 --sex female --pclass 1

# Start API server
uv run titanic-api --port 8000
```

---

## 🚀 Deployment

### Render (Free Tier - Recommended for Demos)

**Live Demo:** 🔗 **[Titanic Survival Predictor on Render](https://titanic-survival-predictor-eymq.onrender.com)**

Render provides free hosting perfect for portfolio projects and demos:

```bash
# 1. Build Docker image
docker build -t titanic-ml .

# 2. Push to Docker Hub
docker tag titanic-ml:latest YOUR_USERNAME/titanic-ml:latest
docker push YOUR_USERNAME/titanic-ml:latest

# 3. Create Render service:
#    - Go to render.com → New Web Service
#    - Select "Deploy existing image from registry"
#    - Image URL: docker.io/YOUR_USERNAME/titanic-ml:latest
#    - Configure health check: /health
#    - Click Create Web Service

# 4. Setup automatic deployments:
#    - GitHub secret: RENDER_DEPLOY_WEBHOOK (copy from Render settings)
#    - CI/CD pipeline automatically redeploys on push
```

**Features:**

- ✅ Free tier with 750 hours/month
- ✅ Auto-deploys on image updates
- ✅ HTTPS included
- ⚠️ Cold starts (~30s) on free tier

---

### Azure App Service (Production)

For production workloads with better performance and scaling:

```bash
# 1. Build Docker image
docker build -t titanic-ml:latest .

# 2. Tag for Azure Container Registry
docker tag titanic-ml:latest YOUR_REGISTRY.azurecr.io/titanic-ml:latest

# 3. Push to registry
docker push YOUR_REGISTRY.azurecr.io/titanic-ml:latest

# 4. Deploy to Azure App Service
az webapp create \
  --resource-group YOUR_RG \
  --plan YOUR_PLAN \
  --name YOUR_APP_NAME \
  --deployment-container-image-name YOUR_REGISTRY.azurecr.io/titanic-ml:latest
```

### Docker Hub

```bash
# Pull pre-built image
docker pull justakris/titanic-ml:latest

# Run container
docker run -d -p 5000:5000 \
  --name titanic-api \
  --health-cmd "curl -f http://localhost:5000/api/health || exit 1" \
  --health-interval=30s \
  justakris/titanic-ml:latest
```

### CI/CD Pipeline

Automated workflows handle testing, building, and deployment:

```
Push to main
    ↓
build.yml (Docker image → Docker Hub)
    ↓
    ├→ deploy-render.yml (auto-deploy to Render)
    └→ deploy-azure.yml (auto-deploy to Azure)
```

**Workflows:**

1. **build-push-docker.yml** - Builds Docker image, pushes to Docker Hub
2. **deploy-render.yml** - Auto-deploys to Render via webhook
3. **deploy-azure.yml** - Auto-deploys to Azure App Service
4. **python-tests.yml** - Tests on Python 3.11-3.12 with coverage
5. **python-lint.yml** - Ruff linting and format checks
6. **python-typecheck.yml** - mypy type checking
7. **security-audit.yml** - Bandit and pip-audit vulnerability scans
8. **markdown-lint.yml** - Markdown linting with pymarkdown
9. **deploy-docs.yml** - Publishes documentation to GitHub Pages

**Required GitHub Secrets:**

- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` - Docker Hub credentials
- `RENDER_DEPLOY_WEBHOOK` - Render auto-deploy webhook (optional)
- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` - Azure credentials (optional)

📖 See `.github/workflows/README.md` for complete CI/CD setup documentation.

---

## 📚 Documentation

Comprehensive documentation available at **[justakris.github.io/Titanic-Machine-Learning-from-Disaster](https://justakris.github.io/Titanic-Machine-Learning-from-Disaster/)**

### Contents

- **Quick Start Guide**: Installation and first predictions
- **API Reference**: Complete function/class documentation
- **Architecture Guide**: System design and patterns
- **Deployment Guide**: Production setup instructions
- **Methodology**: Detailed explanation of approach
- **Advanced Features**: Feature engineering deep-dive

Build docs locally:

```bash
# Install docs dependencies
uv sync --group docs

# Serve documentation
uv run mkdocs serve

# Navigate to http://localhost:8001
```

---

## 🤝 Contributing

Contributions welcome! This is a portfolio project demonstrating ML engineering best practices.

### Development Setup

```bash
# Clone and install
git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster
uv sync --all-groups

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
uv run pytest tests/
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### Pull Request Requirements

All PRs must:

- ✅ Pass all 82 tests (`pytest`)
- ✅ Maintain ≥40% code coverage (`pytest-cov`)
- ✅ Pass security scans (`bandit`, `pip-audit`)
- ✅ Follow code style (`ruff`)
- ✅ Include type hints (`mypy` compatible)
- ✅ Update documentation if adding features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Kristiyan Bonev** - ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kristiyan-bonev/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/JustaKris)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:k.s.bonev@gmail.com)

- 💼 **LinkedIn**: [kristiyan-bonev](https://www.linkedin.com/in/kristiyan-bonev/)
- 🐙 **GitHub**: [@JustaKris](https://github.com/JustaKris)
- 📧 **Email**: <k.s.bonev@gmail.com>
- 🌐 **Project**: [Titanic-Machine-Learning-from-Disaster](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster)

---

## 🎯 Project Goals & Learning Outcomes

This portfolio project demonstrates:

✅ **End-to-End ML Engineering** - Research → Production  
✅ **Software Engineering Best Practices** - Testing, CI/CD, Documentation  
✅ **Cloud Deployment** - Containerization, Azure integration  
✅ **API Development** - RESTful design, Swagger docs  
✅ **Code Quality** - Type hints, linting, formatting  
✅ **Model Interpretability** - SHAP, feature importance  
✅ **Production Readiness** - Logging, error handling, validation  

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

*Built with ❤️ as a portfolio demonstration of professional ML engineering*

</div>
