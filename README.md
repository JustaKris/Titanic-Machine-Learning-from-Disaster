# 🚢 Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/titanic)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

- Python 3.11+
- uv (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter lab notebook/Titanic-Machine-Learning-from-Disaster.ipynb
```

### Running the Web App

```bash
python app.py
# Navigate to http://localhost:5000
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
black src/ notebook/

# Lint code
flake8 src/

# Type checking
mypy src/

# Run tests
pytest tests/ --cov=src
```

### Project Configuration

Key configuration in `src/config.py`:

- Random seeds for reproducibility
- Model hyperparameters
- Feature engineering settings
- File paths and directories

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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
