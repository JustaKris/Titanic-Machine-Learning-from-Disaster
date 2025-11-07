# Titanic Survival Analysis & Prediction

<div align="center">

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/c/titanic)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A production-ready machine learning system for predicting passenger survival on the Titanic**

[Getting Started](guides/quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster){ .md-button }

</div>

---

## Overview

This portfolio project demonstrates professional machine learning engineering practices through the classic Kaggle Titanic competition. It showcases how to build production-grade ML systems with clean code, comprehensive documentation, and deployment-ready architecture:

- 🔬 **Research & Development**: Comprehensive Jupyter notebook with EDA, feature engineering, and model optimization
- 🏗️ **Production Architecture**: Modular, scalable ML pipeline following software engineering best practices
- 🌐 **Web Deployment**: Flask API with Docker containerization and cloud deployment
- 📊 **Model Explainability**: SHAP analysis for interpretable predictions
- ✅ **Code Quality**: Type hints, testing, CI/CD, and comprehensive logging

---

## Quick Links

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Get the project running locally in under 5 minutes

    [:octicons-arrow-right-24: Installation Guide](guides/quickstart.md)

- :material-cloud-upload:{ .lg .middle } **Deployment**

    ---

    Deploy to Docker, Render, or Azure with CI/CD pipelines

    [:octicons-arrow-right-24: Deployment Guide](guides/deployment.md)

- :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Complete documentation of all modules and functions

    [:octicons-arrow-right-24: API Docs](reference/api.md)

- :material-book-open:{ .lg .middle } **Architecture**

    ---

    Understand the system design and ML pipeline

    [:octicons-arrow-right-24: Architecture Guide](guides/architecture.md)

</div>

---

## Project Highlights

### Machine Learning Excellence

- **8+ Model Comparison**: Logistic Regression, Random Forest, XGBoost, CatBoost, SVM, and more
- **Hyperparameter Optimization**: GridSearchCV with stratified K-fold cross-validation
- **Ensemble Methods**: Voting classifiers for robust predictions
- **Advanced Feature Engineering**: Domain-driven features (cabin analysis, title extraction, fare normalization)
- **Model Explainability**: SHAP waterfall plots and feature importance analysis

**Results**: 85.3% accuracy with XGBoost (top 10% on Kaggle leaderboard)

### Software Engineering

- **Modular Design**: Clean separation of data, features, models, and API layers
- **Type Safety**: Comprehensive type hints throughout codebase
- **Testing**: Unit and integration tests with pytest
- **Logging**: Azure-ready structured logging with JSON output
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Containerization**: Optimized Docker images with multi-stage builds

---

## Technology Stack

### Core ML Libraries
- **scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting
- **CatBoost** - Categorical boosting
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Explainability & Visualization
- **SHAP** - Model explainability
- **matplotlib** - Plotting
- **seaborn** - Statistical visualizations

### Web & Deployment
- **Flask** - Web framework
- **Docker** - Containerization
- **Render/Azure** - Cloud hosting
- **GitHub Actions** - CI/CD

---

## Getting Started

Choose your path:

=== "Quick Start"

    Get running locally in 5 minutes:

    ```bash
    # Clone repository
    git clone https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster.git
    cd Titanic-Machine-Learning-from-Disaster

    # Install dependencies
    uv sync

    # Run web app
    python app.py
    ```

    [:octicons-arrow-right-24: Full Setup Guide](guides/quickstart.md)

=== "Analysis Notebook"

    Dive into the analysis in the included Jupyter notebook:

    ```bash
    # Install with notebook dependencies
    uv sync --group notebooks

    # Launch Jupyter
    jupyter lab notebooks/Titanic-Machine-Learning-from-Disaster.ipynb
    ```

    The notebook demonstrates advanced feature engineering, model training, and SHAP explanations.

=== "Docker Deployment"

    Deploy with Docker:

    ```bash
    # Build image
    docker build -t titanic-survival-predictor .

    # Run container
    docker run -p 5000:5000 titanic-survival-predictor
    ```

    [:octicons-arrow-right-24: Deployment Guide](guides/deployment.md)

---

## Documentation Contents

### Guides

- **[Quick Start](guides/quickstart.md)** - Get up and running
- **[Installation](guides/installation.md)** - Detailed setup instructions
- **[Testing](guides/testing.md)** - Run tests and code quality checks
- **[Deployment](guides/deployment.md)** - Deploy to production
- **[Architecture](guides/architecture.md)** - System design overview
- **[Configuration](guides/configuration.md)** - Settings and options
- **[Advanced Features](guides/advanced-features.md)** - Deep dive into feature engineering
- **[API Reference](guides/api.md)** - Complete API reference
- **[Explainability](guides/explainability.md)** - Model interpretation with SHAP

---

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 85.3% | XGBoost (tuned) |
| **F1 Score** | 84.3% | Weighted average |
| **API Latency** | <50ms | Average prediction time |
| **Docker Image** | ~1.2GB | Optimized multi-stage build |

---

## Live Demo

Try the deployed application:

🔗 **[Titanic Survival Predictor](https://titanic-ml-kaggle.onrender.com)** *(may take ~30s to wake from sleep)*

---

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Improve documentation
- Share feedback

---

## License

This project is licensed under the MIT License.

---

## Contact

**Kristiyan Bonev**

- GitHub: [@JustaKris](https://github.com/JustaKris)
- Email: k.s.bonev@gmail.com

---

<div align="center">

**⭐ If this project helps you, please star it on GitHub! ⭐**

</div>
