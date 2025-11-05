# Advanced Features Guide

This guide documents the advanced features added to the Titanic Survival Prediction project.

## Table of Contents

- [SHAP Model Explainability](#shap-model-explainability)
- [Production ML Pipeline](#production-ml-pipeline)
- [Enhanced Visualizations](#enhanced-visualizations)
- [Code Architecture](#code-architecture)

## SHAP Model Explainability

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain machine learning model predictions. It computes the contribution of each feature to a prediction, providing both global and local interpretability.

### Implementation

Located in **Section 9** of the notebook, the SHAP analysis includes:

#### 1. Summary Plot
- Shows global feature importance across all predictions
- Visualizes feature value distributions
- Color-coded by feature value (red = high, blue = low)

```python
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
```

#### 2. Bar Plot
- Mean absolute SHAP values per feature
- Clearer ranking of feature importance
- Complements the summary plot

#### 3. Waterfall Plots
- Individual prediction explanations
- Shows how features push predictions up or down
- Starts from base prediction (E[f(x)]) to final output (f(x))

#### 4. Dependence Plots
- Relationship between feature value and SHAP value
- Reveals feature interactions
- Color-coded by interacting feature

### Key Findings

From SHAP analysis on the Titanic dataset:

1. **Sex** is the dominant predictor (women significantly more likely to survive)
2. **Fare/Pclass** shows strong economic status effect
3. **Age** has non-linear relationship with survival
4. **Title** (extracted from name) provides refined social status signal
5. **Family size** (SibSp + Parch) shows complex interactions

### Best Practices

- Use `TreeExplainer` for tree-based models (fast and exact)
- Sample data for large datasets (500-1000 examples sufficient)
- Focus on class 1 (positive class) for binary classification
- Interpret in context of domain knowledge

---

## Production ML Pipeline

### Architecture

The production pipeline (`src/pipeline/train_pipeline.py`) implements scikit-learn's Pipeline API for reproducible, production-ready ML workflows.

### Components

#### 1. FeatureEngineer (Custom Transformer)
```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw Titanic data:
    - cabin_multiple: Number of cabins owned
    - name_title: Social title from name
    - norm_fare: Log-transformed fare
    """
```

#### 2. TitanicPreprocessor
Uses `ColumnTransformer` to handle different feature types:
- **Numeric**: Median imputation + StandardScaler
- **Categorical**: Mode imputation + OneHotEncoder

#### 3. TitanicTrainingPipeline
Complete end-to-end pipeline:
```python
Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', ColumnTransformer(...)),
    ('model', XGBClassifier(...))
])
```

### Advantages

1. **No Data Leakage**: Preprocessing fits only on training data
2. **Reproducibility**: Single object encapsulates entire workflow
3. **Deployment Ready**: Can be serialized and deployed directly
4. **Maintainability**: Changes propagate automatically
5. **Testing**: Easy to unit test individual components

### Usage

#### Training
```python
from src.pipeline.train_pipeline import TitanicTrainingPipeline

pipeline = TitanicTrainingPipeline(random_state=42)
results = pipeline.train(X_train, y_train, cv_folds=5)
pipeline.save('model.pkl')
```

#### Inference
```python
pipeline = TitanicTrainingPipeline.load('model.pkl')
predictions = pipeline.predict(new_data)
```

### Comparison: Notebook vs Pipeline

| Aspect | Notebook Approach | Pipeline Approach |
|--------|------------------|-------------------|
| **Flexibility** | High - experiment freely | Lower - structured flow |
| **Reproducibility** | Manual seed setting | Built-in via Pipeline |
| **Deployment** | Requires code rewrite | Direct serialization |
| **Testing** | Difficult | Easy - test each step |
| **Data Leakage Risk** | Higher | Lower - automatic |
| **Best For** | Research, EDA | Production, deployment |

**Recommendation**: Use notebook for exploration, pipeline for deployment.

---

## Enhanced Visualizations

### Cross-Validation Score Distributions

**Function**: `plot_cv_score_distribution()`

Shows boxplots of CV scores across folds for multiple models:
- Identifies model stability
- Compares variance across models
- Reveals overfitting indicators

```python
from src.visualization import plot_cv_score_distribution

cv_scores = {
    'Model A': [0.82, 0.81, 0.83, 0.80, 0.82],
    'Model B': [0.85, 0.70, 0.88, 0.72, 0.86]  # High variance!
}

plot_cv_score_distribution(cv_scores, title='CV Score Stability')
```

### Precision-Recall Curves

**Function**: `plot_precision_recall_curve()`

Especially useful for imbalanced datasets:
- Alternative to ROC curves
- Better for rare positive class
- Shows precision/recall trade-off

```python
from src.visualization import plot_precision_recall_curve

plot_precision_recall_curve(
    y_true, 
    y_pred_proba,
    title='Precision-Recall: Titanic Survival'
)
```

### Model Comparison

**Function**: `plot_model_comparison()`

Horizontal bar chart comparing models:
- Clean visualization of scores
- Easy to identify best performer
- Supports any metric

---

## Code Architecture

### Project Organization

Following Python best practices:

```
src/
├── notebook_config.py      # Notebook-specific settings
├── utils.py                # Reusable utility functions
├── visualization.py        # Plotting functions
├── components/             # Modular ML components
└── pipeline/               # Production pipelines
    ├── train_pipeline.py   # Training workflow
    └── predict_pipeline.py # Inference workflow
```

### Design Principles

1. **Separation of Concerns**
   - Notebook: exploration and analysis
   - Config: parameters and constants
   - Utils: reusable logic
   - Pipeline: production code

2. **No Wrapper Functions**
   - Direct imports from `src.utils`
   - Cleaner, more maintainable code
   - Single source of truth

3. **Type Hints & Docstrings**
   ```python
   def evaluate_models_cv(
       models: Dict[str, Any], 
       X_train: Union[pd.DataFrame, np.ndarray], 
       y_train: Union[pd.Series, np.ndarray],
       cv_folds: int = 5,
       scoring: str = 'accuracy'
   ) -> Dict[str, float]:
       """Evaluate multiple models using cross-validation."""
   ```

4. **Configuration Management**
   - `notebook_config.py` for notebook settings
   - Centralized constants (RANDOM_SEED, CV_FOLDS, etc.)
   - Easy to adjust and track

### Dependency Management

**pyproject.toml** with grouped dependencies:
```toml
[project.dependencies]
- Core ML: pandas, numpy, scikit-learn, xgboost, catboost
- Explainability: shap
- Visualization: matplotlib, seaborn
- Web: Flask

[dependency-groups]
dev = [pytest, black, mypy, flake8]
notebooks = [jupyterlab, ipykernel]
docs = [mkdocs, mkdocs-material]
```

---

## Advanced Usage Patterns

### 1. Custom Model Integration

Add new models to the pipeline:
```python
from sklearn.ensemble import HistGradientBoostingClassifier

custom_model = HistGradientBoostingClassifier()
pipeline = TitanicTrainingPipeline(model=custom_model)
```

### 2. Hyperparameter Tuning

Grid search over pipeline parameters:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 0.5]
}

grid_search = GridSearchCV(pipeline.pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 3. Feature Importance from Pipeline

Extract feature importance after training:
```python
# Train pipeline
pipeline.train(X_train, y_train)

# Get feature names after encoding
feature_names = pipeline.pipeline.named_steps['preprocessing'].get_feature_names_out()

# Get model importances
importances = pipeline.pipeline.named_steps['model'].feature_importances_

# Visualize
plot_feature_importance(feature_names, importances)
```

### 4. SHAP on Pipeline Predictions

Explain pipeline predictions with SHAP:
```python
# Get preprocessed data
X_processed = pipeline.pipeline.named_steps['preprocessing'].transform(
    pipeline.pipeline.named_steps['feature_engineering'].transform(X_sample)
)

# Create explainer
explainer = shap.TreeExplainer(pipeline.pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_processed)

# Visualize
shap.summary_plot(shap_values, X_processed)
```

---

## Performance Considerations

### SHAP Computation

- **TreeExplainer**: Fast for tree models (seconds)
- **KernelExplainer**: Slow for any model (minutes)
- **Sample size**: 100-500 examples usually sufficient
- **Trade-off**: Speed vs. comprehensiveness

### Pipeline Serialization

- **Size**: ~10-50MB depending on model
- **Load time**: <1 second
- **Format**: Pickle (sklearn standard)
- **Alternative**: ONNX for production at scale

### Cross-Validation

- **5-fold CV**: Good balance (5-10 recommended)
- **Stratified**: Use for imbalanced classes
- **Parallel**: n_jobs=-1 for speed
- **Cost**: 5x training time

---

## Troubleshooting

### SHAP Issues

**Problem**: "TreeExplainer not supported"
```python
# Solution: Use KernelExplainer
explainer = shap.KernelExplainer(
    model.predict_proba,
    shap.sample(X_train, 100)
)
```

**Problem**: Large memory usage
```python
# Solution: Reduce sample size
X_sample = X_train.sample(n=200, random_state=42)
```

### Pipeline Issues

**Problem**: "Feature names mismatch"
```python
# Solution: Ensure consistent feature engineering
# Both train and predict must use same FeatureEngineer
```

**Problem**: "Cannot pickle local object"
```python
# Solution: Define classes at module level
# Avoid lambda functions in transformers
```

---

## Future Enhancements

Potential additions to consider:

1. **LIME Integration**: Alternative explainability method
2. **Feature Selection**: Automated feature importance thresholding
3. **Automated Retraining**: Monitoring and model refresh
4. **A/B Testing**: Compare model versions in production
5. **Drift Detection**: Monitor data distribution changes
6. **Model Ensemble**: Stacking or blending pipelines

---

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
