# Class Imbalance Handling

Strategies for dealing with imbalanced classes in the Titanic dataset.

## Problem

The Titanic dataset has imbalanced classes:
- **62% Did Not Survive** (0)
- **38% Survived** (1)

This can lead to models biased toward the majority class.

## Solution: Stratified Cross-Validation

All cross-validation uses `StratifiedKFold` to maintain class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Benefits

- Maintains exact class proportions in each fold
- More reliable performance estimates
- Reduces variance across folds
- Better for small datasets

## Additional Techniques

### 1. Class Weights

Penalize misclassifications of minority class more heavily:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
```

### 2. SMOTE

Synthetic Minority Over-sampling Technique:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 3. Ensemble Methods

Use models robust to imbalance:
- **Random Forest**: Built-in class weighting
- **XGBoost**: `scale_pos_weight` parameter
- **CatBoost**: Handles imbalance automatically

### 4. Evaluation Metrics

Use metrics appropriate for imbalanced data:
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Threshold-independent
- **Precision-Recall Curve**: Better than ROC for imbalance

## Current Implementation

The project uses:

1. ✅ **Stratified K-Fold CV** (5 folds)
2. ✅ **Weighted F1-Score** as secondary metric
3. ✅ **XGBoost/CatBoost** (imbalance-aware models)
4. ✅ **ROC curves** for threshold analysis

## When to Use Each Technique

| Technique | Use When | Avoid When |
|-----------|----------|------------|
| Stratified CV | Always (best practice) | Never |
| Class Weights | Moderate imbalance (30-70%) | Extreme imbalance (<5%) |
| SMOTE | Severe imbalance (<20%) | Sufficient minority samples |
| Ensemble | Any imbalance | Need interpretability |

## Further Reading

- [Stratified K-Fold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold)
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Class Weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
