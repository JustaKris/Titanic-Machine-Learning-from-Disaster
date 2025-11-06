# Methodology

This document explains the machine learning approach used in the Titanic Survival Prediction project.

## 1. Data Exploration

### Initial Dataset Analysis

The Titanic dataset contains 891 training samples with 11 features:

**Numeric Features:**
- Age: Passenger age (177 missing values)
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Fare: Ticket price

**Categorical Features:**
- Pclass: Ticket class (1st, 2nd, 3rd)
- Sex: Gender (male, female)
- Embarked: Port of embarkation (C/Q/S, 2 missing values)
- Cabin: Cabin number (687 missing values)
- Ticket: Ticket number
- Name: Passenger name

**Target Variable:**
- Survived: 0 (did not survive) or 1 (survived)

### Key Findings from EDA

1. **Class Imbalance**: 62% did not survive vs 38% survived
2. **Gender Disparity**: ~74% of females survived vs ~19% of males
3. **Class Effect**: 1st class had ~63% survival rate vs 24% in 3rd class
4. **Age Distribution**: Children (<16) had higher survival rates
5. **Fare Correlation**: Higher fares strongly correlated with survival

## 2. Feature Engineering

### Created Features

#### 1. `cabin_multiple`
```python
# Number of cabins a passenger booked
cabin_multiple = lambda x: 0 if pd.isna(x) else len(x.split(' '))
```
**Rationale**: Multiple cabins might indicate wealth and better access to lifeboats.

#### 2. `name_title`
```python
# Extract title from passenger name
name_title = lambda x: x.split(',')[1].split('.')[0].strip()
```
**Rationale**: Titles (Mr., Mrs., Master, etc.) encode both age and social status.

**Title Distribution:**
- Mr: 517 passengers
- Miss: 182 passengers
- Mrs: 125 passengers
- Master: 40 passengers (young boys)
- Rare titles: Rev, Dr, Col, etc.

#### 3. `norm_fare`
```python
# Log normalization of fare
norm_fare = np.log(Fare + 1)
```
**Rationale**: Fare is heavily right-skewed; log transformation creates normal distribution.

### Dropped Features

- **Cabin**: 77% missing values, too sparse for reliable imputation
- **Ticket**: High cardinality, no clear pattern
- **PassengerId**: Just an identifier
- **Name**: Information extracted into `name_title`

## 3. Data Preprocessing

### Missing Value Imputation

| Feature | Strategy | Rationale |
|---------|----------|-----------|
| Age | Median (28 years) | Robust to outliers, represents central tendency |
| Fare | Median ($14.45) | Only 1 missing value in test set |
| Embarked | Most frequent (S) | Only 2 missing values |

### Feature Scaling

**StandardScaler** applied to numeric features:
- Age
- SibSp
- Parch
- norm_fare

**Formula**: `z = (x - μ) / σ`

**Rationale**:
- Required for distance-based algorithms (KNN, SVM)
- Improves convergence for gradient descent
- Prevents feature dominance

### Encoding Categorical Variables

**OneHotEncoding** applied to:
- Pclass → Pclass_1, Pclass_2, Pclass_3
- Sex → Sex_female, Sex_male
- Embarked → Embarked_C, Embarked_Q, Embarked_S
- name_title → name_title_Master, name_title_Miss, etc.

**Final Feature Count**: 40+ features after encoding

## 4. Model Selection

### Baseline Models Evaluated

1. **Logistic Regression** (82.1%)
   - Simple, interpretable baseline
   - Linear decision boundary

2. **K-Nearest Neighbors** (80.5%)
   - Non-parametric, instance-based
   - Sensitive to scaling (hence preprocessing)

3. **Decision Tree** (77.6%)
   - Non-linear decision boundaries
   - Prone to overfitting

4. **Random Forest** (80.6%)
   - Ensemble of decision trees
   - Reduces overfitting through bagging

5. **Naive Bayes** (72.6%)
   - Probabilistic classifier
   - Assumes feature independence (violated here)

6. **Support Vector Classifier** (83.2%)
   - Finds optimal hyperplane
   - Kernel trick for non-linearity

7. **XGBoost** (81.8%)
   - Gradient boosting framework
   - Sequential error correction

8. **CatBoost** (baseline not run)
   - Handles categorical features natively
   - Robust to overfitting

### Model Selection Criteria

- **Primary Metric**: Accuracy (Kaggle competition metric)
- **Secondary Metric**: Weighted F1-score (better for imbalance)
- **Cross-Validation**: 5-fold stratified CV
- **Reproducibility**: Fixed random seed (42)

## 5. Hyperparameter Tuning

### GridSearchCV Configuration

```python
GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)
```

### Tuned Hyperparameters

#### XGBoost (Best Performer: 85.3%)
```python
{
    'n_estimators': 550,
    'learning_rate': 0.5,
    'max_depth': 10,
    'colsample_bytree': 0.75,
    'subsample': 0.6,
    'gamma': 0.5,
    'reg_lambda': 10
}
```

#### Random Forest (83.6%)
```python
{
    'n_estimators': 300,
    'criterion': 'gini',
    'max_depth': 15,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
```

#### CatBoost (84.2%)
```python
{
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 2,
    'border_count': 128
}
```

## 6. Ensemble Methods

### Voting Classifier

Combined predictions from multiple models:

**Hard Voting**: Majority vote
```
Prediction = mode([model1_pred, model2_pred, model3_pred])
```

**Soft Voting**: Average probabilities
```
Probability = mean([model1_proba, model2_proba, model3_proba])
Prediction = argmax(Probability)
```

**Best Configuration**:
- Models: XGBoost, Random Forest, CatBoost
- Voting: Soft
- Weights: Optimized via GridSearchCV
- Performance: 85.1% accuracy

### Why Ensemble Works

1. **Diversity**: Different algorithms capture different patterns
2. **Bias-Variance Trade-off**: Reduces overfitting
3. **Robustness**: More stable predictions
4. **Error Correction**: Models compensate for each other's weaknesses

## 7. Model Evaluation

### Metrics Used

1. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
   - Primary Kaggle metric
   - Overall correctness

2. **Weighted F1-Score**: Harmonic mean of precision and recall
   - Better for imbalanced classes
   - Accounts for class distribution

3. **ROC-AUC**: Area under ROC curve
   - Threshold-independent
   - Measures ranking quality

4. **Confusion Matrix**:
   - True Positives, False Positives
   - True Negatives, False Negatives

### Cross-Validation Strategy

**5-Fold Stratified CV**:
- Maintains class proportions in each fold
- Reduces variance in performance estimates
- Detects overfitting

## 8. Feature Importance

### Top 10 Features (XGBoost)

1. **norm_fare** (0.182): Strongest predictor
2. **Sex_male** (0.156): Gender critical to survival
3. **Age** (0.143): "Women and children first"
4. **name_title_Mr** (0.089): Adult male indicator
5. **Pclass_3** (0.072): Lower class disadvantage
6. **name_title_Mrs** (0.058): Married women prioritized
7. **SibSp** (0.047): Family size effect
8. **Pclass_1** (0.041): First class advantage
9. **cabin_multiple** (0.038): Cabin access
10. **Embarked_S** (0.032): Port effect

### Insights

- **Socioeconomic Status**: Fare and class dominate
- **Demographic Factors**: Gender and age crucial
- **Social Context**: Titles encode multiple signals
- **Feature Engineering**: Created features add value

## 9. Results & Kaggle Submission

### Final Model Performance

| Submission | Model | CV Accuracy | Kaggle Score |
|------------|-------|-------------|--------------|
| 01 | Voting Baseline | 84.5% | TBD |
| 02 | XGBoost Tuned | 85.3% | TBD |
| 03 | Optimized Ensemble | 85.1% | TBD |

### Key Takeaways

1. **Feature Engineering Crucial**: +2-3% improvement
2. **Hyperparameter Tuning Effective**: +1-3% per model
3. **Ensembles Provide Stability**: Consistent performance
4. **Data Quality > Model Complexity**: Clean preprocessing essential

## 10. Future Improvements

Potential enhancements:

1. **Advanced Feature Engineering**:
   - Family group survival analysis
   - Deck location from cabin
   - Ticket prefix patterns

2. **Model Stacking**:
   - Meta-learner on top of base models
   - Potentially higher accuracy

3. **Neural Networks**:
   - TabNet or entity embeddings
   - May capture complex interactions

4. **Feature Selection**:
   - Recursive feature elimination
   - Reduce overfitting risk

5. **External Data**:
   - Historical passenger manifests
   - Ship layout information

---

This methodology represents industry best practices for tabular data classification problems.
