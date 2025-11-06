# Model Explainability with SHAP

This guide explains how SHAP (SHapley Additive exPlanations) is used to interpret model predictions.

## Overview

SHAP provides game theory-based explanations for individual predictions, answering:
- **Which features** contributed to the prediction?
- **How much** did each feature contribute?
- **In what direction** (increase/decrease survival)?

## Why SHAP?

**Advantages**:
- **Model-agnostic**: Works with any ML model
- **Locally accurate**: Explains specific predictions
- **Consistent**: Based on Shapley values from game theory
- **Additive**: Feature contributions sum to final prediction

**Alternatives** (not used):
- LIME: Less consistent, approximate
- Feature importance: Global only, not prediction-specific
- Partial dependence plots: Averages, not individual instances

## Implementation

### Backend (API Endpoint)

The `/api/explain` endpoint generates SHAP plots:

```python
import shap
import matplotlib.pyplot as plt

# Load model and preprocessor
model = load_object(MODEL_PATH)
preprocessor = load_object(PREPROCESSOR_PATH)

# Transform passenger data
features_transformed = preprocessor.transform(passenger_df)

# Create TreeExplainer (optimized for tree-based models)
explainer = shap.TreeExplainer(model)
shap_values = explainer(features_transformed)

# Generate waterfall plot
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig('shap_explanation.png')
```

### Plot Types

**1. Waterfall Plot** (default)
```python
{
  "plot_type": "waterfall"
}
```
- Shows cumulative contribution from base value
- Best for understanding individual predictions
- Visualizes addition/subtraction of feature effects

**2. Force Plot**
```python
{
  "plot_type": "force"
}
```
- Horizontal bar showing features pushing prediction higher/lower
- Red features increase prediction, blue decrease
- Interactive in Jupyter notebooks

**3. Bar Plot**
```python
{
  "plot_type": "bar"
}
```
- Feature importance for single prediction
- Shows absolute SHAP values
- Easier to read than waterfall for many features

## Interpreting SHAP Values

### Base Value

**`base_value = 0.38`**

The average model output (before looking at features). For binary classification:
- Base value ≈ average survival rate in training data
- Starting point before feature contributions

### SHAP Values

**Positive SHAP value**: Feature increases survival probability

Example:
```json
{
  "Sex_female": 0.45,
  "Pclass_1": 0.32
}
```
- Being female increases survival by +45%
- 1st class ticket increases survival by +32%

**Negative SHAP value**: Feature decreases survival probability

Example:
```json
{
  "Age": -0.12,
  "is_alone": -0.08
}
```
- Older age decreases survival by -12%
- Traveling alone decreases survival by -8%

### Final Prediction

$$
\text{Final Prediction} = \text{Base Value} + \sum_{i} \text{SHAP}_i
$$

Example:
```
Base Value:        0.38
Sex_female:      + 0.45
Pclass_1:        + 0.32
Age:             - 0.12
family_size:     + 0.08
fare_per_person: + 0.05
(other features) - 0.29
─────────────────────────
Final:             0.87  (87% survival probability)
```

## Usage Examples

### Python Client

```python
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Request explanation
response = requests.post(
    'http://localhost:8080/api/explain',
    json={
        'pclass': 1,
        'sex': 'female',
        'age': 29.0,
        'sibsp': 0,
        'parch': 0,
        'embarked': 'S',
        'cabin_multiple': 0,
        'name_title': 'Mrs',
        'plot_type': 'waterfall'
    }
)

result = response.json()

# Display top features
print("Top Contributing Features:")
for feature, shap_value in result['top_features'].items():
    print(f"  {feature}: {shap_value:+.3f}")

# Download and display plot
img_url = f"http://localhost:8080{result['explanation_image_url']}"
img_response = requests.get(img_url)
img = Image.open(BytesIO(img_response.content))
img.show()
```

### cURL

```bash
curl -X POST http://localhost:8080/api/explain \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 3,
    "sex": "male",
    "age": 22.0,
    "sibsp": 1,
    "parch": 0,
    "embarked": "S",
    "cabin_multiple": 0,
    "name_title": "Mr",
    "plot_type": "waterfall"
  }'
```

## Common Patterns

### High Survival (>70%)

Typical contributors:
- `Sex_female`: +0.40 to +0.50
- `Pclass_1` or `Pclass_2`: +0.25 to +0.35
- Young `Age`: +0.10 to +0.20
- `family_size` (2-4): +0.05 to +0.15

### Low Survival (<30%)

Typical contributors:
- `Sex_male`: -0.40 to -0.50
- `Pclass_3`: -0.25 to -0.35
- `is_alone`: -0.05 to -0.10
- Large `family_size` (>5): -0.10 to -0.20

### Uncertain Predictions (40-60%)

- Conflicting features (e.g., male + 1st class)
- Missing data (age imputed, cabin unknown)
- Rare combinations (e.g., child in 3rd class)

## Feature Interactions

SHAP captures feature interactions automatically:

**Example**: Young child in 3rd class
- `Age` (child): +0.15 (children prioritized)
- `Pclass_3`: -0.30 (low survival rate)
- **Net effect**: -0.15 (depends on other factors)

**Example**: Adult male in 1st class
- `Sex_male`: -0.45 (males deprioritized)
- `Pclass_1`: +0.32 (wealth helps)
- **Net effect**: -0.13 (wealth partially compensates)

## Advanced: Global Explanations

Generate summary plots for entire dataset:

```python
import shap
import pandas as pd

# Load data and model
df = pd.read_csv('test.csv')
features = preprocessor.transform(df)

# Calculate SHAP values for all samples
explainer = shap.TreeExplainer(model)
shap_values = explainer(features)

# Summary plot (feature importance)
shap.summary_plot(shap_values, features, show=False)
plt.savefig('shap_summary.png')

# Dependence plot (single feature)
shap.dependence_plot(
    'Age',
    shap_values.values,
    features,
    show=False
)
plt.savefig('shap_age_dependence.png')
```

## Performance Considerations

**TreeExplainer Complexity**:
- **Time**: O(TLD²) where T=trees, L=leaves, D=depth
- **Memory**: Stores all tree structures

**Optimization Tips**:
1. **Cache explainer**: Create once, reuse for multiple predictions
2. **Batch predictions**: Compute SHAP for multiple samples together
3. **Limit features**: Show only top N features in plots
4. **Background data**: Use subset of training data (100-1000 samples)

Example caching:
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_explainer():
    model = load_object(MODEL_PATH)
    return shap.TreeExplainer(model)

# Reuse cached explainer
explainer = get_explainer()
shap_values = explainer(features)
```

## Troubleshooting

**ImportError: No module named 'shap'**
```bash
uv pip install shap
```

**Matplotlib backend error**
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

**Plot not showing**
```python
# Save before showing
plt.savefig('plot.png', bbox_inches='tight', dpi=150)
plt.close()  # Clean up memory
```

**SHAP values don't sum to prediction**
- Check `explainer.expected_value` (may be array for multi-class)
- Ensure using same preprocessing as training
- Verify model type (TreeExplainer for tree models only)

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [NIPS Paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
- [Shapley Values in Game Theory](https://en.wikipedia.org/wiki/Shapley_value)

## Future Enhancements

1. **Interactive dashboard**: Real-time SHAP updates with Plotly/Dash
2. **Comparison mode**: Explain differences between two predictions
3. **Counterfactual analysis**: "What if age was 35 instead of 25?"
4. **Global insights**: Pre-computed summary plots for common passenger types
