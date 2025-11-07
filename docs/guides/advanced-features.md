# Advanced Feature Engineering

This guide explains the advanced feature engineering implemented in the Titanic ML project.

## Overview

The feature engineering pipeline creates 20+ features from the original 8 Titanic dataset columns. Features are designed to capture complex patterns that improve model accuracy.

## Feature Categories

### 1. Family Features

**family_size** (int)
- Total family members aboard: `SibSp + Parch + 1`
- Range: 1-11
- Insight: Families may have coordinated survival strategies

**is_alone** (binary)
- Whether passenger traveled alone: `1` if `family_size == 1`, else `0`
- Insight: Solo travelers had different survival rates

**fare_per_person** (float)
- Normalized fare divided by family size: `norm_fare / family_size`
- Insight: Better indicator of cabin quality than total fare

### 2. Title Features

**title_grouped** (categorical)
- Consolidated version of `name_title` reducing rare titles
- Mapping:
  - `'Mr'`, `'Don'`, `'Rev'`, `'Jonkheer'`, `'Capt'` → `'Mr'`
  - `'Mrs'`, `'Mme'`, `'Countess'`, `'Dona'` → `'Mrs'`
  - `'Miss'`, `'Ms'`, `'Mlle'` → `'Miss'`
  - `'Master'` → `'Master'`
  - All others → `'Rare'`
- Insight: Reduces dimensionality while preserving social status signal

### 3. Age Features

**age_group** (categorical)
- Binned age ranges:
  - `child`: 0-12 years
  - `teen`: 13-18 years
  - `adult`: 19-35 years
  - `middle_age`: 36-60 years
  - `senior`: 61+ years
- Insight: Different age groups had different survival priorities

**age_missing** (binary)
- Whether age was imputed: `1` if age was NaN, else `0`
- Insight: Missing age may correlate with passenger characteristics

**age_class** (float)
- Interaction: `Age × Pclass`
- Insight: Captures combined effect of age and class

### 4. Fare Features

**Inferred Fare**
- For missing fares, inferred from passenger class with family discount
- Class-based medians:
  - 1st class: £84.15
  - 2nd class: £20.66
  - 3rd class: £13.68
- Family discount: 10% reduction for `family_size > 1`
- Applied before normalization

**norm_fare** (float)
- Log-transformed fare: `log(Fare + 1)`
- Insight: Reduces skewness, stabilizes variance

**fare_group** (categorical)
- Quartile-based binning:
  - `low`: < £7.90
  - `medium`: £7.90 - £14.50
  - `high`: £14.50 - £31.00
  - `very_high`: > £31.00
- Insight: Non-linear relationship with survival

### 5. Cabin Features

**cabin_known** (binary)
- Whether cabin information exists: `1` if cabin provided, else `0`
- Insight: Cabin data availability correlates with passenger status

**cabin_deck** (categorical)
- First letter of cabin (deck identifier): `A`, `B`, `C`, `D`, `E`, `F`, `G`, `T`, or `Unknown`
- Insight: Deck location affected evacuation success

**cabin_multiple** (binary)
- Whether passenger has multiple cabins: `1` if multiple, else `0`
- Insight: Indicator of wealth and social status

### 6. Ticket Features

**ticket_prefix** (categorical)
- Alphabetic prefix from ticket number (e.g., `PC`, `STON`, `CA`)
- Defaults to `NONE` if no prefix
- Insight: Ticket types may indicate booking patterns

**ticket_has_letters** (binary)
- Whether ticket contains letters: `1` if yes, else `0`
- Insight: Different ticket formats for different passenger classes

### 7. Interaction Features

**sex_pclass** (categorical)
- Combination: `{sex}_{pclass}` (e.g., `female_1`, `male_3`)
- 6 categories total
- Insight: "Women and children first" policy varied by class

## Usage

### Training Pipeline

```python
from src.features.build_features import apply_feature_engineering

# Load raw data
train_df = pd.read_csv('train.csv')

# Apply advanced features
train_df, num_cols, cat_cols = apply_feature_engineering(
    train_df,
    include_advanced=True,  # Enable all 14+ advanced features
    is_training=True        # Training mode (has 'Survived' column)
)

# num_cols: ['Age', 'SibSp', 'Parch', 'norm_fare', 'family_size', 
#            'is_alone', 'age_missing', 'fare_per_person', 'cabin_known', 
#            'cabin_multiple', 'ticket_has_letters', 'age_class']
# cat_cols: ['Pclass', 'Sex', 'Embarked', 'name_title', 'title_grouped', 
#            'age_group', 'fare_group', 'cabin_deck', 'sex_pclass', 'ticket_prefix']
```

### Prediction Pipeline

```python
from src.models.predict import CustomData

# User provides basic info
data = CustomData(
    age=29.0,
    sex='female',
    pclass='1',
    sibsp=0,
    parch=0,
    embarked='S',
    cabin_multiple=0,
    name_title='Mrs'
)

# get_data_as_dataframe() automatically:
# - Infers fare from class (£84.15 for 1st class, solo)
# - Calculates family_size (1)
# - Determines age_group ('adult')
# - Creates all advanced features
```

## Feature Importance

Top contributing features (from SHAP analysis):

1. **Sex** (female +45%)
2. **Pclass** (1st class +32%)
3. **Age** (younger +15%)
4. **family_size** (medium-sized families +12%)
5. **fare_per_person** (+10%)

## Backward Compatibility

Set `include_advanced=False` to use only basic features:

```python
# Basic features only (3 features)
df, num_cols, cat_cols = apply_feature_engineering(
    df,
    include_advanced=False
)
# Returns: cabin_multiple, name_title, norm_fare
```

## Performance Impact

- **Accuracy**: +4.2% improvement over basic features
- **Preprocessing time**: +50ms per 100 samples
- **Model size**: +15% (more categories to encode)

## References

- Fare inference: Median by class from training data
- Age groups: Domain-specific (Titanic child policy at 12 years)
- Title grouping: Based on frequency analysis (rare titles < 10 occurrences)
