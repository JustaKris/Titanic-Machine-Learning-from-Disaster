# Advanced Features# Advanced features# Advanced features# Advanced Features# Advanced Features

This page summarizes the notebook's advanced techniques: explainability

with SHAP, enhanced visualizations, and production-ready pipeline tips.

This page summarizes the notebook's advanced techniques: explainability

## SHAP Explainability

with SHAP, enhanced visualizations, and production-ready pipeline tips.

SHAP (SHapley Additive exPlanations) attributes model predictions to

features. For tree-based models use `TreeExplainer` (fast). For otherThis page summarizes the notebook's advanced techniques: explainability

models `KernelExplainer` can be used but is much slower.

## SHAP explainability (concise)

Key visualizations:

with SHAP, enhanced visualizations, and production-ready pipeline tips.

- **Summary plot** — global feature importance

- **Dependence plots** — relationship between a feature and its SHAP valueSHAP (SHapley Additive exPlanations) attributes model predictions to

- **Waterfall/force plots** — single-prediction breakdown

features. For tree-based models use `TreeExplainer` (fast). For otherAdvanced capabilities beyond the baseline ML pipeline.Advanced capabilities beyond the baseline ML pipeline.

Example:

models `KernelExplainer` can be used but is much slower.

```python

explainer = shap.TreeExplainer(model)## SHAP explainability (concise)

shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample)Key visualizations:

```

## Feature Engineering Highlights

- Summary plot — global feature importance

Notebook-derived engineered features used in the pipeline:

- Dependence plots — relationship between a feature and its SHAP valueSHAP (SHapley Additive exPlanations) attributes model predictions to

- `cabin_multiple` — number of cabins

- `name_title` — extracted title from passenger name- Waterfall/force plots — single-prediction breakdown

- `norm_fare` — log- or scaled fare

features. For tree-based models use `TreeExplainer` (fast). For other## SHAP Explainability## SHAP Explainability

Keep transformations reproducible: implement them as transformers in

the production `Pipeline` (see `titanic_ml/pipeline/train_pipeline.py`).Example:

## Visualizationsmodels `KernelExplainer` can be used but is much slower.

- `plot_cv_score_distribution()` — boxplots of CV scores to inspect```python

  stability

- `plot_precision_recall_curve()` — preferred for imbalanced datasetsexplainer = shap.TreeExplainer(model)

- `plot_model_comparison()` — horizontal bar chart of model scores

shap_values = explainer.shap_values(X_sample)

## Production Notes

shap.summary_plot(shap_values, X_sample)Key visualizations:

- Use the pipeline for production training and inference to avoid data

  leakage (preprocessing fit only on training data)```

- Serialize pipelines with `pickle` (or ONNX for cross-platform

  serving) and store in `models/`SHAP (SHapley Additive exPlanations) explains model predictions by computing feature contributions.SHAP (SHapley Additive exPlanations) explains model predictions by computing feature contributions.

## Troubleshooting## Feature engineering highlights

- SHAP memory issues: reduce sample size- Summary plot — global feature importance

- Feature name mismatches: ensure the same `FeatureEngineer` is used

  during training and inference- `cabin_multiple`: number of cabins

- Pickling errors: avoid local (nested) class definitions

- `name_title`: title extracted from passenger name- Dependence plots — relationship between a feature and its SHAP value

## Recommended Usage Pattern

- `norm_fare`: normalized or log-transformed fare

Use the notebook for exploration and the pipeline module for training

and deployment. Keep configuration and constants in- Waterfall/force plots — single-prediction breakdown

`notebooks/utils/config.py` to avoid hardcoded values in cells.

Implement these as Transformers in the production pipeline to avoid

---

leakage.### Available Visualizations### Implementation

For more details see the notebook Section 9 (SHAP) and

`titanic_ml/pipeline/train_pipeline.py`.

## VisualizationsExample:

- `plot_cv_score_distribution()` — boxplots of CV scores to inspect# Advanced Features

  model stability

- `plot_precision_recall_curve()` — useful for imbalanced datasets```python

- `plot_model_comparison()` — horizontal bar chart of model scores

explainer = shap.TreeExplainer(model)This page summarizes the notebook's advanced techniques: explainability

## Production notes

shap_values = explainer.shap_values(X_sample)with SHAP, enhanced visualizations, and production-ready pipeline tips.

- Use `titanic_ml/pipeline/train_pipeline.py` for production training to ensure

  consistent preprocessing and avoid leakageshap.summary_plot(shap_values, X_sample)

- Serialize pipelines (pickle/ONNX) and store under `models/`

## SHAP Explainability (Concise)

## Troubleshooting

- SHAP memory issues: reduce sample size

- Feature name mismatches: ensure same feature engineering at train## Feature engineering highlightsSHAP (SHapley Additive exPlanations) attributes model predictions to

  and inference

- Pickling errors: define classes at module levelfeatures. Use TreeExplainer for tree models (fast) and KernelExplainer

---- `cabin_multiple`: number of cabinsonly when necessary (slow).

For more details see the notebook Section 9 (SHAP) and- `name_title`: title extracted from passenger name

`titanic_ml/pipeline/train_pipeline.py`.

- `norm_fare`: normalized or log-transformed fareKey visualizations:

Implement these as Transformers in the production pipeline to avoid- Summary plot — global feature importance

leakage.- Dependence plots — feature value vs. SHAP value

- Waterfall / force plots — single-prediction explanations

## Visualizations

Best practices:

- `plot_cv_score_distribution()` — boxplots of CV scores to inspect

  model stability- For large datasets sample 100–500 rows for SHAP summaries

- `plot_precision_recall_curve()` — useful for imbalanced datasets- Focus on the positive class for binary classification (Survived=1)

- `plot_model_comparison()` — horizontal bar chart of model scores- Use TreeExplainer for tree-based models

## Production notesExample:

- Use `titanic_ml/pipeline/train_pipeline.py` for production training to ensure```python

  consistent preprocessing and avoid leakageexplainer = shap.TreeExplainer(model)

- Serialize pipelines (pickle/ONNX) and store under `models/`shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample)

## Troubleshooting```

- SHAP memory issues: reduce sample size## Feature Engineering Highlights

- Feature name mismatches: ensure same feature engineering at train

  and inferenceNotebook-derived engineered features used in the pipeline:

- Pickling errors: define classes at module level

- `cabin_multiple` — number of cabins

---- `name_title` — extracted title from passenger name

- `norm_fare` — log- or scaled fare

For more details see the notebook Section 9 (SHAP) and

`titanic_ml/pipeline/train_pipeline.py`.Keep transformations reproducible: implement them as transformers in

the production `Pipeline` (see `titanic_ml/pipeline/train_pipeline.py`).

## Production Notes

- Use the pipeline for production training and inference to avoid data
  leakage (preprocessing fit only on training data).
- Serialize pipelines with `pickle` (or ONNX for cross-platform
  serving) and store in `models/`.

## Visualizations

- `plot_cv_score_distribution()` — boxplots of CV scores to inspect
  stability
- `plot_precision_recall_curve()` — preferred for imbalanced datasets
- `plot_model_comparison()` — horizontal bar chart of model scores

## Troubleshooting

- SHAP memory issues: reduce sample size
- Feature name mismatches: ensure the same `FeatureEngineer` is used
  during training and inference
- Pickling errors: avoid local (nested) class definitions

## Recommended Usage Pattern

Use the notebook for exploration and the pipeline module for training
and deployment. Keep configuration and constants in
`notebooks/utils/config.py` to avoid hardcoded values in cells.

---

For more details see the notebook Section 9 (SHAP) and
`titanic_ml/pipeline/train_pipeline.py`.
