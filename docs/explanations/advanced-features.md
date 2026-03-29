# Advanced Features

Advanced capabilities beyond the baseline ML pipeline.

This page summarizes the notebook's advanced techniques: explainability
with SHAP, enhanced visualizations, and production-ready pipeline tips.

## SHAP Explainability

SHAP (SHapley Additive exPlanations) attributes model predictions to
features. For tree-based models use `TreeExplainer` (fast). For other
models `KernelExplainer` can be used but is much slower.

Key visualizations:

- **Summary plot** — global feature importance
- **Dependence plots** — relationship between a feature and its SHAP value
- **Waterfall/force plots** — single-prediction breakdown

Best practices:

- For large datasets sample 100-500 rows for SHAP summaries
- Focus on the positive class for binary classification (Survived=1)
- Use TreeExplainer for tree-based models

Example:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
```

## Feature Engineering Highlights

Notebook-derived engineered features used in the pipeline:

- `cabin_multiple` — number of cabins
- `name_title` — extracted title from passenger name
- `norm_fare` — log- or scaled fare

Keep transformations reproducible: implement them as transformers in
the production `Pipeline` (see `src/titanic_ml/pipeline/train_pipeline.py`).

## Visualizations

- `plot_cv_score_distribution()` — boxplots of CV scores to inspect stability
- `plot_precision_recall_curve()` — preferred for imbalanced datasets
- `plot_model_comparison()` — horizontal bar chart of model scores

## Production Notes

- Use the pipeline for production training and inference to avoid data
  leakage (preprocessing fit only on training data)
- Serialize pipelines with `pickle` (or ONNX for cross-platform
  serving) and store in `saved_models/`

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
`src/titanic_ml/pipeline/train_pipeline.py`.
