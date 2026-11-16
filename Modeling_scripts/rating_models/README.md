# Predicting viewer ratings

**Target:** TMDB `vote_average`, 0–10. **Data:** [model_data_rating.csv](../../data/model_data_rating.csv), 13,276 movies, including movies with unknown finances.

Use `ds6001`, run notebooks from this folder, and Run All. Source/model dataset CSVs are never overwritten by these notebooks.

## Models

| Notebook | Method |
|---|---|
| [Linear regression](linear_model.ipynb) | Least squares; tunes numeric representation and missingness indicators. |
| [Ridge](ridge_model.ipynb) | L2 shrinkage; tunes alpha, degree and missingness indicators. |
| [Lasso](lasso_model.ipynb) | L1 shrinkage; used by the app. |
| [Elastic net](elastic_net_model.ipynb) | Combined L1/L2 shrinkage. |
| [Random Forest](random_forest.ipynb) | Averaged trees. |
| [Gradient Boosting](gradient_boosting.ipynb) | Sequential trees that reduce remaining loss. |
| [KNN](knn.ipynb) | Partner plain KNN, six compact presets plus the common full-feature reference. |
| [KNN + PCA](knn_pca.ipynb) | Neighbors after standardized dimensionality reduction; includes a no-PCA validation comparison. |

## Tune near the top

- `MODEL_SETTINGS` / `PARAM_GRID`: original reference and candidate settings. Alpha scales differ across penalized estimators.
- `FEATURE_GROUPS_TO_COMPARE`: original cumulative movie/director/actor/company groups.
- `OVERLAP_TRIMS` and tolerance controls: predefined exclusions evaluated after the original selection.
- `SHOW_DIAGNOSTIC_DETAILS`: False shows interpretations; True adds technical tables.
- Keep `CUTOFF_DATE` and `N_VALIDATION_SPLITS` consistent across models.

Three expanding validation folds remain inside the 9,405-movie training period before **November 18, 2017**. All imputation, encoding, scaling, polynomial expansion, PCA and exact-alias removal are fitted within each training fold. Validation MAE chooses candidates. The later 3,871 test movies never choose features or settings.

## Current test results

These are the overlap-reviewed **Selected** versions. **Before overlap trim** preserves the previous selected pipeline separately in reports and app comparisons. Lower MAE/RMSE and higher R² are better.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Training-mean baseline | 0.612 | 0.775 | -0.073 |
| Linear regression | 0.550 | 0.693 | 0.140 |
| Ridge | 0.498 | 0.639 | 0.268 |
| Lasso | 0.496 | 0.633 | 0.283 |
| Elastic net | 0.497 | 0.634 | 0.281 |
| Random Forest | 0.510 | 0.648 | 0.250 |
| Gradient Boosting | 0.504 | 0.641 | 0.266 |
| KNN + PCA | 0.538 | 0.685 | 0.160 |
| KNN | 0.529 | 0.673 | 0.189 |

## Overlap review and interpretation

Each notebook tests dropping genre count while retaining genre flags, then progressively trims overlapping director counts. The original settings candidates are retuned on each representation. Prefer fewer raw predictors within **0.002 validation MAE** of the best candidate; this practical tolerance is not a statistical equivalence test. Polynomial missingness trials remove constant/exactly dependent transformed directions within each training fold. Remaining high VIFs are reported rather than automatically removed.

See the [diagnostics guide](../DIAGNOSTICS.md) for before/after predictor counts, VIFs, validation scores, test scores, and exclusions. The diagnostics explain what findings mean and what to try next. Selection improves simplicity/validation, not necessarily every test score.

`log_budget` remains a predictor; unknown budgets are imputed only from training. Current-movie rating, revenue, financial labels and identifiers are excluded. Director histories use strictly earlier years but retrospective source snapshots do not verify historical availability. Actor-history coverage is incomplete and flagged. Repeated inspection means these results are exploratory and need confirmation on a new period.

The final report cell feeds [model visualizations](../../Data_visualization/model_visualizations.ipynb). Save and Run All after changes, refresh those charts, and explicitly rebuild app artifacts offline when served models change.


## Short notebook layout

Each notebook now has five sections: setup/controls, model, validation/tuning, final results, and diagnostics. The preprocessing/estimator remains visible; [model_workflow.py](../model_workflow.py) supplies shared data checks, folds, scoring and report export. There are no repeated post-selection feature-group or director-history fits. The last cell still saves all reference versions for app/visualization comparisons.

Set `SHOW_EXTRA_COMPARISONS`, `SHOW_COEFFICIENTS` or `SHOW_DIAGNOSTIC_DETAILS` to True near the top for additional tables. Default output shows baseline/selected results, one main plot, and brief diagnostics. KNN retains one short PCA/no-PCA validation comparison. The director-rating experiment lives in [one shared notebook](../../Data_visualization/history_feature_comparison.ipynb).
