# Predicting financial performance

**Target:** `success` in [model_data_success.csv](../../data/model_data_success.csv), with 6,919 financially eligible movies.

| Code | Category | Revenue / production budget | Movies |
|---|---|---|---:|
| 0 | Bust | ≤1 | 2,475 |
| 1 | Moderate success | >1 and <2.5 | 1,940 |
| 2 | Hit | ≥2.5 | 2,504 |

Missing finances remain unavailable. These categories do not measure audited profit. Raw source data and all predictors are preserved; the cleaned and success-model tables have updated target labels. Earlier director success features still mean **earlier hit fractions**, not averages of category codes.

## Models

- [Logistic regression](logistic_regression.ipynb): regularized three-class linear classifier; used in the app.
- [Random Forest](random_forest.ipynb): averaged classification trees.
- [Gradient Boosting](gradient_boosting.ipynb): sequential trees.
- [KNN + PCA](knn_pca.ipynb): classification by neighboring movies after dimensionality reduction; includes a no-PCA validation comparison.

Use `ds6001`, run from this folder, and Run All. No classifier notebook modifies dataset CSVs.

## Tune near the top

| Control | Purpose |
|---|---|
| `MODEL_SETTINGS` | Fixed original reference settings. |
| `PARAM_GRID` | Candidate estimator/PCA settings. |
| `FEATURE_GROUPS_TO_COMPARE` | Movie characteristics, then director, actor and company/franchise histories. |
| `THRESHOLDS`, `MIN_VALIDATION_PRECISION` | Candidate hit thresholds and the mean validation hit precision requirement (0.70). |
| `SHOW_DIAGNOSTIC_DETAILS` | False: plain-English diagnostics; True: all technical tables. |

Train before **November 18, 2017** (5,535 movies); test on or after it (1,384). Three expanding chronological folds stay inside training. Preprocessing is fitted anew within each fold. Current ratings/financial labels, IDs, revenue and revenue-to-budget ratio are excluded from predictors; log budget is allowed.

## How a label is selected

Each classifier predicts probabilities for all three classes. The estimator grid uses ordinary argmax and hit F1 (class 2). Feature/settings/threshold selection then requires ≥70% **mean validation** hit precision and maximizes hit F1 among qualifying candidates. If none qualifies, the original reference is retained and marked infeasible.

The selected rule chooses Hit if P(Hit) reaches the selected threshold; otherwise it chooses the larger Bust/Moderate probability. The original and settings-only references use a 0.50 hit threshold. Probabilities are unchanged by that decision rule. The validation requirement does not guarantee later test precision.

## Read these results together

- **Macro F1:** equally weights Bust, Moderate and Hit F1.
- **Balanced accuracy:** average of the three recalls.
- **Moderate recall:** how many actual moderate movies are found.
- **Hit precision / recall / F1:** quality and coverage of predicted hits.
- **ROC AUC:** Hit vs rest; **Multiclass ROC AUC:** macro average of three one-vs-rest AUCs.
- **3×3 confusion matrix:** which categories get confused.

Former binary scores cannot be compared directly with this three-class experiment. See the saved notebook tables or the app for current results. Diagnostics explain each finding and suggest candidate experiments in [the guide](../DIAGNOSTICS.md). Test results remain exploratory after repeated inspection; use training validation for further changes.

Overlap review now tests predefined genre/history trims inside training validation, retuning existing candidate settings. Top controls expose the presets and tolerances. **Before overlap trim** preserves the previous selected model; **Validation-selected** includes the final review. See the [diagnostics guide](../DIAGNOSTICS.md) for rules and before/after results. Dataset columns are unchanged.


## Short notebook layout

Each notebook now has five sections: setup/controls, model, validation/tuning, final results, and diagnostics. The preprocessing/estimator remains visible; [model_workflow.py](../model_workflow.py) supplies shared data checks, folds, scoring and report export. There are no repeated post-selection feature-group or director-history fits. The last cell still saves all reference versions for app/visualization comparisons.

Set `SHOW_EXTRA_COMPARISONS`, `SHOW_COEFFICIENTS` or `SHOW_DIAGNOSTIC_DETAILS` to True near the top for additional tables. Default output shows baseline/selected results, one main plot, and brief diagnostics. KNN retains one short PCA/no-PCA validation comparison. The director-rating experiment lives in [one shared notebook](../../Data_visualization/history_feature_comparison.ipynb).
