# Predicting viewer ratings

**Target:** `vote_average`, the TMDB rating on a 0–10 scale.

**Input:** [model_data_rating.csv](../../data/model_data_rating.csv), containing **13,276 movies**. Movies do not need reported budgets or revenues to enter this dataset.

## Where to change tuning settings

Every notebook now starts with **Tuning controls — edit here first**. Change that one settings cell, save, and **Run All**. Later validation sections read those controls instead of resetting them.

- `MODEL_SETTINGS`: the original estimator and settings not searched.
- `PARAM_GRID` (detailed models): lists of trial values; these override the original value during search. OLS searches preprocessing choices; ridge also searches `alpha`.
- `FEATURE_GROUPS_TO_COMPARE`: predictor groups to compare.
- `CUTOFF_DATE` and `N_VALIDATION_SPLITS`: keep consistent across models.

The last cell saves a checked report for the [visualization notebook](../../Data_visualization/model_visualizations.ipynb). See its [guide](../../Data_visualization/README.md) for refreshing graphs.

## What to run

Start with [linear_model.ipynb](linear_model.ipynb). Use `ds6001`, run from this folder, and run cells from top to bottom.

| Notebook | What it tries |
|---|---|
| [Linear model](linear_model.ipynb) | Linear regression with the larger feature set. |
| [Ridge](ridge_model.ipynb) | Coefficient shrinkage, with optional quadratic numeric features. |
| [Lasso](lasso_model.ipynb) | L1 shrinkage, optional quadratic terms, and validation-based feature selection. |
| [Elastic net](elastic_net_model.ipynb) | Combined L1/L2 shrinkage; jointly tunes alpha, mixing weight, degree, and feature groups. |
| [KNN with PCA](knn_pca.ipynb) | Standardized inputs compressed with PCA, then nearest-neighbor prediction; joint chronological tuning. |
| [Random Forest](random_forest.ipynb) | An average of many decision trees. |
| [Gradient Boosting](gradient_boosting.ipynb) | Trees that successively correct earlier errors. |

## Reading the scores

| Score | Meaning | Better direction |
|---|---|---|
| **MAE** | Average size of the rating error. MAE 0.5 means about half a rating point off on average. | Lower |
| **RMSE** | Rating error with a larger penalty for big mistakes. | Lower |
| **R²** | How much rating variation the model explains relative to predicting the evaluation-set mean. It can be negative. | Higher |

The baseline predicts **the training mean rating for every movie**. Beat that before claiming the features help.

## Current test results

All rows use the same **3,871 later movies**. Model rows show the validation-selected model.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Training-mean baseline | 0.612 | 0.775 | −0.073 |
| Detailed linear | 0.553 | 0.696 | 0.133 |
| Ridge, tuned quadratic features | 0.498 | 0.637 | 0.275 |
| Lasso, tuned quadratic features | 0.494 | 0.631 | 0.289 |
| Elastic net, tuned quadratic features | 0.495 | 0.632 | 0.286 |
| KNN + PCA | 0.537 | 0.685 | 0.161 |
| Random Forest | 0.511 | 0.648 | 0.248 |
| Gradient Boosting | 0.504 | 0.640 | 0.267 |

All models now include `log_budget`. Unknown budgets are median-imputed inside training only. Adding budget produced small, mixed rating changes on the same test movies; it did not consistently improve prediction. Results remain exploratory.

## Finding things in the notebooks

| Detailed-notebook section | Purpose |
|---|---|
| 4 | Run the search using the top `PARAM_GRID` and `TUNING_METRIC`; default selection is validation MAE. Ordinary linear regression has no regularization setting to tune. |
| 4b | Compare feature groups using training-period validation; linear/ridge jointly compare every configured setting. |
| 5–6 | Compare baseline/original/selected results and inspect prediction errors. |
| 7–8 | Compare feature groups and isolate the effect of earlier director ratings. |


## Keep in mind

Training ends before **November 18, 2017**: 9,405 training movies and 3,871 test movies. Preprocessing is fitted within training. Current `vote_average` is the target, never an input.

Earlier ratings are observed snapshots, not a complete director filmography or verified historical release-time ratings. Early actor-history coverage is incomplete and flagged. History updates through time, so these are rolling predictions. The expanded test population and repeated inspection mean the results remain exploratory; do not compare them directly with old scores on only financially eligible movies.

For KNN + PCA, edit `PCA_SETTINGS` and `PARAM_GRID` near the top. Section 7 shows retained components and a fixed-settings validation comparison against KNN without PCA. No CSVs are changed.

KNN + PCA selected 35 distance-weighted neighbors and 95% retained input variance (42 components). Its test MAE is higher than tuned ridge. At fixed original settings, PCA slightly worsened validation MAE: 0.554 without PCA versus 0.558 with PCA.

## Linear and ridge tuning results

Both notebooks now search numeric degree 1/2 and missing-value indicators; ridge also searches `alpha` from 0.01 to 10,000. Each candidate refits all preprocessing inside three chronological validation folds. Section 4b jointly ranks settings and feature groups by validation MAE. Ordinary linear regression has no shrinkage parameter; degree 2 changes the feature representation. See [scikit-learn's explanation](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions).

| Model | Test MAE before → after | New RMSE | New R² | Goals met |
|---|---:|---:|---:|---|
| Linear | 0.553 → 0.553 | 0.696 | 0.133 | None |
| Ridge | 0.552 → 0.498 | 0.637 | 0.275 | MAE ≤0.50; RMSE ≤0.65 |

Linear retained original settings and movie/director features. Ridge selected **alpha=100**, **quadratic numeric features**, **no missingness indicators**, and movie/director/actor features (38 source predictors, 130 encoded/expanded inputs). The original plain fits remain visible as baselines.

Ridge has not reached R² ≥0.30 or either stretch goal. Its mean validation MAE improved only slightly (0.5175 for the original degree-1 ridge/director candidate versus 0.5169 selected), so the test gain needs confirmation on a new period. Test rows were not used for selection; no additional search was launched after seeing their scores. Dataset CSVs and the date cutoff are unchanged. Section 5 now displays an explicit goal checklist.

## Lasso and elastic net

[Lasso](lasso_model.ipynb) and [elastic net](elastic_net_model.ipynb) use the same cohort, 2017-11-18 cutoff, three chronological folds, and preprocessing structure as ridge. Both search degree 1/2, alpha `[0.001, 0.003, 0.01, 0.03, 0.1]`, and four feature groups; elastic net additionally searches l1_ratio `[0.05, 0.2, 0.5, 0.8]`. Selection uses validation MAE. Convergence warnings raise errors; each notebook shows coefficients, convergence diagnostics, and verified model comparisons. No success models or dataset CSVs were changed.

Both selected **quadratic numeric features**, **alpha=0.003**, and movie/director/actor features. Elastic net selected **l1_ratio=0.8**. Alpha is defined differently from ridge and should not be compared numerically across these estimators.

| Model | Mean validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Ordinary linear regression | 0.5178 | 0.5527 | 0.6959 | 0.1335 |
| Ridge | 0.5169 | 0.4979 | 0.6366 | 0.2750 |
| Lasso | 0.5086 | 0.4943 | 0.6305 | 0.2887 |
| Elastic net | 0.5093 | 0.4949 | 0.6316 | 0.2864 |

**Finding:** lasso has the lowest recorded test MAE among the retained rating models. Elastic net slightly beats ridge but does not beat lasso. These small differences on an already-inspected period need independent confirmation. Both new models meet MAE ≤0.50 and RMSE ≤0.65, but neither reaches R² ≥0.30. No extra search was launched in response to test scores. The app continues to use its existing verified ridge pipeline; both new models appear in comparisons.
