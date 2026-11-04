# Predicting viewer ratings

**Target:** `vote_average`, the TMDB rating on a 0–10 scale.

**Input:** [model_data_rating.csv](../../data/model_data_rating.csv), containing **13,276 movies**. Movies do not need reported budgets or revenues to enter this dataset.

## Where to change tuning settings

Every notebook now starts with **Tuning controls — edit here first**. Change that one settings cell, save, and **Run All**. Later validation sections read those controls instead of resetting them.

- `MODEL_SETTINGS`: the original estimator and settings not searched.
- `PARAM_GRID` (detailed models): lists of trial values; these override the original value during search. OLS has an empty grid.
- `FEATURE_GROUPS_TO_COMPARE` (detailed) or `FEATURE_SETS` (simple): predictor groups to compare.
- `CUTOFF_DATE` and `N_VALIDATION_SPLITS`: keep consistent across models.

The last cell saves a checked report for the [visualization notebook](../../Data_visualization/model_visualizations.ipynb). See its [guide](../../Data_visualization/README.md) for refreshing graphs.

## What to run

Start with [simple_model.ipynb](simple_model.ipynb). Use `ds6001`, run from this folder, and run cells from top to bottom.

| Notebook | What it tries |
|---|---|
| [Simple model](simple_model.ipynb) | Nine predictors and linear regression; optional small feature comparisons. |
| [Linear model](linear_model.ipynb) | Linear regression with the larger feature set. |
| [Ridge](ridge_model.ipynb) | Linear regression with coefficient shrinkage. |
| [Support Vector Regression](support_vector.ipynb) | Nonlinear relationships between features and ratings. |
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

All rows use the same **3,871 later movies**. Detailed-model rows show the validation-selected model; the simple row below shows its original fixed nine-predictor baseline.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Training-mean baseline | 0.612 | 0.775 | −0.073 |
| Simple linear, nine predictors | 0.551 | 0.704 | 0.114 |
| Detailed linear | 0.552 | 0.695 | 0.136 |
| Ridge | 0.551 | 0.694 | 0.139 |
| Support Vector Regression | **0.475** | **0.607** | **0.341** |
| Random Forest | 0.508 | 0.646 | 0.254 |
| Gradient Boosting | 0.507 | 0.644 | 0.259 |

**Interpretation:** SVR currently has the lowest test error, about 0.475 rating points. This is an observed comparison, not proof it will always be the best model. The simple notebook's optional genre preset had better validation MAE but worse test MAE (0.562); the original baseline is retained.

## Finding things in the notebooks

| Detailed-notebook section | Purpose |
|---|---|
| 4 | Run the search using the top `PARAM_GRID` and `TUNING_METRIC`; default selection is validation MAE. Ordinary linear regression has no regularization setting to tune. |
| 4b | Compare feature groups using training-period validation. |
| 5–6 | Compare baseline/original/selected results and inspect prediction errors. |
| 7–8 | Compare feature groups and isolate the effect of earlier director ratings. |

The simple notebook has four baseline sections and an optional feature comparison in Section 5.

## Keep in mind

Training ends before **November 18, 2017**: 9,405 training movies and 3,871 test movies. Preprocessing is fitted within training. Current `vote_average` is the target, never an input.

Earlier ratings are observed snapshots, not a complete director filmography or verified historical release-time ratings. Early actor-history coverage is incomplete and flagged. History updates through time, so these are rolling predictions. The expanded test population and repeated inspection mean the results remain exploratory; do not compare them directly with old scores on only financially eligible movies.
