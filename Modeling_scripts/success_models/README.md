# Predicting hits and busts

**Target:** `success` — **1 = hit**, **0 = bust**. A hit has reported revenue of at least **2.5 times production budget**; a bust is below that benchmark. These are project labels, not audited profit/loss.

**Input:** [model_data_success.csv](../../data/model_data_success.csv), containing **6,919 movies with usable financial data**.

## Where to change tuning settings

Every notebook now starts with **Tuning controls — edit here first**. Change that one settings cell, save, and **Run All**. Later validation sections read those controls instead of resetting them.

- `MODEL_SETTINGS`: the original estimator and settings not searched.
- `PARAM_GRID` (detailed models): lists of trial values; these override the original value during search.
- `FEATURE_GROUPS_TO_COMPARE`: predictor groups to compare.
- `MIN_VALIDATION_PRECISION` and `THRESHOLDS`: the precision goal and threshold candidates.
- `CUTOFF_DATE` and `N_VALIDATION_SPLITS`: keep consistent across models.

The last cell saves a checked report for the [visualization notebook](../../Data_visualization/model_visualizations.ipynb). See its [guide](../../Data_visualization/README.md) for refreshing graphs.

## What to run

Start with [logistic_regression.ipynb](logistic_regression.ipynb). Use `ds6001`, run from this folder, and run cells from top to bottom.

| Notebook | What it tries |
|---|---|
| [Logistic regression](logistic_regression.ipynb) | A regularized linear classifier. |
| [KNN with PCA](knn_pca.ipynb) | Standardized inputs compressed with PCA, then nearest-neighbor prediction; joint chronological tuning. |
| [Random Forest](random_forest.ipynb) | Many decision trees combined. |
| [Gradient Boosting](gradient_boosting.ipynb) | Trees that successively correct earlier errors. |

## Our selection goal

**First require at least 70% mean validation hit precision; then choose the highest hit F1 among qualifying candidates.**

| Metric | Plain-English meaning |
|---|---|
| **Hit precision** | Of movies predicted to be hits, how many actually are hits? |
| **Hit recall** | Of actual hits, how many did we find? |
| **Hit F1** | Combines hit precision and recall, giving both equal weight. |
| Accuracy | Percentage of all movies classified correctly. |
| Balanced accuracy | Average recall for hits and busts, giving both classes equal weight. |

F1 itself does not emphasize precision; the separate 70% requirement does. Always predicting bust already gives **65.7% test accuracy** but finds no hits, so accuracy alone is not enough.

## Current test results

All four candidates met the **validation** precision requirement. The table shows their performance on **1,384 later test movies**; reaching 70% in validation does not guarantee 70% in testing.

| Model | Hit threshold | Test precision | Test recall | Test hit F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.65 | 64.8% | 26.7% | 0.379 |
| Random Forest | 0.55 | 60.3% | 27.8% | 0.380 |
| Gradient Boosting | 0.60 | 55.4% | 29.1% | 0.381 |
| KNN + PCA | 0.65 | 52.4% | 16.0% | 0.245 |

**Budget update:** all models now include `log_budget`. After rerunning the same validation procedure, logistic test hit F1 rose from 0.200 to 0.379, gradient boosting from 0.314 to 0.381; random forest fell from 0.414 to 0.380. Selection can change settings, features, and thresholds, so these are comparisons of the full procedures. At fixed original settings and threshold 0.5, adding budget increased logistic F1 from 0.500 to 0.531. None of the selected models reached 70% precision on the later test data.

## Finding things in the notebooks

| Detailed-notebook section | Purpose |
|---|---|
| 4 | Run the search using the top controls; compare validation hit F1. |
| 4b | Apply the precision goal and threshold list from the top controls, then select by F1. |
| 5–6 | Read original/selected results, confusion matrices, and prediction examples. |
| 7 | Compare feature groups at the original settings. |

If no candidate meets 70% validation precision, notebooks report that and keep a clearly labeled original reference; they do not quietly lower the target.

## Keep in mind

Train before **November 18, 2017** (5,535 movies); test on or after it (1,384 movies). Select settings/features/thresholds using three chronological validation splits inside training, with training-only preprocessing.

Exclude both `success` and `vote_average`, movie identifiers, raw budget, and revenue from predictors. Earlier histories are incomplete and retrospective. The 70% guard applies to the mean across validation periods, not each period or future movies. Repeated inspection makes these exploratory results, not a future precision guarantee.

For KNN + PCA, edit `PCA_SETTINGS` and `PARAM_GRID` near the top. Section 7 shows retained components and a fixed-settings validation comparison against KNN without PCA. No CSVs are changed.

KNN + PCA selected 15 distance-weighted neighbors and 95% retained input variance (42 components). It met 71.3% mean validation precision, but reached only 52.4% test precision. Its F1 trails the other models. At fixed original settings, PCA slightly worsened validation F1: 0.254 without PCA versus 0.249 with PCA.
