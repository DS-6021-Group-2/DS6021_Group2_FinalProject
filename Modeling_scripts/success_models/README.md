# Predicting hits and busts

**Target:** `success` — **1 = hit**, **0 = bust**. A hit has reported revenue of at least **2.5 times production budget**; a bust is below that benchmark. These are project labels, not audited profit/loss.

**Input:** [model_data_success.csv](../../data/model_data_success.csv), containing **6,919 movies with usable financial data**.

## Where to change tuning settings

Every notebook now starts with **Tuning controls — edit here first**. Change that one settings cell, save, and **Run All**. Later validation sections read those controls instead of resetting them.

- `MODEL_SETTINGS`: the original estimator and settings not searched.
- `PARAM_GRID` (detailed models): lists of trial values; these override the original value during search. OLS has an empty grid.
- `FEATURE_GROUPS_TO_COMPARE` (detailed) or `FEATURE_SETS` (simple): predictor groups to compare.
- `MIN_VALIDATION_PRECISION` and `THRESHOLDS`: the precision goal and threshold candidates.
- `CUTOFF_DATE` and `N_VALIDATION_SPLITS`: keep consistent across models.

The last cell saves a checked report for the [visualization notebook](../../Data_visualization/model_visualizations.ipynb). See its [guide](../../Data_visualization/README.md) for refreshing graphs.

## What to run

Start with [simple_model.ipynb](simple_model.ipynb). Use `ds6001`, run from this folder, and run cells from top to bottom.

| Notebook | What it tries |
|---|---|
| [Simple model](simple_model.ipynb) | Nine-predictor logistic baseline, then small feature/threshold comparisons. |
| [Logistic regression](logistic_regression.ipynb) | A regularized linear classifier. |
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

All four candidates met the **validation** precision requirement. The table shows their performance on **1,384 later test movies**; reaching 70% in validation does not guarantee 70% in testing. The simple row uses its Section 5 selection, not its original fixed-threshold baseline.

| Model | Hit threshold | Test precision | Test recall | Test hit F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.80 | **74.3%** | 11.6% | 0.200 |
| Random Forest | 0.50 | 51.2% | 34.7% | 0.414 |
| Gradient Boosting | 0.65 | 61.7% | 21.1% | 0.314 |
| Simple model | 0.70 | 61.9% | 13.7% | 0.224 |

**Interpretation:** logistic regression predicted hits more cautiously. It reached 74.3% test precision but found only **55 of 475 actual hits**. Higher precision came with low recall and F1. No model is strongest on every metric.

## Finding things in the notebooks

| Detailed-notebook section | Purpose |
|---|---|
| 4 | Run the search using the top controls; compare validation hit F1. |
| 4b | Apply the precision goal and threshold list from the top controls, then select by F1. |
| 5–6 | Read original/selected results, confusion matrices, and prediction examples. |
| 7 | Compare feature groups at the original settings. |

The simple notebook keeps its original baseline in Sections 1–4 and tries small presets/thresholds in Section 5. If no candidate meets 70% validation precision, notebooks report that and keep a clearly labeled original reference; they do not quietly lower the target.

## Keep in mind

Train before **November 18, 2017** (5,535 movies); test on or after it (1,384 movies). Select settings/features/thresholds using three chronological validation splits inside training, with training-only preprocessing.

Exclude both `success` and `vote_average`, movie identifiers, budget, and revenue from predictors. Earlier histories are incomplete and retrospective. The 70% guard applies to the mean across validation periods, not each period or future movies. Repeated inspection makes these exploratory results, not a future precision guarantee.
