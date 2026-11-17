# Model visualizations

Open [model_visualizations.ipynb](model_visualizations.ipynb) for graphs comparing **eight rating models and five success models**. Saved outputs can be viewed without retraining.

## Refresh graphs

1. Save and **Run All** in the model notebooks you want to compare.
2. Run this notebook from `Data_visualization/` using **`ds6001`**.

If a report is missing or stale, rerun its model notebook first. Reports in `results/` are generated locally and ignored by Git. This notebook does not train models or change CSVs.

## Plot controls

- `MODEL_VARIANT`: `"Selected"`, `"Before overlap trim"`, or `"Original"`.
- `RATING_MODELS` / `SUCCESS_MODELS`: models to display; keep at least one per task.
- `SAVE_FIGURES = True`: export PNGs to the ignored `figures/` folder.

Graphs cover prediction errors, training/test comparisons, confusion matrices, ROC curves, and precision/recall/F1. Lower rating errors are better; high hit precision can come with low recall. Use training validation—not these test plots—to choose settings.

See the [modeling guide](../Modeling_scripts/README.md) for datasets and model locations. [model_reports.py](model_reports.py) handles report export and freshness checks.

Financial plots now use **three classes**: Bust (0), Moderate success (1), Hit (2). Start with macro F1, balanced accuracy, moderate recall, and the 3×3 confusion matrices. ROC/precision–recall curves explicitly evaluate Hit vs rest. Old binary reports are rejected by schema and fingerprint checks.

For the overlap experiment, reports retain **Before overlap trim** alongside **Selected**. Each model notebook shows the validation and VIF comparison; app comparisons expose both versions.


## Shared feature experiment

[history_feature_comparison.ipynb](history_feature_comparison.ipynb) compares tuned lasso and three-class logistic with versus without **all nine director-history features**. Both arms independently tune identical grids using nested chronological training validation. Financial tuning uses the same 70% precision / hit-F1 goal as every financial model. Outer validation scores are reported honestly, including infeasible inner goals; the later test set is not scored. This experiment trains comparison models; the main visualization notebook still only reads saved reports. Neither changes CSVs.


Current full-history experiment (mean outer validation scores):

| Metric | Without director history | With all nine fields |
|---|---:|---:|
| Lasso rating MAE ↓ | 0.5190 | 0.5182 |
| Logistic macro F1 ↑ | 0.3743 | 0.3874 |
| Logistic hit F1 ↑ | 0.2585 | 0.2834 |

Rating gains varied across folds; financial macro/hit F1 improved in all three. Each financial arm met the 70% **inner** precision goal in only one of three outer training windows; the other windows retain flagged original references. The goal was not relaxed, and outer precision is not guaranteed. These are exploratory nested-validation results, not later-test or causal estimates.

Fold-level comparison metadata is saved in [director_history_results.json](director_history_results.json) for review without rerunning the experiment; the large per-movie model reports remain ignored.
