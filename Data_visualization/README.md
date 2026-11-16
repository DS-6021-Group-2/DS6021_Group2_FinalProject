# Model visualizations

Open [model_visualizations.ipynb](model_visualizations.ipynb) for graphs comparing **seven rating models and four success models**. Saved outputs can be viewed without retraining.

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

[history_feature_comparison.ipynb](history_feature_comparison.ipynb) is the single director-rating-history experiment formerly repeated in multiple rating notebooks. It fits fixed-settings Ridge with versus without three earlier director-rating features using chronological **training validation only**. It intentionally trains those small comparison models; the main `model_visualizations.ipynb` remains read-only and does not retrain models. Neither notebook changes CSVs.
