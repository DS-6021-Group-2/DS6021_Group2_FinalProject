# Model visualizations

Open [model_visualizations.ipynb](model_visualizations.ipynb) for graphs comparing **seven rating models and five success models**. Saved outputs can be viewed without retraining.

## Refresh graphs

1. Save and **Run All** in the model notebooks you want to compare.
2. Run this notebook from `Data_visualization/` using **`ds6001`**.

If a report is missing or stale, rerun its model notebook first. Reports in `results/` are generated locally and ignored by Git. This notebook does not train models or change CSVs.

## Plot controls

- `MODEL_VARIANT`: `"Selected"` or `"Original"`.
- `RATING_MODELS` / `SUCCESS_MODELS`: models to display; keep at least one per task.
- `SAVE_FIGURES = True`: export PNGs to the ignored `figures/` folder.

Graphs cover prediction errors, training/test comparisons, confusion matrices, ROC curves, and precision/recall/F1. Lower rating errors are better; high hit precision can come with low recall. Use training validation—not these test plots—to choose settings.

See the [modeling guide](../Modeling_scripts/README.md) for datasets and model locations. [model_reports.py](model_reports.py) handles report export and freshness checks.
