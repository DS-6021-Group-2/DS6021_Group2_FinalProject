# Model visualizations

Open [model_visualizations.ipynb](model_visualizations.ipynb) to view graphs for **all six rating models and all four success models**. Use `ds6001` and run from this folder.

## Files in this folder

| File / folder | Purpose |
|---|---|
| [model_visualizations.ipynb](model_visualizations.ipynb) | The organized graphs and comparison tables. Start here. |
| [model_reports.py](model_reports.py) | Saves model predictions/metrics and checks report freshness. |
| `results/` | Generated local reports from the model notebooks; ignored by Git. |
| `figures/` | Optional exported PNGs; ignored by Git. |

**To view the existing graphs:** open the notebook and read its saved outputs. No retraining is needed. **To regenerate them:** follow the steps below.

## How to refresh the graphs

1. In each model notebook, edit its **top Tuning controls cell**, save, then **Run All**.
2. The final cell writes a local metrics/prediction report into `results/rating/` or `results/success/`.
3. Run this visualization notebook. It checks that reports match the current model code, datasets, and evaluation rows before plotting.

Reports are local generated files, ignored by Git. A missing/stale report gives instructions to rerun the corresponding model. Notebook graphs remain saved in its outputs for viewing. The visualization notebook does not train models or change dataset CSVs.

## Plot controls

- `MODEL_VARIANT = "Selected"` shows the final validation-selected model; change to `"Original"` for the fixed original model. The simple models follow the same convention.
- `RATING_MODELS` / `SUCCESS_MODELS` choose which notebooks appear in the plots.
- `SAVE_FIGURES = True` also saves PNGs under the ignored `figures/` folder. Default is display-only.

## What the graphs show

| Section | Graphs / use |
|---|---|
| 1. Data | Test rating distribution and hit/bust counts. |
| 2. Ratings | MAE/RMSE comparisons, training/test R², actual versus predicted ratings, and residuals for every model. |
| 3. Success | Test precision/recall/F1, every confusion matrix, ROC/precision–recall curves, and training/test AUC. |
| 4. Validation | Validation precision versus test precision, with the configured validation goal. |
| 5. Configuration | Which model version, features, thresholds, and source notebooks produced the plots. |

Lower rating errors are better. High hit precision can come with low recall; F1 combines both equally. A large training/test gap may indicate overfitting or changes over time. Do not select new settings or thresholds from these test graphs: tuning belongs in training-period validation. Reused selection scores and already-inspected test periods remain exploratory.

`model_reports.py` is the shared export/freshness-check helper. No fitted model objects are serialized. Models and CSVs stay in their existing folders.

## If a report is missing or stale

After cloning or pulling the project, generated reports may not exist locally. Run the model notebooks you want to compare from their own folders, then rerun this notebook. If you only run a subset, update `RATING_MODELS` and `SUCCESS_MODELS` accordingly, keeping at least one model in each list.

Do not edit cached JSON reports to bypass the freshness check. If model code or a dataset changed, save and rerun the corresponding model. Comparisons also require matching test rows and date cutoffs within each task. See the [modeling guide](../Modeling_scripts/README.md) for the two datasets and notebook locations.
