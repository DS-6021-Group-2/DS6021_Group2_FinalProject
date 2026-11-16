# Modeling guide

There are **two different prediction tasks**. Choose the folder that matches your target.

| Task | Target | Dataset | Start with |
|---|---|---|---|
| Predict a viewer rating | `vote_average` | `model_data_rating.csv` — 13,276 movies | [Linear regression](rating_models/linear_model.ipynb) |
| Predict bust / moderate / hit | `success` (0, 1, 2) | `model_data_success.csv` — 6,919 movies | [Logistic regression](success_models/logistic_regression.ipynb) |

Use **`ds6001`**, run each notebook from its own subfolder, and run cells from top to bottom. Model notebooks read the saved data; they do not overwrite CSVs.

## Five-section notebooks

Every model now uses the same layout:

1. **Setup and controls:** edit model/search settings and optional output switches.
2. **Model:** visible preprocessing and estimator definition.
3. **Validation and tuning:** one shared call handles settings, features, thresholds and overlap review.
4. **Final results:** baseline versus selected training/test scores and one main plot. KNN also shows one short PCA/no-PCA validation comparison.
5. **Diagnostics:** brief safeguards, before/after VIF, limitations and report export.

Repeated fold construction, scoring and reporting live in [model_workflow.py](helpers/model_workflow.py). The model itself stays in each notebook. All original/settings-only/before-trim/selected versions remain in saved reports and the app.

| Top switch | Default | Shows |
|---|---|---|
| `SHOW_EXTRA_COMPARISONS` | False | Intermediate models, candidate details and rating goal checks. |
| `SHOW_COEFFICIENTS` | False | Largest coefficients for linear/logistic estimators. |
| `SHOW_DIAGNOSTIC_DETAILS` | False | Full diagnostic tables. |

The repeated director-rating-history experiment is now one [shared notebook](../Data_visualization/history_feature_comparison.ipynb). All-model charts remain in [model visualizations](../Data_visualization/model_visualizations.ipynb). See the [rating](rating_models/README.md) or [success](success_models/README.md) guide for model details.

## How we compare models

Train on movies released **before November 18, 2017** and test on movies released on or after that date.

| Dataset | Training movies | Test movies |
|---|---:|---:|
| Ratings | 9,405 | 3,871 |
| Success | 5,535 | 1,384 |

Settings and features are selected using three earlier-to-later validation splits **inside the training data**. Missing-value filling, encoding, and scaling are learned from training rows only. Current-movie targets and identifiers are excluded from predictors.

**Ratings:** prefer lower MAE. **Success:** require at least 70% mean validation hit precision, then compare hit F1; also report three-class macro F1 and moderate recall. The success guide explains the trade-off.

The expanded rating dataset has a different test population from earlier experiments. These data and results have also been inspected repeatedly, so results are exploratory rather than an independent final evaluation.

## Tuning and graphs

Change settings in the **Tuning controls** cell at the top of each notebook, save, and Run All. The final cell refreshes local reports for [model_visualizations.ipynb](../Data_visualization/model_visualizations.ipynb). This combines graphs for every model; see the [visualization guide](../Data_visualization/README.md).

## Diagnostics

Every notebook includes training-matrix multicollinearity checks and model-appropriate quality checks before report export. See the [diagnostics guide](DIAGNOSTICS.md) for VIF, numerical stability, leakage safeguards, residuals, class balance, probability quality, and interpretation limits. These checks do not change models automatically.

Overlap review now tests predefined genre/history trims inside training validation, retuning existing candidate settings. Top controls expose the presets and tolerances. **Before overlap trim** preserves the previous selected model; **Validation-selected** includes the final review. See the [diagnostics guide](DIAGNOSTICS.md) for rules and before/after results. Dataset columns are unchanged.


## Verify the workflow

`/opt/miniconda3/envs/ds6001/bin/python tests/verify_model_workflow.py` checks repeated-date fold separation, invalid source/target rejection, and that changing only test outcomes cannot change selection or fitted predictions. The simplification was also checked against every pre-refactor model variant's features, thresholds, scores and held-out predictions. CSVs and unrelated project edits were preserved.


Shared dates, folds, task objectives and metric goals live in [project_policy.py](helpers/project_policy.py). Every active notebook imports them; changing a model's search settings does not silently change its objective. Check with `/opt/miniconda3/envs/ds6001/bin/python tests/verify_project_consistency.py --artifacts`.

The shared [director-history experiment](../Data_visualization/history_feature_comparison.ipynb) now compares tuned lasso and three-class logistic with versus without **all nine** director-history fields. It uses nested chronological validation entirely inside training; the later test set is not scored.

Shared Python code lives in [helpers/](helpers/README.md). Verification scripts live in [tests/](../tests/README.md).
