# Modeling guide

There are **two different prediction tasks**. Choose the folder that matches your target.

| Task | Target | Dataset | Start with |
|---|---|---|---|
| Predict a viewer rating | `vote_average` | `model_data_rating.csv` — 13,276 movies | [Simple rating model](rating_models/simple_model.ipynb) |
| Predict hit or bust | `success` (1 or 0) | `model_data_success.csv` — 6,919 movies | [Simple success model](success_models/simple_model.ipynb) |

Use **`ds6001`**, run each notebook from its own subfolder, and run cells from top to bottom. Model notebooks read the saved data; they do not overwrite CSVs.

## Simple or detailed?

- **Simple notebooks:** start with ten predictors and a fixed linear/logistic model. An optional section tries a few small feature sets.
- **Detailed notebooks:** compare more predictors, tune model settings, and inspect prediction errors. Success notebooks also test decision thresholds.

See the [rating guide](rating_models/README.md) or [success guide](success_models/README.md) for notebook lists and results.

## How we compare models

Train on movies released **before November 18, 2017** and test on movies released on or after that date.

| Dataset | Training movies | Test movies |
|---|---:|---:|
| Ratings | 9,405 | 3,871 |
| Success | 5,535 | 1,384 |

Settings and features are selected using three earlier-to-later validation splits **inside the training data**. Missing-value filling, encoding, and scaling are learned from training rows only. Current-movie targets and identifiers are excluded from predictors.

**Ratings:** prefer lower MAE. **Success:** require at least 70% mean validation hit precision, then compare hit F1. The success guide explains the trade-off.

The expanded rating dataset has a different test population from earlier experiments. These data and results have also been inspected repeatedly, so results are exploratory rather than an independent final evaluation.

## Tuning and graphs

Change settings in the **Tuning controls** cell at the top of each notebook, save, and Run All. The final cell refreshes local reports for [model_visualizations.ipynb](../Data_visualization/model_visualizations.ipynb). This combines graphs for every model; see the [visualization guide](../Data_visualization/README.md).
