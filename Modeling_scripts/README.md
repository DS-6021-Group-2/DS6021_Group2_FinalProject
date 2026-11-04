# Modeling

Models are separated by their prediction target. Both tasks use the feature-engineering output, `../data/model_data.csv`.

| Folder | Target | Task |
|---|---|---|
| [rating_models](rating_models/README.md) | `vote_average` | Predict TMDB viewer ratings with regression. Includes Linear, Random Forest, Ridge, Support Vector, and Gradient Boosting notebooks. |
| [success_models](success_models/README.md) | `success` | Binary financial classification: hit if revenue reaches 2.5 times budget, bust otherwise. Includes Logistic Regression, Random Forest, and Gradient Boosting classifiers. |

Run each notebook from its own subfolder with the `ds6001` kernel. From either subfolder the dataset path is `../../data/model_data.csv`. Exclude both targets and identifiers from predictors. Use chronological splits and training-only preprocessing; compare models within the same task and split.

The current financial task is binary hit/bust, not three-class classification. Regression metrics such as MAE/RMSE belong to ratings; classification metrics belong to success predictions. See each folder's README for details.

Every detailed notebook includes **Section 4: Validation and tuning**. Rating models select by validation MAE (OLS receives validation only); success models use hit F1 with a mean validation precision guard for final selection. Both use training-only chronological folds and show original versus selected results. Section 4b compares feature groups and additionally selects a success threshold using training-period validation. Success candidates must meet the configured validation precision requirement, then are ranked by hit F1. Original and settings-only results remain visible.

For a smaller starting point, each subfolder includes **`simple_model.ipynb`** with nine predictors: linear regression for ratings and logistic regression for success. These simple baselines use the shared chronological split and training-only preprocessing without hyperparameter tuning. Optional Section 5 compares six compact feature presets using training-period validation. The validation/tuning descriptions above apply to the detailed notebooks.
