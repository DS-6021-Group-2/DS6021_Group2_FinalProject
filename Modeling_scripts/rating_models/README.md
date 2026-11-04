# Rating models

## Start with the simple model

`simple_model.ipynb` fits linear regression using only **9 predictors**: runtime, director film experience, smoothed earlier director rating, rated-film support count, certification, release month, sequel status, English-language status, and a no-rating-history flag. It uses the same chronological train/test rows and training-only preprocessing, but no tuning, feature search, or repeated validation sections. Compare MAE/RMSE/R² with the training-mean baseline. The other notebooks remain available for more detailed comparisons.

These notebooks predict TMDB viewer rating from the same engineered `model_data.csv`.

| Notebook | Model | Initial settings |
|---|---|---|
| `linear_model.ipynb` | Ordinary linear regression | No regularization |
| `random_forest.ipynb` | Random Forest | Initial forest plus training-period tuning of depth and leaf size; 100 trees, seed 42 |
| `ridge_model.ipynb` | Ridge regression | alpha 10 |
| `support_vector.ipynb` | Support Vector Regression | RBF kernel, C 1, epsilon 0.1 |
| `gradient_boosting.ipynb` | Gradient Boosting | 150 trees, learning rate 0.05, depth 3, minimum leaf size 5, seed 42 |

Run notebooks independently from this folder using `ds6001`. Each is self-contained and uses the same data split, preprocessing rules, metrics, and four cumulative feature groups. Section 4 in each detailed notebook validates or tunes settings using training-period validation MAE. OLS has no regularization hyperparameter and receives validation only; Ridge, SVR, Random Forest, and Gradient Boosting perform compact grid searches. There is no automatic selection of a winning model across notebooks.

`linear_model.ipynb` trains ordinary least-squares linear regression to predict TMDB `vote_average` from `../../data/model_data.csv`. Run all cells from this folder using the `ds6001` kernel. It trains in memory and saves results in notebook outputs; it does not modify CSVs or export model files.

The notebook uses approximately the earliest 80% of movies for training and the latest 20% for evaluation, keeping equal release dates in one split. Numeric imputation/scaling and categorical imputation/one-hot encoding are fitted on training data only. Both outcome columns (`success`, `vote_average`) and movie identifiers/full dates are excluded from predictors; release year remains numeric and month categorical.

Results compare training/test linear regression with a training-mean rating baseline using MAE, RMSE, and R². Prediction examples and residual plots help inspect errors. These are exploratory results on a financially eligible sample, not a final independent evaluation or a comparison proving the feature revisions help.

History uses earlier-year outcomes updated through time, representing rolling predictions; freeze history at the cutoff for a single forecast made at that date. Binary financial classification uses `success` and belongs in `../success_models/`; clustering and the app are separate future steps. Keep model comparisons and tuning inside training-period forward validation.

The simple model's exploratory later-test MAE is **0.5430**, RMSE **0.6871**, and R² **0.0702**; the training-mean baseline MAE is 0.6197.

## Feature comparison

Section 7 compares movie characteristics, then adds director history, actor history, and company/franchise history. Within each notebook, each version uses the same estimator and three expanding chronological validation splits entirely within the earlier training period, with fresh preprocessing per split. Tables show mean validation MAE/RMSE, variation across periods, and each fold's MAE. Positive MAE improvement versus movie-only indicates a lower error. The comparison is cumulative and does not isolate the effect of individual transformations or establish causal importance. It does not select or retune the previously evaluated test model.

## Initial results

All-feature results from the shared split and three training-period validation windows (MAE in rating points; lower is better). This table preserves the original fixed-setting trials. The notebooks now show initial and validation-selected results separately.

| Model | Mean validation MAE | Later test MAE |
|---|---:|---:|
| Linear regression | 0.5172 | 0.5430 |
| Random Forest | 0.5070 | 0.5030 |
| Ridge regression | 0.5162 | 0.5424 |
| Support Vector Regression | 0.5328 | 0.5013 |
| Gradient Boosting | 0.5039 | 0.5055 |

The training-mean baseline test MAE is 0.6197. Validation and test rankings differ; use training-period validation for model/settings decisions. Test results are exploratory because this period has already been inspected. See each notebook for training error, RMSE, R², residuals, and feature-group results.

## Random Forest complexity tuning

The forest notebook compares its original unrestricted-depth, leaf-size-3 model with nine combinations: depth 6, 12, or unrestricted; minimum leaf size 3, 10, or 25. Grid search uses three expanding chronological splits inside the earlier training period and refits all preprocessing per fold. The lowest mean validation MAE selects a setting, followed by a full-training refit. Tables show validation MAE/R² and the initial/tuned training and test metrics to distinguish reducing the gap from improving predictions. The feature-group section retains initial settings for comparability; it does not reuse the search to make independent feature-selection claims. Earlier results in the initial-results table remain the untuned trials.

The first search selected depth 12 and minimum leaf size 10. Mean validation MAE improved from 0.5070 to 0.5036. Training R² decreased from 0.7928 to 0.5227 and the training–test R² gap narrowed, but test MAE worsened from 0.5030 to 0.5111 and test R² decreased from 0.1963 to 0.1847. This reduced training fit without improving the inspected later test period; both initial and tuned results remain visible.

## Validation/tuning section (Section 4)

Each notebook exposes `PARAM_GRID` and `TUNING_METRIC`. Three expanding date-based folds stay entirely within the outer training period; each fit includes the preprocessing pipeline. Rating models select the lowest mean validation MAE and also show validation RMSE/R² and training scores. The search refits its chosen pipeline on all outer training data. Section 5 compares original and selected training/test results; Section 7 keeps the original model settings for cumulative feature comparisons. Search scores are not independent estimates after selection. Do not repeatedly change grids based on the already-inspected test period.

| Notebook | Search |
|---|---|
| Linear | Fixed OLS validation; no artificial hyperparameter search |
| Ridge | alpha 0.1, 1, 10, 100 |
| SVR | C 0.5, 1, 5; gamma scale or 0.01 |
| Random Forest | depth 6, 12, unrestricted; leaf size 3, 10, 25 |
| Gradient Boosting | 100 or 150 trees; depth 2 or 3 |

## Settings-only selection results (Section 4)

Settings below were selected by the stated training-period metric, not by test scores. The test period was already inspected and remains exploratory.

| Model | Selected settings | Initial test MAE | Selected test MAE |
|---|---|---:|---:|
| Linear regression | Fixed OLS (validation only) | 0.5430 | 0.5430 |
| Ridge regression | alpha=10.0 | 0.5424 | 0.5424 |
| Support Vector Regression | C=1.0, gamma=0.01 | 0.5013 | 0.4957 |
| Random Forest | max_depth=12, min_samples_leaf=10 | 0.5030 | 0.5111 |
| Gradient Boosting | max_depth=2, n_estimators=150 | 0.5055 | 0.5156 |

## Feature selection (Section 4b)

After the settings search, compare all four cumulative feature groups with the original settings and the validation-selected settings, skipping duplicates. Each fit uses training-only preprocessing and the same three expanding chronological folds. Select by validation MAE (or the deliberately configured metric), refit on all outer training data, then evaluate later movies. OLS can improve only through its feature selection, not a regularization search.

Section 5 preserves the original, settings-only, and feature-selected results side by side. Plots use the feature-selected candidate. The extra search reuses already-inspected validation folds: its best score is optimistic after selection, not an independent generalization estimate. Future claims require a fresh evaluation period or nested chronological validation; do not adjust the search in response to later test scores.

## Feature/threshold selection results (Section 4b)

The following later-period scores compare original models with candidates selected entirely from the reused training-period folds. These remain exploratory comparisons, not independent confirmation of improvement.

| Model | Selected feature group | Original test MAE | Section 4b test MAE | Original / Section 4b R² |
|---|---|---:|---:|---:|
| Linear regression | Movie characteristics | 0.5430 | 0.5420 | 0.0938 / 0.0919 |
| Ridge regression | Movie characteristics | 0.5424 | 0.5414 | 0.0964 / 0.0939 |
| Support Vector Regression | + Actor history | 0.5013 | 0.4954 | 0.2096 / 0.2352 |
| Random Forest | All features | 0.5030 | 0.5111 | 0.1963 / 0.1847 |
| Gradient Boosting | Movie characteristics | 0.5055 | 0.5152 | 0.2089 / 0.1694 |

Some later-period metrics improve and others worsen. Preserve the original models; do not present every validation-selected candidate as a universally better model. Reusing validation for multiple selection stages increases optimism in the best selection score.

## Current data revision: earlier director ratings

The model table now has 45 columns (39 predictors), adding a smoothed earlier director mean rating, supporting rated-film count (`log1p`), and no-rating-history flag. The director feature group includes all three. Current-movie rating remains excluded from predictors. Earlier ratings use strictly earlier years in the broader matched sample, counting co-directed films once; historical release-time rating availability is unverified.

Older result tables above describe earlier feature versions. Current notebook outputs use the updated feature table. Section 8 supplies a paired ablation with and without just these three new predictors, holding original model settings, sample rows, and chronological folds fixed. It reports both validation and later-test errors; the test comparison does not select a feature version.

## Paired comparison: earlier director rating history

Section 8 holds initial estimator settings, sample rows, and chronological folds fixed. Only the three director-rating-history predictors differ. Lower MAE is better. These are descriptive retrospective comparisons, not independent confirmation after repeated data inspection.

| Model | Validation MAE without / with | Test MAE without / with | Test R² without / with |
|---|---:|---:|---:|
| linear_model | 0.5172 / 0.5162 | 0.5430 / 0.5420 | 0.0938 / 0.0969 |
| ridge_model | 0.5162 / 0.5156 | 0.5424 / 0.5412 | 0.0964 / 0.0995 |
| support_vector | 0.5328 / 0.5223 | 0.5013 / 0.4853 | 0.2096 / 0.2484 |
| random_forest | 0.5070 / 0.5003 | 0.5030 / 0.4974 | 0.1963 / 0.2130 |
| gradient_boosting | 0.5039 / 0.4970 | 0.5055 / 0.4980 | 0.2089 / 0.2255 |

The feature-selected/tuned model in Sections 5–6 can differ from this fixed-setting comparison. For example, current SVR uses director and actor history with C 1 and gamma 0.01; its exploratory later-test MAE is 0.4795 and R² is 0.2697. The controlled initial-settings SVR comparison is 0.5013 versus 0.4853 MAE. Do not attribute all tuning differences to the added feature block.

## Simple-model feature trials (Section 5)

The simple notebook now compares six named presets with 5–10 predictors: movie basics, director experience, the original director history, actor/company/franchise history, four genre flags, and a compact mix. Estimator settings remain fixed, as does the success threshold (0.5). Three expanding chronological folds stay inside the training period; each candidate gets newly fitted preprocessing. The original nine-predictor baseline remains unchanged.

Rating validation selects the lowest mean MAE. The basics-plus-genres preset had the lowest validation MAE (0.5249 versus 0.5382 for the original), but later-test MAE worsened to 0.5587 from 0.5430. This is not a generalization improvement; both results remain visible, and the original model is retained. Do not switch presets based on the test ranking.

Edit `feature_versions` in Section 5 to define another compact preset. The validation/test periods have already been inspected, so further comparisons remain exploratory.
