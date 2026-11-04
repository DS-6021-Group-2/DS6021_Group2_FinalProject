# Success models: hit or bust

## Start with the simple model

`simple_model.ipynb` fits logistic regression using only **9 predictors**: runtime, director film experience, smoothed earlier director success rate, known-outcome support count, certification, release month, sequel status, English-language status, and a no-track-record flag. It uses the same chronological train/test rows and training-only preprocessing, with fixed C 1 and threshold 0.5. The original nine-feature baseline has no tuning; the optional feature comparison applies the validation precision guard and reports hit precision, recall, and F1 alongside accuracy, balanced accuracy, and ROC AUC. The other notebooks retain the detailed precision/F1 workflow.

Predict `success` in `../../data/model_data.csv`: **1 = hit** (revenue at least 2.5 times budget), **0 = bust** (below that benchmark). These are project categories, not audited profit/loss. Financial cleaning excludes unreported financial values; they are not automatically labeled busts.

| Notebook | Classifier | Initial settings |
|---|---|---|
| `logistic_regression.ipynb` | Logistic regression | C 1, maximum 3,000 iterations |
| `random_forest.ipynb` | Random Forest | 150 trees, depth 12, minimum leaf size 10, seed 42 |
| `gradient_boosting.ipynb` | Gradient Boosting | 150 trees, learning rate 0.05, depth 3, minimum leaf size 10, seed 42 |

Run notebooks independently from this folder with the `ds6001` kernel. They train in memory and save tables/plots in notebook outputs; they do not change CSVs or export trained models. Rating regressors are in `../rating_models/`.

The simple model's exploratory later-test accuracy is **66.5%**, hit precision **51.8%**, hit recall **36.6%**, hit F1 **0.4291**, and ROC AUC **0.6608**. The majority baseline accuracy is 65.7%.

## Inputs and evaluation

Exclude both `success` and `vote_average`, plus movie IDs, title, and full release date from predictors. Release date orders the chronological split; year/month remain features. Budget and revenue define the label rather than serving as predictors.

Use approximately the earliest 80% (5,535 movies) for training and the latest 20% (1,384 movies) for evaluation, keeping equal release dates together. Fit imputation, scaling, and one-hot encoding within each training fold. All models share three expanding chronological validation windows inside the earlier training period.

## Current priority: precision and hit F1

**Hit precision** is the proportion of predicted hits that actually are hits. **Hit recall** is the proportion of actual hits found. **Hit F1** is their harmonic mean and gives them equal weight; **macro F1** instead averages class-specific F1 across hits and busts. F1 itself does not prioritize precision.

- **Section 4 — settings search:** set `TUNING_METRIC = "Hit_F1"`. Show hit precision/recall, macro F1, balanced accuracy, accuracy, and ROC AUC alongside it. The estimator search uses threshold 0.5. Logistic regression tunes C and class weights; Random Forest tunes depth, leaf size, and class weights; Gradient Boosting tunes depth and tree count.
- **Section 4b — precision guard and final selection:** compare the four cumulative feature groups using original and selected settings, and thresholds 0.30–0.90 by 0.05. `MIN_VALIDATION_PRECISION = None` requires at least the original all-feature model's mean validation precision at threshold 0.5. A deliberately specified floor can replace that reference. Among eligible candidates, maximize mean hit F1; break ties with higher precision, then a threshold nearer 0.5. Stop clearly if an explicit requirement is infeasible.
- **Sections 5–6 — evaluation:** keep original, settings-only, and precision/F1-selected training/test results side by side. Show hit precision, recall, F1, confusion matrices, ROC/precision–recall curves, and chronological examples. Final plots use the selected threshold.
- **Section 7 — feature comparisons:** retain original settings and threshold 0.5 to show the original cumulative feature comparisons, including hit F1/precision/recall. This is separate from the final selection.

The precision guard applies to the mean across validation windows, not every window. It discourages sacrificing validation precision for recall but cannot guarantee later test precision. Avoid choosing precision alone, which can favor predicting very few hits.

## Limits

Feature/settings/threshold choices reuse previously inspected training folds. Their best scores are selection scores, not independent validation estimates. The later period has already been inspected too; further generalization claims require a fresh evaluation period or nested chronological validation. Test outcomes do not determine the precision floor, settings, features, or threshold.

Earlier-year financial histories update through time, representing rolling predictions rather than a single forecast at the split date. Retrospective sources do not verify when final outcomes were reported. Results cover the financially eligible sample, not all TMDB movies. Probability calibration has not been evaluated. Report precision, recall, and F1 together; tuning may improve one while worsening another.

## Current precision/F1 results

Per-model validation floors use that model's original validation precision; they are not a shared externally specified business requirement. Test precision is not guaranteed by these guards. All results below are exploratory.

| Model | Mean validation precision floor | Selected validation precision | Threshold | Test hit precision | Test hit recall | Test hit F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.584 | 0.585 | 0.45 | 0.475 | 0.566 | 0.517 |
| Gradient Boosting classifier | 0.594 | 0.619 | 0.50 | 0.526 | 0.429 | 0.473 |
| Random Forest classifier | 0.649 | 0.693 | 0.60 | 0.552 | 0.312 | 0.398 |

Higher precision can reduce recall and F1. The forest precision emphasis illustrates that trade-off: its test precision is slightly above the original forest, but recall and F1 are lower. The logistic model meets the validation guard but its later test precision is below its original test precision. Keep initial results visible and do not raise/lower the floor in response to this test table.

## Current data revision: earlier director ratings

The model table now has 45 columns (39 predictors), adding a smoothed earlier director mean rating, supporting rated-film count (`log1p`), and no-rating-history flag. The director feature group includes all three. Current-movie rating remains excluded from predictors. Earlier ratings use strictly earlier years in the broader matched sample, counting co-directed films once; historical release-time rating availability is unverified.

Older result tables above describe earlier feature versions. Current notebook outputs use the updated feature table. The classifiers also consume the new director-history predictors with the same precision/F1 selection rules.

## Current 45-column feature-table results

This rerun includes earlier director rating history. Selection still uses hit F1 with the validation precision guard; these later-test metrics are exploratory. Earlier result tables describe prior feature versions.

| Classifier | Selected feature group | Threshold | Hit precision | Hit recall | Hit F1 |
|---|---|---:|---:|---:|---:|
| logistic_regression | + Director history | 0.65 | 0.523 | 0.472 | 0.496 |
| gradient_boosting | All features | 0.50 | 0.545 | 0.444 | 0.490 |
| random_forest | All features | 0.60 | 0.554 | 0.322 | 0.407 |

## Simple-model feature trials (Section 5)

The simple notebook now compares six named presets with 5–10 predictors: movie basics, director experience, the original director history, actor/company/franchise history, four genre flags, and a compact mix. Estimator settings remain fixed, as does the success threshold (0.5). Three expanding chronological folds stay inside the training period; each candidate gets newly fitted preprocessing. The original nine-predictor baseline remains unchanged.

Success validation selects the highest mean hit F1 subject to mean precision at least that of the original nine-feature model. The original director-history preset remained selected: validation F1 0.3163, test hit F1 0.4291, test precision 0.5179, test recall 0.3663. The new small alternatives did not improve the validation selection objective.

Edit `feature_versions` in Section 5 to define another compact preset. The validation/test periods have already been inspected, so further comparisons remain exploratory.
