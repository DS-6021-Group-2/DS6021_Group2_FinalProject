# Reading model diagnostics

Every model notebook ends with **5. Diagnostics**. Run All with `ds6001`. The default output is a short safeguards/limitations table and the before/after overlap table; it does not repeat the full results section.

- VIF ≥10 calls for careful coefficient interpretation, not automatic deletion.
- The overlap table reports raw and transformed input counts, high-VIF counts, maximum VIF, and removed inputs.
- Missingness and numerical fitting are summarized; temporal/retrospective limitations remain visible.
- `diagnostic_results` retains every underlying table, including class metrics, residuals, drift, folds and influence checks.

Set **`SHOW_DIAGNOSTIC_DETAILS = True`** in the top controls for the longer Status / Your result / Meaning / Next experiment table and all technical details. The recommendations below remain candidate experiments for chronological training validation; diagnostics never silently change models.

## What matters most?

1. **Beat the baseline.** Rating MAE/RMSE should be lower than the training-mean baseline. Financial models need useful macro F1 and each class's recall as well as hit precision/F1; majority accuracy alone is insufficient.
2. **Inspect generalization.** Compare training, chronological validation, and later-test performance. KNN can use each training query as its own neighbor, so near-perfect in-sample scores can be misleading.
3. **Review overlapping information.** VIF 1 means no linear overlap, 5–10 merits review, above 10 is high, and infinity indicates exact linear dependence. These are guidelines, not feature-deletion rules. Full rank rules out exact dependence at numerical tolerance, not high correlation.
4. **Review missingness and fitting.** Imputation does not recover unknown budgets. Iteration-limit warnings can require checking scaling, tolerance or `max_iter`.
5. **Document limitations.** Test drift, residuals and probability reliability are descriptive. Safeguards do not prove historical source availability or eliminate selection bias.

## Useful next experiments

| Finding | Candidate experiment inside training validation |
|---|---|
| Genre count overlaps with genre flags | Compare removing `n_genres` while retaining the flags. |
| Many overlapping director histories / quadratic terms | Compare simpler history groups and linear vs quadratic terms; retune the penalty for each candidate. Preserve meaningful parent terms when testing interactions. |
| Missing budgets | Compare a missing-budget indicator or a model without budget on the same rating cohort. Do not invent budgets or financial labels. |
| Large training–validation gap | Try stronger penalties, shallower trees, larger leaves, or more neighbors, as appropriate. |
| Weak Moderate success recall | Compare class weights and feature groups; assess macro F1, all-class recall and hit precision together. |
| High hit precision but low recall | Compare the configured hit thresholds on training validation. A lower threshold can find more hits while reducing precision. |
| Unreliable probability estimates | Consider calibration fitted within chronological training validation; do not calibrate against the test set. |

Choose the simplest candidate with competitive validation results. Lower VIF is not itself proof of better predictions. Do not repeatedly tune on the already-inspected test period; the current findings are exploratory.

## Technical scope

[model_diagnostics.py](model_diagnostics.py) checks the actual transformed **training** matrix, including encoding and polynomial terms. VIF is computed by centered/scaled SVD with an intercept. Exact dependencies receive infinite VIF, constants undefined VIF. PCA models show matrices before and after reduction; full dummy encodings can have expected dependencies. Regularization changes coefficients, not input VIF. Trees do not require independent predictors, although correlations can distort importance interpretation.

Other checks cover IDs, exclusions, date/row separation and chronological folds; numeric imputation/scaling and PCA state; missingness, finite transforms, unseen categories, duplicate patterns and drift; baseline comparisons and convergence. Regression adds residual/bias summaries, rating bounds, and OLS-only leverage/Cook's distance. Classification adds class balance, all three class metrics, multiclass Brier/log loss, and a hit-vs-rest reliability table. Brier/log loss measure overall probability quality, not calibration alone. No assumption test proves predictive validity.

## Verify

From the project root:

```bash
/opt/miniconda3/envs/ds6001/bin/python tests/verify_diagnostics.py
/opt/miniconda3/envs/ds6001/bin/python tests/verify_financial_outcomes.py
/opt/miniconda3/envs/ds6001/bin/python app/verify_models.py
```

The first checks VIF numerically against independent OLS, exact dependencies, constants and orthogonal inputs. The second checks financial boundaries, missing values, three-class decisions/metrics and shared datasets. The app verification checks saved predictions, provenance and UI behavior. Model reports fingerprint shared helpers as well as datasets and notebook code.

AI-assisted. References: [VIF](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html), [scikit-learn evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).


## Automatic overlap review in the model notebooks

After the original settings/feature selection, each notebook evaluates three predefined trims using the **same chronological folds inside training**:

| Preset | Inputs excluded from that model (not from the CSV) |
|---|---|
| Drop genre count | `n_genres`; individual genre flags remain. |
| Lean history | Above, plus `dir_years_active_log1p` and `dir_prior_rated_films_log1p`. |
| Lean history + director count | Above, plus `dir_prior_known_films_log1p` and `imdb_director_count`. |

Retained director rates, prior-film experience and no-history flags remain available. The presets compare supporting-count tradeoffs; no claim is made that these features are universally unnecessary. The existing candidate settings are refitted for each trim; thresholds are reconsidered for success. All preprocessing is fitted within each fold.

The top controls declare the simplicity tradeoff **before evaluation**: rating candidates can be within **0.002 MAE** of the best training-validation candidate. Success candidates must meet the existing precision guard, remain within **0.005 hit F1** of the best eligible candidate, and lose at most **0.01 macro F1** relative to the previous selected model. Within those limits, prefer fewer raw predictors. If no trim qualifies, retain the reference. These are practical tolerances, not statistical equivalence tests or independent validation estimates.

Polynomial trials that expand missing-value indicators can create exact aliases such as an indicator and its square. For trimmed trials, a fold-fitted QR step removes constant/exactly dependent transformed directions in that representation. Its column selection is learned only on training rows; it does not iteratively delete high-VIF features. Original references remain unchanged. PCA already removes zero-variance directions in its retained representation; trees do not require independent inputs.

Diagnostics include **Before / After overlap review** tables with raw/expanded counts, VIF ≥10 counts, maximum VIF, validation scores, and excluded inputs. All candidate fold scores are saved in the ignored model reports. The app comparison includes **Before overlap trim**, preserving the previous selected model separately from the new **Validation-selected** model. Test scores are reported after selection and never choose trims.

Run `/opt/miniconda3/envs/ds6001/bin/python tests/verify_overlap_review.py` to check the tolerances, precision/macro-F1 guards, reference fallback, fold isolation, and exact-alias removal.


## Current before/after results

The latest main integration retains partner grids/exclusions and uses the shared policy. Both overlap versions use identical data/folds. VIF counts refer to transformed inputs (after PCA where used); raw counts refer to supplied dataset fields. Remaining high VIFs are reported, not hidden.

### Rating models

| Model | Raw predictors | VIF ≥10 | Validation MAE | Test MAE |
|---|---:|---:|---:|---:|
| Linear regression | 36 → 31 | 23 → 3 | 0.5178 → 0.5175 | 0.5527 → 0.5501 |
| Ridge | 38 → 33 | 52 → 4 | 0.5169 → 0.5077 | 0.4979 → 0.4985 |
| Lasso | 38 → 33 | 52 → 3 | 0.5086 → 0.5061 | 0.4943 → 0.4963 |
| Elastic net | 38 → 33 | 52 → 3 | 0.5093 → 0.5062 | 0.4949 → 0.4971 |
| Random Forest | 40 → 35 | 23 → 3 | 0.4929 → 0.4939 | 0.5111 → 0.5105 |
| Gradient Boosting | 40 → 35 | 23 → 3 | 0.4893 → 0.4906 | 0.5040 → 0.5040 |
| KNN + PCA | 38 → 33 | 0 → 0 | 0.5545 → 0.5534 | 0.5371 → 0.5381 |
| KNN | 40 → 35 | 23 → 3 | 0.5325 → 0.5286 | 0.5254 → 0.5289 |

### Three-class financial models

| Model | Raw predictors | VIF ≥10 | Validation hit F1 | Test macro F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 25 → 25 | 0 → 0 | 0.1687 → 0.1687 | 0.4259 → 0.4259 |
| Random Forest | 32 → 30 | 0 → 0 | 0.1867 → 0.2087 | 0.4130 → 0.4231 |
| Gradient Boosting | 38 → 33 | 23 → 2 | 0.2448 → 0.2453 | 0.4172 → 0.4246 |
| KNN + PCA | 40 → 37 | 0 → 0 | 0.1852 → 0.1852 | 0.4070 → 0.4124 |

The common precision requirement is a mean training-validation selection rule, not a test guarantee. Original selected references remain available as Before overlap trim. The separate full-history experiment uses nested validation and can have explicitly infeasible inner goals.
