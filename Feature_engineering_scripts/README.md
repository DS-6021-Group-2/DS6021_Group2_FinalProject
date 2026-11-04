# Feature engineering

This folder turns saved movie data into predictors for the rating and success models.

## Run this notebook

Open [feature_engineering.ipynb](feature_engineering.ipynb), select **`ds6001`**, and run all cells from this folder.

| Output | Movies | Used for |
|---|---:|---|
| `../data/model_data_success.csv` | 6,919 | Predicting hit or bust. |
| `../data/model_data_rating.csv` | 13,276 | Predicting viewer rating, without financial filters. |

The notebook reads `cleaned_movies.csv`, `combined_dataset.csv`, `imdb_movies.csv`, and `lead_actor_backfill.csv` from `../data/`. The actor supplement is already saved; no API pull is needed.

## What happens where?

1. **Steps 1–12:** inspect the financially eligible data and build features.
2. **Step 13:** verify or save the success-model table.
3. **Step 14:** prepare the broader rating cohort from the combined data and build the same features.
4. Train models in the appropriate [model folder](../Modeling_scripts/README.md).

Source and cleaned CSVs are preserved. Identical model exports are reused. If an existing export differs, the notebook stops: review the change before moving the previous export and rerunning.

## What the predictors describe

| Group | What the model learns about |
|---|---|
| Movie characteristics | Runtime, release year/month, genres, language, certification, and director count. |
| Director experience | Number of earlier IMDb films and years since the earliest observed film. |
| Director financial record | Success rate of earlier films with known finances, plus supporting count and no-record flag. |
| Director rating record | Average rating of observed earlier movies, plus supporting count and no-history flag. |
| Lead actor | Observed film activity in the previous five years, including eligible 1985–1990 IMDb credits. |
| Company and franchise | Recent production-company activity, earlier franchise films, and sequel status. |

**Smoothing** pulls small director records toward the earlier overall average, using a five-film prior. **`log1p`** compresses large experience counts while keeping zero at zero. Raw counts and averages remain available in the notebook. See the [codebook](../CODEBOOK.md) for exact formulas.

`log_budget = log10(budget)` is included as a numeric movie characteristic, with budget plots retained for interpretation. Raw budget and revenue are excluded from predictors. Current-movie ratings and success labels are outcomes, not predictors. Models fill missing values, encode categories, and scale using training data only.

## History coverage

Only strictly earlier years count for director/actor/company history; franchise history uses earlier dates. Co-directed films count once. Earlier ratings are TMDB snapshots linked through IMDb director IDs, not a complete set of IMDb ratings.

The actor supplement covers the original success cohort. Newly added early-window rating movies without a verified match are flagged rather than given fabricated history. Shared movies retain the same predictors in both exports. History updates through time, so evaluation represents rolling predictions; historical release-time availability is not verified.

To rebuild the original actor supplement, place the required IMDb downloads in `data/` and run from the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python collecting_and_cleaning_data_scripts/imdb_actor_backfill.py
```

See the [collection guide](../collecting_and_cleaning_data_scripts/README.md) for download requirements. Normal modeling uses the existing files.

`log_budget` uses positive reported budgets. The rating table keeps all movies, including **5,439 unknown budgets** stored as missing values; each training pipeline learns its own median for imputation. The success table has no missing budgets. Source and cleaned CSVs are unchanged.
