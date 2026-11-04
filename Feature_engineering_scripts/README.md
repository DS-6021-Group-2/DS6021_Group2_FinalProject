# Feature engineering

This folder turns the existing cleaned movie data into predictors. It does not repeat the API pulls or the cleaning decisions.

## Notebook

Run **`feature_engineering.ipynb`** to build and inspect the revised predictors. Its output is **`../data/model_data.csv`**.

Select the `ds6001` Python environment. The feature notebook uses paths relative to this folder, so run it with `Feature_engineering_scripts/` as the kernel's working directory. Run cells from top to bottom.

## Inputs and run order

The feature notebook reads:

- `../data/cleaned_movies.csv`: the 6,919 movies and their existing binary `success` labels.
- `../data/combined_dataset.csv`: the broader matched movie sample used for historical activity counts.
- `../data/imdb_movies.csv`: IMDb director identities, film links, and release years.
- `../data/lead_actor_backfill.csv`: the already-built IMDb 1985–1990 lead-actor supplement.

The backfill CSV is already present; **you do not need to download IMDb files or rebuild it to run the notebook**. If rebuilding is necessary, place `title.basics.tsv.gz`, `title.principals.tsv.gz`, and `name.basics.tsv.gz` in `data/`, then run this from the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python collecting_and_cleaning_data_scripts/imdb_actor_backfill.py
```

Then run `feature_engineering.ipynb`. Predictive modeling belongs in `../Modeling_scripts/`. Its `rating_models/` subfolder contains rating regressors and feature comparisons; `success_models/` contains binary financial classifiers.

## How the features work

| Feature group | Construction |
|---|---|
| Genres and language | Separate genre flags, number of genres, and an English-language indicator. |
| Director experience | Distinct earlier IMDb films, years since the earliest recorded earlier film, and a no-earlier-film flag. Co-directors' earlier films are pooled without counting the same film twice. |
| Director track record | Earlier films with known financial outcomes and their success rate, plus a no-track-record flag. |
| Director rating history | Earlier-year mean viewer rating, shrunk toward the earlier overall rating, plus supporting count and a no-history flag. |
| Lead actor | The first actor in TMDB `top_cast`; activity in the preceding five release years, supplemented with eligible older IMDb credits. |
| Production company | The largest preceding-five-year movie count among the current movie's companies. |
| Franchise | Number of earlier observed films in the same collection. |
| Other characteristics | Runtime, release year/month, certification, number of directors, and the existing sequel indicator. |

### Current trials

**Budget:** Keep the original budget and `log10(budget)` histograms for interpretation. Neither `budget` nor `log_budget` is selected as a predictor. Budget and revenue were used in cleaning to define `success = 1` when revenue is at least 2.5 times budget; feature engineering does not change those labels.

**Early actor history:** Add IMDb acting-principal feature-film credits from **1985–1990 only**, for the same first-listed TMDB actor. A credit must fall in the current movie's preceding five-year window and must not already appear in the existing matched reference sample. Current/same-year/later films are excluded. Ambiguous/missing matches or inconsistent target release years retain the original count with `backfill_unavailable = 1`. This added credits for 493 movies; it is not a complete actor filmography or a total-career actor feature.

**Director smoothing:** Use

```text
(earlier successes + 5 × earlier global success fraction)
-------------------------------------------------------
             (earlier known films + 5)
```

The global fraction uses only earlier release years, with a neutral 0.5 prior when none exist. The fixed strength of five is a trial setting. Keep the original rate and supporting count for inspection; select the smoothed rate for the revised model.

**Experience transformations:** Apply `log1p(x) = ln(1 + x)` to director film counts, director years active, known-outcome and rated-film counts, backfilled actor activity, company activity, and franchise history. Zero stays zero, and large values are compressed. Original counts remain available in the notebook. This is not mean-zero/unit-variance scaling.

## Which CSVs change?

**Source and cleaned CSVs remain unchanged. `model_data.csv` now contains the revised predictors and is the single model table used across the project.**

| File | Status |
|---|---|
| `tmdb_movies.csv`, `imdb_movies.csv`, `combined_dataset.csv` | Existing inputs; unchanged by feature engineering. |
| `cleaned_movies.csv` | Existing cleaned cohort and labels; unchanged. |
| `imdb_lead_actor_1985_1990.csv` | New historical acting-credit reference, created by the backfill script. |
| `lead_actor_backfill.csv` | New movie-level backfill counts and availability flags, created by the script. |
| `model_data.csv` | Canonical model table: the same 6,919 movies and labels, with revised predictors, `success` and `vote_average` targets, and no budget/log-budget column. |

Use **`model_data.csv`** when testing these changes. It contains 6,919 movies and 45 columns. The former baseline has been replaced; its earlier version remains in Git history. Export checks reuse an identical revised snapshot and stop if an existing output differs. Rename the previous output before deliberately creating a replacement.

## Interpretation and limits

- Imputation, categorical encoding, scaling, and any learned feature selection belong inside the training pipeline after a chronological split.
- Earlier-year filters prevent using current/later-year film outcomes, but the sources are retrospective snapshots, not verified records of what was available on each historical release date.
- Principal cast, movie sampling, name matching, and missing financial outcomes limit history coverage. The 1985–1990 backfill helps early actor windows; it does not expand production-company history.
- These remain trial features. A reproducible chronological model comparison is needed before claiming an improvement.
- Outcome-based history is updated using earlier-year outcomes throughout the dataset. A multi-year test block therefore represents rolling predictions; a forecast made once at the split date must freeze history at that date.
- Actor/company names can differ across sources, and both TMDB sampling and principal-credit coverage can undercount experience. Missing history is not proof of no experience.

## Model targets

`success` is the existing binary financial target, and `vote_average` is the TMDB rating target for regression. Step 11 retains both in the export and defines `FEATURE_COLUMNS` separately. Exclude both targets and all identifier/date columns when constructing model inputs. The current financial task uses binary hit/bust labels rather than three classes.

## Director rating history

The history cell before Step 11 reads the movie–director rows of `combined_dataset.csv`, retains rated titles, and links directors by IMDb person ID. It pools distinct earlier-year films across co-directors. Current/same-year/later films are excluded. The global prior averages distinct earlier movies, not movie–director rows. A five-film prior smooths the director mean; use the 0–10 scale midpoint 5.0 only when no earlier reference films exist.

Step 11 exports three predictors: `dir_prior_rating_smoothed`, `dir_prior_rated_films_log1p`, and `dir_no_rating_history`. The raw `dir_prior_rating_mean` and `dir_prior_rated_films` remain in memory for inspection. Coverage is 4,697 movies with observed earlier ratings out of 6,919. The old predictors and labels are unchanged. Neutral/default history does not mean the director actually had average ratings.

Model notebooks read the updated table. Section 8 of each rating notebook compares the previous feature set with the additional three features under identical initial model settings, folds, and later-test rows. This separates the feature effect from tuning changes. Snapshot ratings do not establish what was known at each historical release date; the evaluation remains retrospective and exploratory.
