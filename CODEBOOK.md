# TMDB and IMDb Dataset Codebook

## Sources and structure

| Source file | One row represents | Purpose |
|---|---|---|
| `data/tmdb_movies.csv` | One movie | Movie characteristics, viewer ratings, and financial outcomes. |
| `data/imdb_movies.csv` | One movie–director pair | IMDb director identities and links to their films. Co-directed movies have multiple rows. |

Join sources using **`imdb_id`**, not titles or director names. IMDb `tt…` IDs identify movies; `nm…` IDs identify people.

The TMDB sample was selected from 1990–2025 releases with at least 100 votes, up to 500 movies per year ordered by vote count. The current manually installed TMDB file contains 14,509 movies and is not a complete movie census. The IMDb file covers a broader filmography and now includes release year (`startYear`), but has no full release dates, ratings, budgets, or revenues.

## TMDB source variables

| Variable | Type / unit | Definition |
|---|---|---|
| `tmdb_id` | Integer ID | Unique TMDB movie identifier. |
| `imdb_id` | Text ID | IMDb movie identifier; join key shared with IMDb. |
| `title` | Text | TMDB movie title; not necessarily unique. |
| `release_date` | YYYY-MM-DD | Release date; used to order films and derive year/month. |
| `budget` | Integer, USD | Reported production budget; not inflation-adjusted. |
| `revenue` | Integer, USD | Reported gross revenue; not inflation-adjusted. |
| `runtime` | Integer, minutes | Running time. |
| `genres` | Text list | Genre names. |
| `original_language` | Category | Original-language code; the en-US response setting does not restrict movies to English. |
| `popularity` | Decimal | TMDB popularity score at retrieval. |
| `production_companies` | Text list | Production-company names. |
| `director` | Text list | Director names from TMDB credits, without person IDs. |
| `top_cast` | Text list | First five cast names returned in credits. |
| `vote_count` | Integer | Number of viewer votes at retrieval. |
| `vote_average` | Decimal, 0–10 | TMDB viewer rating; regression target. |
| `collection_id` | Nullable integer ID | TMDB collection/franchise identifier; blank when no collection is recorded. |
| `collection_name` | Text | Name of the recorded collection/franchise. |
| `us_certification` | Category | US content certification from release information, such as PG-13 or R; not viewer rating. Blank means unknown. |

TMDB text lists use a pipe (`|`) separator. Blank cells are missing values.

## IMDb source variables

| Variable | Type | Definition |
|---|---|---|
| `imdb_id` | Text ID (`tt…`) | Movie identifier; repeats when a movie has multiple directors. |
| `nconst` | Text ID (`nm…`) | Stable IMDb person identifier for the director. |
| `director` | Text | Director name corresponding to `nconst`. |
| `title` | Text | IMDb movie title; may differ from TMDB's title. |
| `startYear` | Nullable integer year | IMDb release year, retained separately from TMDB release date; missing years remain unknown. |

The pair (`imdb_id`, `nconst`) identifies a movie–director relationship. Repeated movie IDs here are expected, not necessarily duplicate records.

## Joined variables and naming

**`Modeling_scripts/rating_models/linear_model.ipynb`:** trains linear regression for `vote_average` using `../../data/model_data_rating.csv`, an earlier/later chronological split, and training-only preprocessing. Current-movie outcomes and identifiers are excluded from predictors. It compares MAE, RMSE, and R² with a training-mean baseline and does not change CSVs.

**`collecting_and_cleaning_data_scripts/data_cleaning.ipynb`:** groups the combined movie–director rows to one movie row and replaces `nconst` / `director_imdb` with:

| Added variable | Type | Definition |
|---|---|---|
| `imdb_director_ids` | Text list | IMDb director IDs, pipe-separated. |
| `imdb_directors` | Text list | Unique IMDb director names, pipe-separated. |
| `imdb_director_count` | Integer | Number of directors on this movie, not their previous films. |

The in-memory `movies` table has one row per matched movie. Movies excluded by the earlier inner join are not restored. `release_date` is parsed as a date. No `imdb_matched` flag is needed because all input rows are already matched.

**`combining_data.ipynb`:** defines an inner join without aggregating directors. Its naming is different: `title_tmdb` / `title_imdb` and `director_tmdb` / `director_imdb`; `nconst` remains a separate column. Other TMDB columns keep their names, and IMDb `startYear` is retained. The updated combined file has 14,824 movie–director rows, 22 columns, and 13,276 unique movies. This join retains only matched movies and can produce several rows per movie. The model table retains identifiers, engineered predictors, and targets rather than all joined source fields.

## Project use and cautions

- **Rating target:** `vote_average` in `model_data_rating.csv`; 13,276 matched rated movies without financial filters. Exclude the rating from predictors.
- **Financial dataset:** `model_data_success.csv` preserves the former 6,919-movie model table, including both outcome columns. Exclude both `success` and `vote_average` from success predictors.
- **Current performance target:** `success = 1` when `revenue >= 2.5 * budget`, otherwise 0, for the financially eligible cleaned cohort. The current task is binary hit/bust (1/0), with classifier notebooks organized under `Modeling_scripts/success_models/`. The ratio is not net profit.
- **Implemented director history:** earlier IMDb feature-film counts and years active, plus earlier known-outcome counts and smoothed success rates within the cleaned financial cohort. The broader matched sample also supplies earlier-year director rating history, using TMDB ratings linked through IMDb IDs. Same-year films are excluded; retrospective sources do not verify historical availability of outcomes.
- Investigate zero budgets, revenues, and runtimes. Zero budget makes the ratio undefined; zero revenue is not automatically a bust.
- For pre-release models, exclude the current movie's revenue, rating, popularity, vote count, and target-derived fields from predictors. Use chronological splits and fit preprocessing on training data only.

Definitions checked against both source CSV headers and notebook join code. TMDB references: [movie details](https://developer.themoviedb.org/reference/movie-details), [credits](https://developer.themoviedb.org/reference/movie-credits).

## Revised feature trial

The current feature-engineering notebook writes **`data/model_data_success.csv`** and **`data/model_data_rating.csv`**, preserving `cleaned_movies.csv`. Budget/log-budget plots remain exploratory; neither budget variable is selected as a predictor.

| Revised variable | Definition |
|---|---|
| `dir_prior_success_rate_smoothed` | `(earlier successes + 5 × earlier global success fraction) / (earlier known films + 5)`; global prior uses strictly earlier release years, or 0.5 when none exist. |
| `dir_prior_films_log1p`, `dir_years_active_log1p`, `dir_prior_known_films_log1p` | Natural log of one plus each original director-experience count. |
| `lead_actor_films_5yr_backfilled_log1p` | Natural log of one plus the existing five-year count plus eligible, non-duplicated IMDb 1985–1990 acting-principal credits for the same lead actor. |
| `company_films_5yr_log1p`, `franchise_prior_films_log1p` | Natural log of one plus the original company/franchise experience count. |
| `backfill_unavailable` | A 1985–1990 backfill window applies, but the actor match or target-year agreement is insufficient; the original count is retained. |

`data/imdb_lead_actor_1985_1990.csv` contains distinct (`imdb_id`, `actor_imdb_id`) credits with IMDb `startYear` in 1985–1990 for matched current leads. `data/lead_actor_backfill.csv` contains one row per cleaned movie, the lead match/status, `imdb_extra_films_5yr`, `backfill_needed`, and `backfill_unavailable`. Backfill excludes current/same-year/later movies and titles already in the matched TMDB sample. Target IMDb/TMDB year disagreements are flagged rather than guessed. Only principal actor/actress credits are counted, so this is not a complete cast filmography. No total-career actor feature is used.

## Earlier director rating history

| Variable | Role / definition |
|---|---|
| `dir_prior_rating_smoothed` | Numeric predictor, 0–10: `(sum of earlier director-film ratings + 5 × earlier global mean rating) / (earlier rated films + 5)`. The prior uses distinct earlier reference movies; fallback 5.0 only if none exist. |
| `dir_prior_rated_films_log1p` | Numeric predictor: `ln(1 + count of distinct earlier rated films)` across the movie's directors. |
| `dir_no_rating_history` | Binary predictor: 1 when no earlier rated film is observed for the director(s). |
| `dir_prior_rating_mean` | Inspection only in the feature notebook: unshrunk earlier-film mean, missing when no history exists. |
| `dir_prior_rated_films` | Inspection only: supporting raw count before `log1p`. |

Reference: the broader matched TMDB/IMDb table, including financially ineligible films. Compare TMDB release years on both sides. Exclude current, same-year, and future films; co-directed films count once. The current movie's `vote_average` remains a target, never an input. Earlier-film ratings come from retrospective snapshots and are not verified historical release-time ratings. `model_data_success.csv` has 45 columns; the expanded `model_data_rating.csv` has 44 because it has only the rating target, not current-movie success.

## Task-specific model exports

The original shared model table is renamed to `model_data_success.csv` without changing its cohort or values. `model_data_rating.csv` is constructed from unique combined movies with valid dates/ratings, applying nonfinancial preparation and the same predictor definitions. Missing finances are not labeled as busts. Shared predictors/ratings match exactly across files. Neither budget nor revenue is a predictor.

The success objective now requires 0.70 mean training-validation hit precision, then maximizes hit F1 among qualifying settings/features/thresholds. If infeasible, notebooks report it rather than relax the target. This validation requirement does not guarantee later test precision.
