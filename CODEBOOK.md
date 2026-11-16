# Movie dataset codebook

## Files and targets

| File in `data/` | Row represents | Use |
|---|---|---|
| `tmdb_movies.csv` | One movie | TMDB characteristics, ratings and finances. |
| `imdb_movies.csv` | One movie–director pair | IMDb identities and earlier film careers. |
| `combined_dataset.csv` | One matched movie–director pair | Joined by `imdb_id`; co-directors create multiple rows. |
| `cleaned_movies.csv` | One financially eligible movie | Financial preparation before feature engineering. |
| `model_data_rating.csv` | One movie; 13,276 rows | Predict `vote_average`; 40 available predictors. |
| `model_data_success.csv` | One movie; 6,919 rows | Predict `success`; same predictors, also retains `vote_average`. |

Financial eligibility: reported budget ≥$250,000 and revenue ≥$1,000. Rating eligibility does not require finances. Model tables include four ID/date columns; these are not predictors. Model-specific trims leave CSV columns intact.

| `success` | Category | Revenue / budget |
|---|---|---|
| 0 | Bust | ≤1 |
| 1 | Moderate success | >1 and <2.5 |
| 2 | Hit | ≥2.5 |

Missing finances remain unclassified. These are revenue-to-budget categories, not audited profit; never average the 0/1/2 codes as a hit rate.

## Source and identification fields

| Variable(s) | Meaning |
|---|---|
| `tmdb_id`, `imdb_id` | Movie IDs; IMDb `tt…` identifies a title. Join sources by `imdb_id`, not title. |
| `title` → `title_tmdb`, `title_imdb` | Source titles; suffixes distinguish joined fields. |
| `release_date`, `startYear` | TMDB date and IMDb release year; retained separately. |
| `nconst` | IMDb director person ID (`nm…`); repeats across their films. |
| `imdb_director_ids`, `imdb_directors` | Aggregated director IDs/names in cleaned data. |
| `budget`, `revenue` | Reported USD amounts; not inflation-adjusted. Zero means unreported. |
| `runtime` | Minutes; zero becomes missing. |
| `genres`, `production_companies`, `top_cast` | Pipe-separated (`|`) lists; top cast contains up to five names. |
| `director_tmdb`, `director_imdb` | Director names from the two sources; raw files use `director`. |
| `original_language`, `us_certification` | Language code and US content rating; blank certification becomes `Unknown`. R/NC-17 are grouped during cleaning. |
| `collection_id`, `collection_name` | Franchise identifiers/name; missing for movies outside a recorded collection. |
| `vote_average`, `vote_count`, `popularity` | TMDB rating (0–10), number of votes, and retrieval-time popularity. |
| `revenue_to_budget` | Analysis-only revenue/budget ratio; never a predictor. |

## Engineered predictors

| Variable(s) | Definition |
|---|---|
| `log_budget` | `log10(budget)` for positive reported budgets; otherwise missing. |
| `release_year`, `release_month` | Derived from TMDB release date; month is categorical. |
| `n_genres`, `genre_*` | Genre count and individual 0/1 flags. |
| `imdb_director_count` | Directors on the current movie, not earlier-film experience. |
| `is_english`, `is_sequel` | 0/1: English language; an earlier movie in the same collection is observed. |
| `dir_prior_films_log1p` | `ln(1 + earlier IMDb feature-film count)`. |
| `dir_years_active_log1p` | `ln(1 + years since the earliest observed IMDb film)`. |
| `dir_prior_known_films_log1p` | `ln(1 + earlier financially eligible films with known outcomes)`. |
| `dir_prior_success_rate_smoothed` | Earlier **hit fraction**: `(hits + 5 × earlier global hit fraction)/(known films + 5)`; global fallback 0.5. |
| `dir_prior_rating_smoothed` | `(sum of earlier TMDB ratings + 5 × earlier global mean)/(rated films + 5)`; global fallback 5.0. |
| `dir_prior_rated_films_log1p` | `ln(1 + distinct earlier rated films)` across the movie’s directors. |
| `dir_first_film`, `dir_no_track_record`, `dir_no_rating_history` | 0/1: no observed earlier career films, financial outcomes, or ratings respectively. |
| `lead_actor_films_5yr_backfilled_log1p` | `ln(1 + observed five-year activity of the first-listed TMDB actor + eligible IMDb supplement)`. |
| `company_films_5yr_log1p` | `ln(1 + largest five-year film count among the movie’s companies)`. |
| `franchise_prior_films_log1p` | `ln(1 + observed earlier films in the same collection)`. |
| `backfill_unavailable` | 0/1: an early actor-history supplement is needed but cannot be reliably matched. |

Actor supplements: `imdb_lead_actor_1985_1990.csv` contains distinct earlier movie–actor credits; `lead_actor_backfill.csv` records per-movie matching and extra counts. Existing TMDB-sample credits are not counted twice.

## Interpretation

- Director/actor/company history excludes current, same-year and future films; franchise history uses strictly earlier dates. Co-directed films count once.
- Earlier ratings are TMDB snapshots linked through IMDb identities; IMDb source data has no ratings or finances. Histories are incomplete and retrospective.
- Current outcomes, revenue/ratio, popularity, vote count and IDs are excluded from predictors. Missing values are imputed inside training folds only.
- Train before **2017-11-18**; tune on three expanding chronological folds. Rating selection minimizes MAE. Financial selection requires **≥70% mean validation hit precision**, then maximizes hit F1; also report macro F1 and moderate recall. This is not a test-performance guarantee.
- See [data preparation](data/README.md), [modeling](Modeling_scripts/README.md), and [diagnostics](Modeling_scripts/DIAGNOSTICS.md) for workflow details.

Shared goals: [project_policy.py](Modeling_scripts/project_policy.py). The dated `model_data_success_2026-11-04.csv` is a historical binary snapshot, excluded from active training and app inputs.
