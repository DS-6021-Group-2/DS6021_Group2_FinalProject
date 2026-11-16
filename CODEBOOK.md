# Movie dataset codebook

## Files and targets

| File in `data/` | Row represents | Use |
|---|---|---|
| `tmdb_movies.csv` | Movie | Characteristics, ratings and finances. |
| `imdb_movies.csv` | Movie–director pair | IMDb IDs and career history. |
| `combined_dataset.csv` | Matched movie–director pair | Joined on `imdb_id`; co-directors repeat movies. |
| `cleaned_movies.csv` | Financially eligible movie | Input to financial feature engineering. |
| `model_data_rating.csv` | Movie; 13,276 rows | Target: `vote_average`. |
| `model_data_success.csv` | Movie; 6,919 rows | Target: `success`; retains `vote_average`. |

Financial eligibility: budget ≥$250,000 and revenue ≥$1,000; ratings require no finances. Both model tables offer 40 predictors; four ID/date columns are excluded. Model trims leave CSVs intact.

| `success` | Category | Revenue / budget |
|---|---|---|
| 0 | Bust | ≤1 |
| 1 | Moderate success | >1 and <2.5 |
| 2 | Hit | ≥2.5 |

Missing finances are unclassified. Categories measure revenue/budget, not profit; hit rates count class 2 only.

## Source and identification fields

| Variable(s) | Meaning |
|---|---|
| `tmdb_id`, `imdb_id` | Movie IDs; join on `imdb_id` (`tt…`). |
| `title` → `title_tmdb`, `title_imdb` | Source titles, suffixed after joining. |
| `release_date`, `startYear` | TMDB date; IMDb year. |
| `nconst` | IMDb director ID (`nm…`). |
| `imdb_director_ids`, `imdb_directors` | Aggregated director IDs/names. |
| `budget`, `revenue` | Nominal USD; zero means unreported. |
| `runtime` | Minutes; zero becomes missing. |
| `genres`, `production_companies`, `top_cast` | `|`-separated lists; up to five cast names. |
| `director_tmdb`, `director_imdb` | Source director names; raw field: `director`. |
| `original_language`, `us_certification` | Language code; US content rating (`Unknown` if blank; R/NC-17 grouped). |
| `collection_id`, `collection_name` | Recorded franchise ID/name, otherwise missing. |
| `vote_average`, `vote_count`, `popularity` | TMDB rating (0–10), votes, snapshot popularity. |
| `revenue_to_budget` | Revenue/budget; analysis only. |

## Engineered predictors

| Variable(s) | Definition |
|---|---|
| `log_budget` | `log10(budget)`; missing if nonpositive. |
| `release_year`, `release_month` | Release year; categorical month. |
| `n_genres`, `genre_*` | Genre count; 0/1 genre flags. |
| `imdb_director_count` | Current movie's director count. |
| `is_english`, `is_sequel` | 0/1: English; observed earlier collection film. |
| `dir_prior_films_log1p` | Log count of earlier IMDb feature films. |
| `dir_years_active_log1p` | Log years since earliest observed IMDb film. |
| `dir_prior_known_films_log1p` | Log count of earlier financially eligible films. |
| `dir_prior_success_rate_smoothed` | `(earlier hits + 5 × earlier global hit fraction)/(known films + 5)`; global fallback: 0.5. |
| `dir_prior_rating_smoothed` | `(earlier rating sum + 5 × earlier global mean)/(rated films + 5)`; global fallback: 5.0. |
| `dir_prior_rated_films_log1p` | Log count of distinct earlier rated films across directors. |
| `dir_first_film`, `dir_no_track_record`, `dir_no_rating_history` | 0/1: no earlier career films, financial outcomes, or ratings. |
| `lead_actor_films_5yr_backfilled_log1p` | Log five-year film count for first-listed TMDB actor, including IMDb supplement. |
| `company_films_5yr_log1p` | Log largest five-year film count among companies. |
| `franchise_prior_films_log1p` | Log count of earlier collection films. |
| `backfill_unavailable` | 0/1: needed actor supplement cannot be reliably matched. |

`*_log1p` means `ln(1 + value)`. Actor supplement: `imdb_lead_actor_1985_1990.csv`; matching/counts: `lead_actor_backfill.csv`. Existing credits are not double-counted.

## Interpretation

- Director/actor/company histories exclude same-year and later films; franchises use earlier dates. Co-directed films count once.
- Histories are incomplete and retrospective. Earlier ratings come from TMDB; IMDb provides identities/careers, not ratings or finances.
- Outcomes, revenue/ratio, popularity, vote count and IDs are excluded from predictors. Imputation uses training folds only.
- Train before **2017-11-18**, with three expanding validation folds. Minimize rating MAE; for finances require **≥70% mean validation hit precision**, then maximize hit F1. Also report macro F1/moderate recall; test performance may differ.

Details: [data](data/README.md), [modeling](Modeling_scripts/README.md), [diagnostics](Modeling_scripts/DIAGNOSTICS.md), [shared goals](Modeling_scripts/project_policy.py). `model_data_success_2026-11-04.csv` is an inactive binary snapshot.
