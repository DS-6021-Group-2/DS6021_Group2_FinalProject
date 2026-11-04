# Project datasets

CSV files stay in this folder so existing notebook paths continue to work. Counts below were checked against the installed files on October 28, 2026.

| File | Rows | Columns | One row / role |
|---|---:|---:|---|
| `tmdb_movies.csv` | 14,509 | 18 | TMDB movie; source characteristics and outcomes. |
| `imdb_movies.csv` | 764,246 | 5 | Movie–director pair, including IMDb year; 679,688 distinct movies. |
| `combined_dataset.csv` | 14,824 | 22 | Matched movie–director pair; 13,276 distinct movies. |
| `cleaned_movies.csv` | 6,919 | 28 | One financially eligible movie and binary success label. |
| `imdb_lead_actor_1985_1990.csv` | 3,550 | 3 | IMDb movie–actor acting-principal credit from 1985–1990. |
| `lead_actor_backfill.csv` | 6,919 | 8 | Cleaned movie's actor match, eligible extra credit count, and availability flags. |
| `model_data.csv` | 6,919 | 45 | Engineered predictors plus `success` and `vote_average` targets; budget and revenue excluded. |

The cleaned cohort has 2,504 successes and 4,415 other movies. Success means `revenue >= 2.5 * budget`; it is a proxy based on gross revenue, not audited profit. Cleaning requires budget at least $250,000 and revenue at least $1,000. Unreported financial values cannot define this target.

The cleaned and model tables and the actor backfill use the same cleaned movie identities and order. The revised model table has three missing runtime values, intentionally left for training-only imputation. No other model columns are missing. `vote_average` is the rating target, copied from the cleaned table by movie ID. Exclude both `success` and `vote_average` from predictors for either model; these are outcomes, not input features. Source fields such as collections and IMDb release years may legitimately be missing.

## How files are produced

- Collection notebooks create the two source CSVs.
- Combining and cleaning notebooks create their corresponding tables.
- `../collecting_and_cleaning_data_scripts/imdb_actor_backfill.py` creates the two actor supplements, without rewriting the source or cleaned CSVs.
- `../Feature_engineering_scripts/feature_engineering.ipynb` creates or verifies `model_data.csv`. It stops if an existing revised snapshot differs.

See `../CODEBOOK.md` for variable definitions. Movie IDs, rather than titles, link datasets; repeated IMDb movie IDs in director data are expected for co-directed movies.

## Large IMDb download files

`title.basics.tsv.gz`, `title.principals.tsv.gz`, and `name.basics.tsv.gz` are local inputs for rebuilding actor history. Director collection additionally requires `title.crew.tsv.gz`. These downloads are ignored by Git. Existing CSVs can be used without redownloading them.

## Preservation and limitations

`model_data.csv` now contains the revised feature table, replacing the former baseline. The separate revised filename was retired. Source, combined, cleaned, and actor-history CSVs were not changed during this promotion. The former baseline is available in Git history. Reading data or editing an in-memory DataFrame does not change a CSV; explicit export code can. Git ignore rules do not prevent overwrites or incoming changes to tracked files. Check diffs before pulling or committing dataset changes.

These checks establish consistency with the installed sources and pipeline; they do not verify reported budgets/revenues against external records. The vote-count-selected TMDB sample is not a census. Name matching, missing finances, retrospective snapshots, and incomplete earlier-film coverage limit interpretation.

## Earlier director rating features

The feature notebook adds `dir_prior_rating_smoothed`, `dir_prior_rated_films_log1p`, and `dir_no_rating_history`. They use TMDB viewer ratings in the broader matched sample, linked through IMDb director IDs, including movies without known financial values. A movie counts once across co-directors; only strictly earlier release years qualify. The mean is shrunk with a five-film prior at the strictly earlier global mean (5.0 when no prior reference films exist).

4,697 of the 6,919 model movies have an observed earlier director rating; no-history rows use the prior and flag. Raw mean/supporting count remain available in the feature notebook. All 42 previous model columns and targets were preserved when the three columns were added; source, combined, cleaned, and actor-history datasets were unchanged. Ratings are retrospective snapshots, not archived release-time values.
