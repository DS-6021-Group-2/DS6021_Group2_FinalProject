# Project datasets

CSV files stay in this folder so existing notebook paths continue to work. Counts below were checked against the installed files on October 28, 2026.

| File | Rows | Columns | One row / role |
|---|---:|---:|---|
| `tmdb_movies.csv` | 14,509 | 18 | TMDB movie; source characteristics and outcomes. |
| `imdb_movies.csv` | 764,246 | 5 | Movie–director pair, including IMDb year; 679,688 distinct movies. |
| `combined_dataset.csv` | 14,824 | 22 | Matched movie–director pair; 13,276 distinct movies. |
| `cleaned_movies.csv` | 6,919 | 28 | One financially eligible movie and binary success label. |
| `model_data.csv` | 6,919 | 41 | Preserved baseline features; includes `log_budget`, which is excluded from current predictors. |
| `imdb_lead_actor_1985_1990.csv` | 3,550 | 3 | IMDb movie–actor acting-principal credit from 1985–1990. |
| `lead_actor_backfill.csv` | 6,919 | 8 | Cleaned movie's actor match, eligible extra credit count, and availability flags. |
| `model_data_revised.csv` | 6,919 | 41 | Revised predictors and unchanged labels; budget and current-movie outcome fields excluded. |

The cleaned cohort has 2,504 successes and 4,415 other movies. Success means `revenue >= 2.5 * budget`; it is a proxy based on gross revenue, not audited profit. Cleaning requires budget at least $250,000 and revenue at least $1,000. Unreported financial values cannot define this target.

All three movie-level derived snapshots and the actor backfill use the same cleaned movie identities and order. The revised model table has three missing runtime values, intentionally left for training-only imputation. No other revised columns are missing. Source fields such as collections and IMDb release years may legitimately be missing.

## How files are produced

- Collection notebooks create the two source CSVs.
- Combining and cleaning notebooks create their corresponding tables.
- `../collecting_and_cleaning_data_scripts/imdb_actor_backfill.py` creates the two actor supplements, without rewriting the source or cleaned CSVs.
- `../Feature_engineering_scripts/feature_engineering.ipynb` creates or verifies `model_data_revised.csv`. It stops if an existing revised snapshot differs.

See `../CODEBOOK.md` for variable definitions. Movie IDs, rather than titles, link datasets; repeated IMDb movie IDs in director data are expected for co-directed movies.

## Large IMDb download files

`title.basics.tsv.gz`, `title.principals.tsv.gz`, and `name.basics.tsv.gz` are local inputs for rebuilding actor history. Director collection additionally requires `title.crew.tsv.gz`. These downloads are ignored by Git. Existing CSVs can be used without redownloading them.

## Preservation and limitations

No CSV was moved or modified during this review and organization. Reading data or editing an in-memory DataFrame does not change a CSV; explicit export code can. Git ignore rules do not prevent overwrites or incoming changes to tracked files. Check diffs before pulling or committing dataset changes.

These checks establish consistency with the installed sources and pipeline; they do not verify reported budgets/revenues against external records. The vote-count-selected TMDB sample is not a census. Name matching, missing finances, retrospective snapshots, and incomplete earlier-film coverage limit interpretation.
