# Collection and cleaning

This folder gets the source data, joins TMDB movies to IMDb directors, and prepares the financial labels. **Use the saved CSVs unless you intend to rebuild them.**

## Run order

Use the **`ds6001` kernel**, run notebooks from this folder, and follow this order:

| Step | Notebook | Output in `../data/` |
|---|---|---|
| 1 | [pulling_tmdb_data.ipynb](pulling_tmdb_data.ipynb) | `tmdb_movies.csv` — 14,509 movies |
| 2 | [pulling_imdb_data.ipynb](pulling_imdb_data.ipynb) | `imdb_movies.csv` — 764,246 movie–director pairs |
| 3 | [combining_data.ipynb](combining_data.ipynb) | `combined_dataset.csv` — 14,824 matched movie–director pairs |
| 4 | [data_cleaning.ipynb](data_cleaning.ipynb) | `cleaned_movies.csv` — 6,919 financially eligible movies |

Collection, joining, and cleaning can overwrite their outputs. Reading the installed CSVs for analysis does not require rerunning these steps.

## What we collected

**TMDB:** movies released in 1990–2025 with at least 100 votes, selecting up to 500 per year ordered by vote count. This is a sample, not every movie. Details include ratings, finances, genres, cast, companies, collections, and US certification.

**IMDb:** feature-film identities, director IDs/names, and release years. A movie with two directors has two rows. Join with TMDB using **`imdb_id`**, rather than titles or director names.

The join retains **13,276 unique movies**. Co-directors explain why the combined CSV has **14,824 rows**. Unmatched source movies are excluded.

## Financial cleaning

| Decision | What the code does |
|---|---|
| One row per movie | Groups director IDs/names before movie-level analysis. |
| Usable financial outcomes | Requires budget and revenue above zero. |
| Financial floors | Keeps budgets ≥ 250,000 dollars and revenues ≥ 1,000 dollars. This also excludes some valid low-budget films. |
| Target | `success = 1` when revenue ≥ 2.5 × budget; otherwise 0. |
| Runtime | Changes zero to missing for later training-only imputation. |
| Certification | Fills missing values with `Unknown`. Groups R and NC-17 together. |

The result has **2,504 hits out of 6,919 movies**. Zero financial values are treated as unreported, not automatic busts. The separate rating dataset is later built from the broader combined data without these financial filters; see [feature engineering](../Feature_engineering_scripts/README.md).

## Setup for rebuilding

Source snapshots were collected on **October 28, 2026**; later pulls can change.

- **TMDB:** put the Read Access Token in the repository-root `.env` as `token=<your token>`. The file is ignored by Git. [TMDB documentation](https://developer.themoviedb.org/).
- **IMDb director data:** place `title.basics.tsv.gz`, `title.crew.tsv.gz`, and `name.basics.tsv.gz` in `data/`. [IMDb downloads](https://datasets.imdbws.com/).
- **Actor supplement:** also needs `title.principals.tsv.gz`. Large downloads stay ignored by Git. IMDb datasets are for personal/non-commercial use.

## Other files

| File | Purpose |
|---|---|
| [api_access.ipynb](api_access.ipynb) | Preview the saved TMDB CSV without an API request. |
| [tmdb.py](tmdb.py) | Optional TMDB client/connection check. Running it makes an API request. |
| [imdb_actor_backfill.py](imdb_actor_backfill.py) | Build the original cohort's 1985–1990 actor supplement. Stop if an existing output differs. |

For the actor script, run from the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python collecting_and_cleaning_data_scripts/imdb_actor_backfill.py
```
