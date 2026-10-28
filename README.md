# DS6021_Group2_FinalProject

## Project question
Using only information available before a film's release, can we predict whether it will be a box-office success, defined as worldwide gross revenue of at least 2.5 times its production budget? We frame this as a data science firm helping studios and theaters identify films that are likely to be profitable.

## Repository structure and run order

| Folder / file | Purpose |
|---|---|
| `collecting_and_cleaning_data_scripts/` | Source collection, IMDb actor supplement, joining, and cleaning. |
| `Feature_engineering_scripts/` | Derive revised model predictors and inspect feature distributions. |
| `Modeling_scripts/` | Exploratory movie overview; predictive modeling can be added here. |
| `data/` | Source data, intermediate tables, and model snapshots; see its README for schemas and counts. |
| `CODEBOOK.md` | Variable definitions across TMDB, IMDb, and revised features. |
| `requirements.txt` | Python dependencies. |

Use `/opt/miniconda3/envs/ds6001/bin/python` (the `ds6001` notebook kernel). Run each notebook from its containing folder so `../data/` resolves correctly. The root `.env` holds credentials and is ignored by Git.

1. Source notebooks: `pulling_tmdb_data.ipynb` and `pulling_imdb_data.ipynb` in the collection folder. Skip these when using the installed CSVs.
2. `collecting_and_cleaning_data_scripts/combining_data.ipynb` creates the matched movie–director table.
3. `collecting_and_cleaning_data_scripts/data_cleaning.ipynb` creates one cleaned row per movie and the binary target.
4. The actor-backfill script creates the early-history supplement if it needs rebuilding.
5. `Feature_engineering_scripts/feature_engineering.ipynb` verifies or creates the revised feature snapshot.
6. `Modeling_scripts/first_model.ipynb` is an optional overview of the broader combined data, despite its original filename; it does not fit a model.

Collection, joining, and cleaning notebooks can overwrite their outputs. The revised feature notebook preserves the baseline and refuses to replace a differing revised snapshot. Reading data for exploration does not change a CSV.

## Feature-engineering revision (October 28, 2026)

`Feature_engineering_scripts/feature_engineering.ipynb` keeps budget and log-budget graphs for interpretation but excludes both from the revised predictors. It tries `log1p` for director/actor/company/franchise experience counts and smooths director success rates with a fixed five-film prior based only on earlier-year outcomes (neutral 0.5 when no prior global outcomes exist). The original rates and counts remain available in the notebook for interpretation.

The lead-actor backfill is limited to **IMDb feature films released in 1985–1990**; it does not restore the discarded total-career actor-history experiment. It matches the existing first-listed TMDB actor to an acting principal on the same IMDb movie using normalized names, then adds earlier five-year-window credits absent from the existing TMDB reference sample. Same-year/current films are excluded. Failed matches or disagreeing target years retain the original count and are flagged. Other actors, companies, and later missing movie coverage are not expanded.

Run order for this optional revision:

1. Place `title.basics.tsv.gz`, `title.principals.tsv.gz`, and `name.basics.tsv.gz` from the IMDb public datasets in `data/` (ignored by Git).
2. Run `collecting_and_cleaning_data_scripts/imdb_actor_backfill.py` with the project Python environment. It creates `data/imdb_lead_actor_1985_1990.csv` and `data/lead_actor_backfill.csv` without changing the existing datasets.
3. Run `Feature_engineering_scripts/feature_engineering.ipynb` to create **`data/model_data_revised.csv`**, leaving `model_data.csv` and `cleaned_movies.csv` unchanged.
4. Evaluate revised features with a chronological split and training-only preprocessing before treating them as a final feature set. There is currently no saved comparison notebook in `Modeling_scripts/`.

No raw source CSVs are written by these feature-engineering steps. New snapshots refuse to overwrite differing existing outputs.

## Data sources

### TMDB API
- Source: https://developer.themoviedb.org/
- Pulled: 10/28/2026. TMDB data changes over time, so a later pull may produce slightly different results.
- Running `pulling_tmdb_data.ipynb` requires a free TMDB API Read Access Token, saved in a `.env` file in the project folder as `token=<your token>`. The `.env` file is excluded by `.gitignore`.

### IMDb Non-Commercial Datasets
- Source: https://datasets.imdbws.com/
- Downloaded: 10/28/2026. IMDb refreshes these files daily, so a later download may produce slightly different results.
- Files used by `collecting_and_cleaning_data_scripts/pulling_imdb_data.ipynb`:
  - `title.basics.tsv.gz` (title, title type, and release year)
  - `title.crew.tsv.gz` (director IDs)
  - `name.basics.tsv.gz` (director names)
- These files are too large for GitHub and are excluded by `.gitignore`. To re-run the notebook, download the three
  files from the link above and place them, still gzipped, in the `data/` folder.
- IMDb permits use of these datasets for personal and non-commercial purposes only.

### Initial revision findings

The 1985–1990 IMDb supplement adds earlier credits for 493 cleaned movies; 26 of the 720 movies whose windows overlap these years have an unavailable match or inconsistent target year and retain their original counts. The revised table keeps all 6,919 movies and excludes budget/log budget. The revised features remain trials; the current repository does not include a reproducible model comparison establishing an improvement.
