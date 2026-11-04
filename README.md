# DS6021_Group2_FinalProject

## Project question
How accurately can we predict a movie's TMDB viewer rating and classify its box-office performance as a hit or bust using movie characteristics and earlier film history? A hit means reported worldwide revenue reaches at least 2.5 times production budget.

## Repository structure and run order

| Folder / file | Purpose |
|---|---|
| `collecting_and_cleaning_data_scripts/` | Source collection, IMDb actor supplement, joining, and cleaning. |
| `Feature_engineering_scripts/` | Prepare features for modeling; see its [README](Feature_engineering_scripts/README.md). |
| `Modeling_scripts/` | Separate `rating_models/` regressors and `success_models/` hit/bust classifiers; see the folder READMEs. |
| `data/` | Source data, intermediate tables, and model snapshots; see its README for schemas and counts. |
| `CODEBOOK.md` | Variable definitions across TMDB, IMDb, and revised features. |
| `requirements.txt` | Python dependencies. |

Use `/opt/miniconda3/envs/ds6001/bin/python` (the `ds6001` notebook kernel). Run each notebook from its containing folder so `../data/` resolves correctly. The root `.env` holds credentials and is ignored by Git.

1. Source notebooks: `pulling_tmdb_data.ipynb` and `pulling_imdb_data.ipynb` in the collection folder. Skip these when using the installed CSVs.
2. `collecting_and_cleaning_data_scripts/combining_data.ipynb` creates the matched movie–director table.
3. `collecting_and_cleaning_data_scripts/data_cleaning.ipynb` creates one cleaned row per movie and the binary target.
4. The actor-backfill script creates the early-history supplement if it needs rebuilding.
5. `Feature_engineering_scripts/feature_engineering.ipynb` verifies or creates `data/model_data.csv`.
6. `Modeling_scripts/rating_models/linear_model.ipynb` is the linear rating baseline; the modeling folder also contains Random Forest, Ridge, Support Vector, and Gradient Boosting regressors. See its README for comparisons.

Collection, joining, and cleaning notebooks can overwrite their outputs. The feature notebook refuses to replace a differing existing model snapshot. Reading data for exploration does not change a CSV.

Feature engineering prepares movie characteristics and earlier-film history for modeling. See the [feature engineering README](Feature_engineering_scripts/README.md) for the methods, inputs, and output details.

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
