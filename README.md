# DS6021_Group2_FinalProject

## Project question
Using only information available before a film's release, can we predict whether it will be a box-office success, defined as worldwide gross revenue of at least 2.5 times its production budget? We frame this as a data science firm helping studios and theaters identify films that are likely to be profitable.

## Repository structure and run order
- `collecting_and_cleaning_data_scripts/` contains the data notebooks, run in this order:
  1. `pulling_tmdb_data.ipynb` pulls movie details from the TMDB API into `data/tmdb_movies.csv`.
  2. `pulling_imdb_data.ipynb` builds director data from the IMDb files into `data/imdb_movies.csv`.
  3. `combining_data.ipynb` merges the two into `data/combined_dataset.csv`.
  4. `data_cleaning.ipynb` cleans the data and creates the target, saved as `data/cleaned_movies.csv` (6,919 movies, 36.2% successes).
- `collecting_and_cleaning_data_scripts/README.md` describes each step and the cleaning decisions in detail.
- `data/` holds the raw and cleaned CSV files.

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
