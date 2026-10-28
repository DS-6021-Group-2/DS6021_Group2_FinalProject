# DS6021_Group2_FinalProject

## Data sources

### IMDb Non-Commercial Datasets
- Source: https://datasets.imdbws.com/
- Downloaded: 10/28/2026. IMDb refreshes these files daily, so a later download may produce slightly different results.
- Files used by `pulling_imdb_data.ipynb`:
  - `title.basics.tsv.gz` (title, title type, and release year)
  - `title.crew.tsv.gz` (director IDs)
  - `name.basics.tsv.gz` (director names)
- These files are too large for GitHub and are excluded by `.gitignore`. To re-run the notebook, download the three
  files from the link above and place them, still gzipped, in the `data/` folder.
- IMDb permits use of these datasets for personal and non-commercial purposes only.
