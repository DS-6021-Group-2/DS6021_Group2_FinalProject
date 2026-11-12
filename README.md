# DS 6021 Movie Prediction Project

We predict two things from movie characteristics and earlier film history:

- **Viewer rating:** the movie's TMDB `vote_average`.
- **Financial success:** a **hit** if revenue reaches at least 2.5 times production budget; a **bust** otherwise.

## Start here

The datasets are already saved. Start with a simple model; collection notebooks are only needed when rebuilding the data.

| I want to… | Open |
|---|---|
| Preview the app layout | [App guide](app/README.md) |
| Predict viewer ratings | [Simple rating model](Modeling_scripts/rating_models/simple_model.ipynb) |
| Predict hits and busts | [Simple success model](Modeling_scripts/success_models/simple_model.ipynb) |
| Compare the other models | [Modeling guide](Modeling_scripts/README.md) |
| Understand the datasets or variables | [Dataset guide](data/README.md) · [Codebook](CODEBOOK.md) |

Use the **`ds6001` Python kernel** and run notebooks from their own folders, from top to bottom.

## How the project fits together

**Collect → combine → clean → create features → train and compare models**

| Folder | Purpose |
|---|---|
| [Collection and cleaning](collecting_and_cleaning_data_scripts/README.md) | Get the source data, join movies to directors, and prepare financial labels. |
| [Feature engineering](Feature_engineering_scripts/README.md) | Build the two model datasets. |
| [App](app/README.md) | Three-tab Streamlit layout; live prediction models are not connected yet. |
| [Modeling](Modeling_scripts/README.md) | Separate rating regressors and hit/bust classifiers. |
| [Visualizations](Data_visualization/README.md) | Compare all model results and inspect prediction errors. |
| [Data](data/README.md) | Saved source files and model inputs. |

Feature engineering produces **`model_data_rating.csv`** (13,276 movies) and **`model_data_success.csv`** (6,919 movies). Each model folder uses its matching dataset.

## Sources

[TMDB](https://developer.themoviedb.org/) supplies movie details, ratings, and finances. [IMDb](https://datasets.imdbws.com/) supplies film and person identities for earlier-film histories. The source snapshots were collected on October 28, 2026; later downloads can differ.

Dependencies are listed in [requirements.txt](requirements.txt). Collection credentials belong in the ignored root `.env` file; setup details are in the collection guide.
