## This folder contains the scripts that pull the TMDB data through an API (`tmdb_movies.csv`), build the IMDb director data (`imdb_movies.csv`), merge the two into `combined_dataset.csv`, and clean the result into `cleaned_movies.csv`.

**Run order:** `pulling_tmdb_data.ipynb` → `pulling_imdb_data.ipynb` → `combining_data.ipynb` → `data_cleaning.ipynb`. All notebooks read from and write to the `../data/` folder.


**TMDB Pulling Process:**

Available online, [TMDB](https://developer.themoviedb.org/reference/discover-movie?utm_source) has a free API key request. 

Once getting the key, we pulled all movies that had at least 100 ratings, using `vote_count.gte=100`. This was done so that vote_average is based on enough votes to be meaningful. It was looped over release years 1990 to 2025 and then saved as a CSV. For each movie we also collected its franchise collection (`collection_id`, `collection_name`) and its US certification (`us_certification`).

This resulted in 14,509 rows saved to `tmdb_movies.csv`.

Because movies were selected by their current vote count, which is only known after release, movies that performed poorly are likely under-represented. This is a limitation of the dataset.

***

**IMDB Pulling Process:**

IMDb does not provide an API for this data, so we used the [IMDb Non-Commercial Datasets](https://datasets.imdbws.com/), downloaded on 10/28/2026. Three files are needed in the `data/` folder: `title.basics.tsv.gz`, `title.crew.tsv.gz`, and `name.basics.tsv.gz`. These files are too large for GitHub, so they are excluded by `.gitignore` and must be downloaded before running `pulling_imdb_data.ipynb`.

The notebook keeps only titles that IMDb lists as `movie` (760,792 titles), adds their director IDs from `title.crew` and director names from `name.basics`, and drops movies without a director (679,726 movies remain). It keeps `startYear` so that later features can use only films released earlier. The output has one row per movie and director pair, 764,246 rows covering 679,688 movies, saved as `imdb_movies.csv`. The director IDs (`nconst`) identify each director reliably, since director names are sometimes written differently in TMDB and IMDb.

***

**Merging Data Process**

Both datasets contain a column called `imdb_id` that serves as our linking variable. Before merging, `imdb_id` was checked for missing values and duplicates.

TMDB had 10 movies with a missing `imdb_id`, and its 9 repeated IDs were those blanks counted as repeats of each other, so these 10 rows were dropped. IMDb had no missing IDs. Its repeated IDs are expected, because a movie with two directors has two rows.

An inner merge was done on `imdb_id` with the suffixes `_tmdb` and `_imdb` on the overlapping columns. It kept 13,276 of the 14,499 TMDB movies; the other 1,223 are not listed as `movie` in IMDb. The combined dataset has 14,824 rows (one per movie and director pair) and 22 columns and is saved as `combined_dataset.csv`.

Titles match exactly in about 92% of rows. The rest are the same movies under different title versions, such as "Nikita" and "La Femme Nikita", which is expected because the merge uses `imdb_id`, not titles.

***

**Data Cleaning Process**

`data_cleaning.ipynb` collapses the combined data to one row per movie and prepares it for modeling. The main steps are listed below, and each step is explained in the notebook.

1. Collapse to one row per movie, keeping the IMDb director IDs as a `|` separated list (13,276 movies).
2. Convert `release_date` to a date and add `release_year` and `release_month`.
3. Add `is_sequel`, which is 1 only when an earlier-released movie in the same collection exists.
4. Drop movies whose budget or revenue is 0, since TMDB uses 0 for an unreported value and the target needs both (7,031 movies remain).
5. Drop budgets under $250,000 (6,928 remain) and revenues under $1,000 (6,919 remain).
6. Create the target `success`, which is 1 when worldwide revenue is at least 2.5 times the production budget.
7. Set runtimes of 0 to missing, fill missing certification with "Unknown", and merge NC-17 with R.

The cleaned dataset has 6,919 movies and 28 columns, and 2,504 of the movies (36.2%) are successes. It is saved as `cleaned_movies.csv`.

Decisions agreed by the team:

| Decision | Choice | Reason |
|---|---|---|
| Budget floor | Drop budgets under $250,000 | Removes unit errors and micro-budget movies with extreme revenue to budget ratios; follows a common definition of a low-budget film |
| Revenue floor | Drop revenues under $1,000 | Values such as $1 or $27 cannot be real worldwide totals, so their labels cannot be trusted |
| 2025 movies | Keep | Their success rate is close to the overall rate |
| Missing certification | Fill with "Unknown", kept separate from "NR" | Not Rated is recorded information, while Unknown is not |
| NC-17 | Merge with R into "R/NC-17" | Only 20 movies, too few for a model to learn a reliable effect |
| Runtime of 0 | Set to missing | Filled later in the model pipeline using training data only |

`revenue`, `revenue_to_budget`, `popularity`, `vote_count`, and `vote_average` are known only after release or are built from the target, so they must never be used as predictors for the current movie.
