# TMDB and IMDb Dataset Codebook

## Sources and structure

| Source file | One row represents | Purpose |
|---|---|---|
| `data/tmdb_movies.csv` | One movie | Movie characteristics, viewer ratings, and financial outcomes. |
| `data/imdb_movies.csv` | One movie–director pair | IMDb director identities and links to their films. Co-directed movies have multiple rows. |

Join sources using **`imdb_id`**, not titles or director names. IMDb `tt…` IDs identify movies; `nm…` IDs identify people.

The TMDB sample was selected from 1990–2025 releases with at least 100 votes, up to 500 movies per year ordered by vote count. The current manually installed TMDB file contains 14,509 movies and is not a complete movie census. The IMDb file covers a broader filmography and now includes release year (`startYear`), but has no full release dates, ratings, budgets, or revenues.

## TMDB source variables

| Variable | Type / unit | Definition |
|---|---|---|
| `tmdb_id` | Integer ID | Unique TMDB movie identifier. |
| `imdb_id` | Text ID | IMDb movie identifier; join key shared with IMDb. |
| `title` | Text | TMDB movie title; not necessarily unique. |
| `release_date` | YYYY-MM-DD | Release date; used to order films and derive year/month. |
| `budget` | Integer, USD | Reported production budget; not inflation-adjusted. |
| `revenue` | Integer, USD | Reported gross revenue; not inflation-adjusted. |
| `runtime` | Integer, minutes | Running time. |
| `genres` | Text list | Genre names. |
| `original_language` | Category | Original-language code; the en-US response setting does not restrict movies to English. |
| `popularity` | Decimal | TMDB popularity score at retrieval. |
| `production_companies` | Text list | Production-company names. |
| `director` | Text list | Director names from TMDB credits, without person IDs. |
| `top_cast` | Text list | First five cast names returned in credits. |
| `vote_count` | Integer | Number of viewer votes at retrieval. |
| `vote_average` | Decimal, 0–10 | TMDB viewer rating; regression target. |
| `collection_id` | Nullable integer ID | TMDB collection/franchise identifier; blank when no collection is recorded. |
| `collection_name` | Text | Name of the recorded collection/franchise. |
| `us_certification` | Category | US content certification from release information, such as PG-13 or R; not viewer rating. Blank means unknown. |

TMDB text lists use a pipe (`|`) separator. Blank cells are missing values.

## IMDb source variables

| Variable | Type | Definition |
|---|---|---|
| `imdb_id` | Text ID (`tt…`) | Movie identifier; repeats when a movie has multiple directors. |
| `nconst` | Text ID (`nm…`) | Stable IMDb person identifier for the director. |
| `director` | Text | Director name corresponding to `nconst`. |
| `title` | Text | IMDb movie title; may differ from TMDB's title. |
| `startYear` | Nullable integer year | IMDb release year, retained separately from TMDB release date; missing years remain unknown. |

The pair (`imdb_id`, `nconst`) identifies a movie–director relationship. Repeated movie IDs here are expected, not necessarily duplicate records.

## Joined variables and naming

**`first_model.ipynb`:** loads `data/combined_dataset.csv` rather than repeating the source join. It groups directors to keep one row per matched movie, retains `title_tmdb`, `title_imdb`, and `director_tmdb`, and replaces `nconst` / `director_imdb` with:

| Added variable | Type | Definition |
|---|---|---|
| `imdb_director_ids` | Text list | IMDb director IDs, pipe-separated. |
| `imdb_directors` | Text list | Unique IMDb director names, pipe-separated. |
| `imdb_director_count` | Integer | Number of directors on this movie, not their previous films. |

The in-memory `movies` table has one row per matched movie. Movies excluded by the earlier inner join are not restored. `release_date` is parsed as a date. No `imdb_matched` flag is needed because all input rows are already matched.

**`combining_data.ipynb`:** defines an inner join without aggregating directors. Its naming is different: `title_tmdb` / `title_imdb` and `director_tmdb` / `director_imdb`; `nconst` remains a separate column. Other TMDB columns keep their names, and IMDb `startYear` is retained. The updated combined file has 14,824 movie–director rows, 22 columns, and 13,276 unique movies. This join retains only matched movies and can produce several rows per movie. These names describe that notebook's join, not the `first_model.ipynb` table.

## Project use and cautions

- **Rating target:** `vote_average`.
- **Performance target:** derive `revenue / budget`; the 2.5× benchmark is specified, but the second boundary for bust/average/hit still needs agreement. The ratio is not net profit.
- **Director history (planned):** use IMDb identities with TMDB dates and outcomes to calculate earlier-film count, mean rating, and performance. Only films before the target movie qualify; coverage is limited to films with the required data. IMDb `startYear` can order films by year but does not resolve which film came first within the same year.
- Investigate zero budgets, revenues, and runtimes. Zero budget makes the ratio undefined; zero revenue is not automatically a bust.
- For pre-release models, exclude the current movie's revenue, rating, popularity, vote count, and target-derived fields from predictors. Use chronological splits and fit preprocessing on training data only.

Definitions checked against both source CSV headers and notebook join code. TMDB references: [movie details](https://developer.themoviedb.org/reference/movie-details), [credits](https://developer.themoviedb.org/reference/movie-credits).
