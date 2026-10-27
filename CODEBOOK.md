# TMDB Dataset Codebook

`data/tmdb_movies.csv` contains one row per movie and 15 variables. Source: TMDB movie details and credits, collected by `pulling_tmdb_data.ipynb`.

The original selection used release years **1990–2025**, **at least 100 votes**, and up to **500 movies per year** by descending vote count. The notebook has editable collection settings. Default `reference` mode copies the Downloads CSV exactly (14,000 movies). Optional `discover` mode uses configurable years, votes, pages per year, and total movie limit; it may select different movies.

| Variable | Type / unit | Description |
|---|---|---|
| `tmdb_id` | Integer ID | Unique TMDB movie identifier. |
| `imdb_id` | Text ID | IMDb title identifier for joining movie records. |
| `title` | Text | Movie title; not necessarily unique. |
| `release_date` | YYYY-MM-DD | Release date; derive release year and month. |
| `budget` | Integer, USD | Reported production budget; not inflation-adjusted. |
| `revenue` | Integer, USD | Reported gross revenue; not inflation-adjusted. |
| `runtime` | Integer, minutes | Movie running time. |
| `genres` | Text list | Genre names. |
| `original_language` | Category | Original-language code. |
| `popularity` | Decimal | TMDB popularity score at retrieval. |
| `production_companies` | Text list | Production-company names. |
| `director` | Text list | Director names from credits. |
| `top_cast` | Text list | First five cast names returned in credits. |
| `vote_count` | Integer | Number of TMDB viewer votes at retrieval. |
| `vote_average` | Decimal, 0–10 | TMDB viewer rating; regression target. |

Text lists use a pipe (`|`) separator. Blank cells indicate missing values. The en-US API setting controls returned text, not the movie’s original language.

## Project targets and planned features

- **Rating:** predict `vote_average`.
- **Box-office performance:** calculate `revenue / budget` and classify bust, average, or hit. The **2.5× budget** benchmark is specified; the second boundary and equality rules still need agreement. This ratio is not net profit.
- **Director history (not yet included):** earlier-film count, mean rating, mean revenue-to-budget ratio, hit rate, and years of experience. Match director identities and use only films released before the current movie.

## Cleaning and modeling notes

- Investigate zero budgets, revenues, and runtimes; do not automatically classify zero revenue as a bust. A zero budget makes the ratio undefined.
- Use separate eligible samples for rating and financial models; missing financial data need not exclude a movie from rating prediction.
- IDs are join keys. Director and cast names require identity matching; `imdb_id` identifies a movie, not a person.
- For pre-release models, exclude the current movie’s revenue, rating, popularity, vote count, and target-derived fields from predictors.
- Split chronologically; fit preprocessing on training data only. Current ratings of earlier films are retrospective, not historical snapshots.

API references: [movie details](https://developer.themoviedb.org/reference/movie-details), [credits](https://developer.themoviedb.org/reference/movie-credits).
