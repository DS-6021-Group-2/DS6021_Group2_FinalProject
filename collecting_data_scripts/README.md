This folder contains scripts that were used to pull the `tmdb_movies.csv` data using an API and to upload the `imdb_movies.csv` data. Additionally, it contains the script that merged the tmdb and imdb data into `combined_dataset.csv`.

*** 

**TMDB pulling process:**

Available online, [TMDB](https://developer.themoviedb.org/reference/discover-movie?utm_source) has a free API key request. Once getting the key, we pulled all movies that had at least 100 ratings, using `vote_count.gte=100`. This was done so that vote_average is based on enough votes to be meaningful. It was looped over release years 1990 to 2025 and then saved as a CSV. This resulted in 14,509 rows saved to `tmdb_movies.csv`.

***

**IMDB pulling process**






***

**Merging Data Process**




