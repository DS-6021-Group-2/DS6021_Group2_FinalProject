## This folder contains scripts that were used to pull the `tmdb_movies.csv` data using an API and to upload the `imdb_movies.csv` data. Additionally, it contains the script that merged the tmdb and imdb data into `combined_dataset.csv`.


**TMDB Pulling Process:**

Available online, [TMDB](https://developer.themoviedb.org/reference/discover-movie?utm_source) has a free API key request. 

Once getting the key, we pulled all movies that had at least 100 ratings, using `vote_count.gte=100`. This was done so that vote_average is based on enough votes to be meaningful. It was looped over release years 1990 to 2025 and then saved as a CSV. 

This resulted in 14,509 rows saved to `tmdb_movies.csv`.

***

**IMDB Pulling Process:**






***

**Merging Data Process**
Both datasets contained a column called `imdb_id` that served as out linking variable. Before merging, `imdb_id` was checked for missing values and duplicates. 

After checking, Each table had 10 missing IDs and 9 duplicates (assuming they were blanks). The 10 rows with a missing `imdb_id` were dropped from each table. Inner-merge was done on the tables on `imdb_id`and suffixes `_tmdb` and `_dmb` on the overlapping columns. The final dataset had 14,824 rows and 22 columns


Lastly, we confirmed the `title` columns match in 100% of rows and saved the result as `combined_dataset.csv`




