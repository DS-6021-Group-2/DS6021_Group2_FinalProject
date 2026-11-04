# Dataset guide

## Which file should I use?

| Task | Model input | Movies | Columns | Target |
|---|---|---:|---:|---|
| Predict viewer ratings | [model_data_rating.csv](model_data_rating.csv) | **13,276** | 45 | `vote_average` |
| Predict hit or bust | [model_data_success.csv](model_data_success.csv) | **6,919** | 46 | `success` |

Both contain **40 predictors** and four identifier/date columns. The success table also retains `vote_average`, but neither current-movie outcome belongs in its predictor inputs. The old shared model filename has been replaced by these two files.

## Why are the sizes different?

**Rating data:** a movie can have a valid viewer rating even without reported finances. This table keeps eligible rated movies from the combined data without budget/revenue filters. It has **6,357 more movies** than the success table.

**Success data:** the label needs reliable budget and revenue values. The cohort keeps budgets of at least **$250,000** and revenues of at least **$1,000**. Revenue at least 2.5 times budget gives `success = 1`; otherwise it is 0. Missing financial values are not labeled as busts.

The success CSV is the former model table renamed without changing its contents. Shared movies have identical predictors and ratings in both files. Missing runtimes (8 for ratings, 3 for success) are left for training-only imputation.

## Other files

| File | Rows | What one row represents |
|---|---:|---|
| `tmdb_movies.csv` | 14,509 | One TMDB movie. |
| `imdb_movies.csv` | 764,246 | One movie–director pair. |
| `combined_dataset.csv` | 14,824 | One matched movie–director pair; **13,276 unique movies**. |
| `cleaned_movies.csv` | 6,919 | One financially eligible movie before feature engineering. |
| `imdb_lead_actor_1985_1990.csv` | 3,550 | One earlier movie–actor credit. |
| `lead_actor_backfill.csv` | 6,919 | One original movie's actor match and additional history count. |

A co-directed film has multiple rows in the director/combined tables. Model tables have **one row per movie**. See the [codebook](../CODEBOOK.md) for variable definitions.

## How model files are made

Run [feature_engineering.ipynb](../Feature_engineering_scripts/feature_engineering.ipynb): **Step 13** verifies/writes the success CSV; **Step 14** builds the rating CSV. Existing identical exports are reused; differing exports stop for review. Source and cleaned files are inputs.

## Coverage and file handling

Earlier histories exclude current/same-year/later films, but they are incomplete snapshots. The actor supplement was matched for the original success cohort only. Extra early-window rating movies without a verified supplement keep observed TMDB counts and are flagged; **528 rating rows** have an unavailable early supplement.

The sample favors movies with enough TMDB votes; it is not every movie. Large IMDb `.tsv.gz` downloads are local rebuild inputs and are ignored by Git. Existing CSVs do not need another download.

Reading data does not change a CSV. Export code and incoming Git changes can change files; `.gitignore` does not protect tracked datasets from overwrites.

`log_budget` uses positive reported budgets. The rating table keeps all movies, including **5,439 unknown budgets** stored as missing values; each training pipeline learns its own median for imputation. The success table has no missing budgets. Source and cleaned CSVs are unchanged.
