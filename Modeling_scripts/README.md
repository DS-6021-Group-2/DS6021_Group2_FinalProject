# Modeling and exploratory analysis

`first_model.ipynb` is a read-only overview of `../data/combined_dataset.csv`. It groups co-directors into one movie row and shows financial reporting groups, certifications, and release-year differences. Its name is historical: it does not train a model or save a CSV.

Run it from this folder with the `ds6001` kernel. Its counts describe the broader matched sample, rather than only the financially eligible cleaned cohort.

For future models, use `../data/model_data_revised.csv`: `success` is the binary target, and `tmdb_id`, `imdb_id`, `title_tmdb`, and `release_date` are identifiers/split information. Exclude these identifiers from predictors. The baseline `model_data.csv` still contains `log_budget`, which must be excluded under the current project design.

Use a chronological split. Fit imputation, encoding, and scaling on training data only. Earlier-year outcome history supports rolling predictions; freeze it at the cutoff if evaluating a single forecast made at that date. Report a majority-class baseline, balanced accuracy, macro F1, and ROC AUC alongside accuracy. The repository currently contains no saved feature-comparison notebook or validated final model.
