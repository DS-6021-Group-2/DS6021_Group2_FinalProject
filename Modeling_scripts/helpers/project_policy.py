"""Shared evaluation policy for every active model and the app."""
CUTOFF_DATE = '2017-11-18'
N_VALIDATION_SPLITS = 3
RATING_SELECTION_METRIC = 'MAE'
RATING_METRIC_GOALS = {'MAE': .50, 'RMSE': .65, 'R2': .30}
RATING_STRETCH_GOALS = {'MAE': .45, 'R2': .40}
SUCCESS_SELECTION_METRIC = 'Hit_F1'
SUCCESS_MIN_VALIDATION_PRECISION = .70
SUCCESS_THRESHOLDS = (.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90)
BASE_THRESHOLD = .50
OVERLAP_MAE_TOLERANCE = .002
OVERLAP_HIT_F1_TOLERANCE = .005
OVERLAP_MACRO_F1_TOLERANCE = .01
DIRECTOR_HISTORY_FEATURES = (
    'dir_prior_success_rate_smoothed', 'dir_prior_films_log1p', 'dir_years_active_log1p',
    'dir_prior_known_films_log1p', 'dir_first_film', 'dir_no_track_record',
    'dir_prior_rating_smoothed', 'dir_prior_rated_films_log1p', 'dir_no_rating_history',
)


def check_objective(controls, task):
    expected = SUCCESS_SELECTION_METRIC if task == 'success' else RATING_SELECTION_METRIC
    assert controls['TUNING_METRIC'] == expected, 'Use the shared project selection metric.'
    if task == 'success':
        assert controls['MIN_VALIDATION_PRECISION'] == SUCCESS_MIN_VALIDATION_PRECISION, 'Use the shared hit precision requirement.'
