"""Shared notebook workflow: chronological tuning, evaluation and reporting.

AI-assisted refactor. Estimators/preprocessing stay visible in each notebook.
No dataset is written. Only training folds select features, parameters or thresholds.
"""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             make_scorer, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit

from financial_outcomes import (CLASSES, HIT_CLASS, class_probabilities,
                                predict_classes, financial_metrics)
from overlap_review import review_overlap, prepare_trimmed_model
from model_diagnostics import diagnose, show_diagnostics
from project_policy import check_objective, DIRECTOR_HISTORY_FEATURES

NUMERIC_FEATURES = [
    'runtime', 'log_budget', 'release_year', 'n_genres', 'imdb_director_count',
    'dir_prior_success_rate_smoothed', 'dir_prior_rating_smoothed',
    'dir_prior_rated_films_log1p', 'dir_prior_films_log1p',
    'dir_years_active_log1p', 'dir_prior_known_films_log1p',
    'lead_actor_films_5yr_backfilled_log1p', 'company_films_5yr_log1p',
    'franchise_prior_films_log1p',
]
CATEGORICAL_FEATURES = ['us_certification', 'release_month']
DIRECTOR_RATING_FEATURES = ['dir_prior_rating_smoothed', 'dir_prior_rated_films_log1p', 'dir_no_rating_history']
CONTROL_NAMES = [
    'DATA_FILE', 'CUTOFF_DATE', 'N_VALIDATION_SPLITS', 'MODEL_SETTINGS', 'PARAM_GRID',
    'TUNING_METRIC', 'FEATURE_GROUPS_TO_COMPARE', 'MIN_VALIDATION_PRECISION',
    'THRESHOLDS', 'BASE_THRESHOLD', 'PCA_SETTINGS', 'RATING_GOALS', 'STRETCH_GOALS',
    'SHOW_EXTRA_COMPARISONS', 'SHOW_COEFFICIENTS', 'SHOW_DIAGNOSTIC_DETAILS',
    'OVERLAP_TRIMS', 'OVERLAP_MAE_TOLERANCE', 'OVERLAP_HIT_F1_TOLERANCE', 'OVERLAP_MACRO_F1_TOLERANCE',
    'EXCLUDED_PREDICTORS', 'FEATURE_PRESETS',
]


def chronological_folds(dates, n_splits):
    """Split unique dates: equal release dates never cross a fold boundary."""
    unique = np.sort(dates.unique())
    splits, periods = [], []
    for fold, (earlier, later) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(unique), 1):
        fit = np.flatnonzero(dates.isin(unique[earlier]).to_numpy())
        val = np.flatnonzero(dates.isin(unique[later]).to_numpy())
        assert dates.iloc[fit].max() < dates.iloc[val].min()
        assert set(fit).isdisjoint(val)
        splits.append((fit, val))
        periods.append({'Fold': fold, 'Training movies': len(fit), 'Validation movies': len(val),
                        'Training ends': dates.iloc[fit].max().date(),
                        'Validation starts': dates.iloc[val].min().date(),
                        'Validation ends': dates.iloc[val].max().date()})
    return splits, pd.DataFrame(periods)


def load_data(path, task, cutoff, n_splits):
    """Read the canonical feature table; validate targets/exclusions and split dates."""
    assert task in ['rating', 'success']
    assert isinstance(n_splits, int) and n_splits >= 2
    movies = pd.read_csv(path, parse_dates=['release_date'])
    movies = movies.sort_values('release_date', kind='stable').reset_index(drop=True)
    assert movies.tmdb_id.is_unique and movies.release_date.notna().all()
    assert movies.vote_average.notna().all() and movies.vote_average.between(0, 10).all()
    ids = ['tmdb_id', 'imdb_id', 'title_tmdb', 'release_date']
    targets = ['success', 'vote_average'] if task == 'success' else ['vote_average']
    if task == 'success':assert movies.success.isin(CLASSES).all()
    binary = ['is_sequel', 'is_english', 'dir_first_film', 'dir_no_track_record',
              'backfill_unavailable', 'dir_no_rating_history'] + [c for c in movies if c.startswith('genre_')]
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + binary
    assert len(features) == len(set(features))
    assert set(movies) == set(ids + targets + features)
    assert set(features).isdisjoint(ids + targets + ['budget', 'revenue', 'revenue_to_budget', 'popularity', 'vote_count'])
    assert movies[binary].isin([0, 1]).all().all()
    x = movies[features].copy()
    for c in CATEGORICAL_FEATURES:
        x[c] = x[c].map(lambda value: str(value) if pd.notna(value) else np.nan)
    y = movies[targets[0]].astype(int if task == 'success' else float).copy()
    training = movies.release_date.lt(pd.Timestamp(cutoff))
    assert training.any() and (~training).any()
    assert movies.loc[training, 'release_date'].max() < movies.loc[~training, 'release_date'].min()
    folds, periods = chronological_folds(movies.loc[training, 'release_date'], n_splits)
    if task == 'success':
        assert set(y.loc[training]) == set(CLASSES) and set(y.loc[~training]) == set(CLASSES)
        for fit, val in folds:
            assert set(y.loc[training].iloc[fit]) == set(CLASSES)
            assert set(y.loc[training].iloc[val]) == set(CLASSES)
    print(f'{len(movies):,} movies | {int(training.sum()):,} training / {int((~training).sum()):,} later test | {len(features)} available predictors')
    return {'task': task, 'movies': movies, 'X': x, 'y': y,
            'X_train': x.loc[training].copy(), 'X_test': x.loc[~training].copy(),
            'y_train': y.loc[training].copy(), 'y_test': y.loc[~training].copy(),
            'train_mask': training, 'test_mask': ~training, 'tuning_splits': folds,
            'tuning_periods': periods, 'ID_COLUMNS': ids, 'TARGET_COLUMNS': targets,
            'NUMERIC_FEATURES': list(NUMERIC_FEATURES), 'CATEGORICAL_FEATURES': list(CATEGORICAL_FEATURES),
            'BINARY_FEATURES': binary, 'FEATURE_COLUMNS': features}


def feature_groups(data, names, presets=None):
    movie = ['runtime', 'log_budget', 'release_year', 'n_genres', 'imdb_director_count'] + data['CATEGORICAL_FEATURES'] + ['is_sequel', 'is_english'] + [c for c in data['BINARY_FEATURES'] if c.startswith('genre_')]
    director = ['dir_prior_success_rate_smoothed', 'dir_prior_films_log1p', 'dir_years_active_log1p',
                'dir_prior_known_films_log1p', 'dir_first_film', 'dir_no_track_record',
                'dir_prior_rating_smoothed', 'dir_prior_rated_films_log1p', 'dir_no_rating_history']
    actor = ['lead_actor_films_5yr_backfilled_log1p', 'backfill_unavailable']
    company = ['company_films_5yr_log1p', 'franchise_prior_films_log1p']
    groups = {'Movie characteristics': movie, '+ Director history': movie + director,
              '+ Actor history': movie + director + actor, 'All features': movie + director + actor + company}
    if presets:
        groups = {name: (data['FEATURE_COLUMNS'] if columns is None else columns) for name,columns in presets.items()}
    groups = {name:[c for c in columns if c in data['FEATURE_COLUMNS']] for name,columns in groups.items()}
    assert 'All features' in names and set(names) <= set(groups)
    assert set(groups['All features']) == set(data['FEATURE_COLUMNS'])
    return {name: groups[name] for name in names}


def rating_metrics(actual, predicted):
    return {'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)), 'R2': r2_score(actual, predicted)}


def _hit_auc(estimator, x, actual):
    return roc_auc_score(np.asarray(actual) == HIT_CLASS, class_probabilities(estimator, x)[:, HIT_CLASS])


def _scoring(classification):
    if not classification:return {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}
    return {'Hit_F1': make_scorer(f1_score, labels=[HIT_CLASS], average='macro', zero_division=0),
            'Hit_precision': make_scorer(precision_score, labels=[HIT_CLASS], average='macro', zero_division=0),
            'Hit_recall': make_scorer(recall_score, labels=[HIT_CLASS], average='macro', zero_division=0),
            'Balanced_accuracy': 'balanced_accuracy', 'Accuracy': 'accuracy', 'Macro_F1': 'f1_macro', 'ROC_AUC': _hit_auc}


def _joint_search(factory, data, controls, search, groups, scoring):
    """Reuse the all-feature search, then compare every setting in other groups."""
    configurations, labels = {'Initial settings': {}}, {}
    initial = factory(data['FEATURE_COLUMNS']).get_params()
    for parameters in ParameterGrid(controls['PARAM_GRID']):
        if all(initial[k] == v for k, v in parameters.items()):label = 'Initial settings'
        else:
            label = f'Candidate {len(configurations):02d}'
            configurations[label] = parameters.copy()
        labels[json.dumps(parameters, sort_keys=True)] = label
    rows = []
    for name, columns in groups.items():
        current = search if name == 'All features' else GridSearchCV(
            factory(columns), controls['PARAM_GRID'], scoring=scoring, refit=False,
            cv=data['tuning_splits'], n_jobs=1, return_train_score=False, error_score='raise',
        ).fit(data['X_train'], data['y_train'])
        for index, parameters in enumerate(current.cv_results_['params']):
            setting = labels[json.dumps(parameters, sort_keys=True)]
            for fold in range(controls['N_VALIDATION_SPLITS']):
                row = {'Feature set': name, 'Settings': setting, 'Fold': fold + 1}
                for metric in scoring:
                    sign = -1 if metric in ['MAE', 'RMSE'] else 1
                    row[metric] = sign * current.cv_results_[f'split{fold}_test_{metric}'][index]
                rows.append(row)
    return configurations, pd.DataFrame(rows)


def _bounded_search(factory, data, controls, search, groups, classification):
    """Preserve the original initial/best-settings feature and threshold search."""
    configurations = {'Initial settings': {}}
    initial = factory(data['FEATURE_COLUMNS']).get_params()
    if any(initial[k] != v for k, v in search.best_params_.items()):
        configurations['Settings selected in 4'] = search.best_params_.copy()
    rows = []
    for setting, parameters in configurations.items():
        for name, columns in groups.items():
            for fold, (fit, val) in enumerate(data['tuning_splits'], 1):
                candidate = factory(columns).set_params(**parameters)
                candidate.fit(data['X_train'].iloc[fit], data['y_train'].iloc[fit])
                actual = data['y_train'].iloc[val]
                if classification:
                    probability = class_probabilities(candidate, data['X_train'].iloc[val])
                    for threshold in sorted(set(controls['THRESHOLDS'] + [controls['BASE_THRESHOLD']])):
                        scores = financial_metrics(actual, predict_classes(probability, threshold), probability)
                        rows.append({'Feature set': name, 'Settings': setting, 'Threshold': threshold, 'Fold': fold,
                                     **{k.replace(' ', '_'): v for k, v in scores.items()}})
                else:
                    scores = rating_metrics(actual, candidate.predict(data['X_train'].iloc[val]))
                    rows.append({'Feature set': name, 'Settings': setting, 'Fold': fold, **scores})
    return configurations, pd.DataFrame(rows)


def _select_reference(folds, controls, classification):
    metrics = ['Hit_F1', 'Hit_precision', 'Hit_recall', 'Balanced_accuracy', 'Accuracy', 'Macro_F1', 'ROC_AUC'] if classification else ['MAE', 'RMSE', 'R2']
    keys = ['Feature set', 'Settings'] + (['Threshold'] if classification else [])
    assert folds.groupby(keys).Fold.nunique().eq(controls['N_VALIDATION_SPLITS']).all()
    assert np.isfinite(folds[metrics].to_numpy()).all()
    summary = folds.groupby(keys, sort=False).agg(
        **{m: (m, 'mean') for m in metrics}, Score_std=(controls['TUNING_METRIC'], 'std'),
    ).reset_index()
    if not classification:
        summary = summary.sort_values(controls['TUNING_METRIC'],
            ascending=controls['TUNING_METRIC'] in ['MAE', 'RMSE'], kind='stable').reset_index(drop=True)
        return summary.iloc[0].copy(), summary, None
    original = summary.loc[summary['Feature set'].eq('All features') & summary.Settings.eq('Initial settings') & summary.Threshold.eq(controls['BASE_THRESHOLD'])]
    assert len(original) == 1
    floor = float(original.iloc[0].Hit_precision) if controls['MIN_VALIDATION_PRECISION'] is None else float(controls['MIN_VALIDATION_PRECISION'])
    assert 0 <= floor <= 1
    summary['Meets precision floor'] = summary.Hit_precision.ge(floor - 1e-12)
    summary['Threshold distance'] = (summary.Threshold - .5).abs()
    eligible = summary.loc[summary['Meets precision floor'] & summary.Threshold.isin(controls['THRESHOLDS'])].sort_values(
        [controls['TUNING_METRIC'], 'Hit_precision', 'Threshold distance'],
        ascending=[False, False, True], kind='stable').reset_index(drop=True)
    return (eligible.iloc[0] if len(eligible) else original.iloc[0]).copy(), summary, floor


def train_models(factory, data, controls, joint_search=False):
    """One entry point; retain original/settings/before-trim/selected references.

    Search functions receive training data only, never X_test/y_test.
    """
    config = {k: controls[k] for k in CONTROL_NAMES if k in controls}
    classification = data['task'] == 'success'
    check_objective(config, data['task'])
    excluded = config.get('EXCLUDED_PREDICTORS', [])
    assert set(excluded) <= set(data['FEATURE_COLUMNS'])
    if excluded:
        data = data.copy()
        for name in ['FEATURE_COLUMNS','NUMERIC_FEATURES','CATEGORICAL_FEATURES','BINARY_FEATURES']:
            data[name] = [c for c in data[name] if c not in excluded]
        for name in ['X','X_train','X_test']:data[name] = data[name][data['FEATURE_COLUMNS']].copy()
    assert not (classification and joint_search)
    if classification:
        assert config['THRESHOLDS'] and all(0 < t < 1 for t in config['THRESHOLDS'])
    ns = {**data, **config, 'factory': factory}
    training = {k: data[k] for k in ['X_train', 'y_train', 'tuning_splits', 'FEATURE_COLUMNS']}
    groups = feature_groups(data, config['FEATURE_GROUPS_TO_COMPARE'], config.get('FEATURE_PRESETS'))
    original = factory(data['FEATURE_COLUMNS']).fit(data['X_train'], data['y_train'])
    baseline = (DummyClassifier(strategy='most_frequent') if classification else DummyRegressor(strategy='mean'))
    baseline.fit(np.zeros((len(data['y_train']), 1)), data['y_train'])
    scoring = _scoring(classification)
    search = GridSearchCV(factory(data['FEATURE_COLUMNS']), config['PARAM_GRID'], scoring=scoring,
        refit=config['TUNING_METRIC'], cv=data['tuning_splits'], n_jobs=1,
        return_train_score=True, error_score='raise').fit(data['X_train'], data['y_train'])
    if joint_search:
        configurations, folds = _joint_search(factory, training, config, search, groups, scoring)
    else:
        configurations, folds = _bounded_search(factory, training, config, search, groups, classification)
    reference, summary, floor = _select_reference(folds, config, classification)
    reference_columns = groups[reference['Feature set']]
    reference_parameters = configurations[reference.Settings]
    before = factory(reference_columns).set_params(**reference_parameters).fit(data['X_train'], data['y_train'])
    reference_threshold = float(reference.Threshold) if classification else None
    review = review_overlap(factory, data['X_train'], data['y_train'], data['tuning_splits'],
        reference_columns, reference_parameters, reference, configurations,
        classification=classification, thresholds=config.get('THRESHOLDS'),
        reference_threshold=reference_threshold, precision_floor=floor if classification else .70,
        presets=config['OVERLAP_TRIMS'], mae_tolerance=config['OVERLAP_MAE_TOLERANCE'],
        hit_f1_tolerance=config['OVERLAP_HIT_F1_TOLERANCE'], macro_f1_tolerance=config['OVERLAP_MACRO_F1_TOLERANCE'])
    selected = (prepare_trimmed_model(factory(review['columns']).set_params(**review['parameters']))
                if review['winner'].Trim != 'Before overlap trim' else before)
    selected.fit(data['X_train'], data['y_train'])
    original_folds = folds.loc[folds.Settings.eq('Initial settings')].copy()
    if classification:original_folds = original_folds.loc[original_folds.Threshold.eq(config['BASE_THRESHOLD'])]
    # The original settings need not be in a user-edited grid. Preserve baseline CV honestly.
    if original_folds.empty:
        original_rows = []
        for name, columns in groups.items():
            for fold, (fit, val) in enumerate(data['tuning_splits'], 1):
                candidate = factory(columns).fit(data['X_train'].iloc[fit], data['y_train'].iloc[fit])
                original_rows.append({'Feature set': name, 'Fold': fold,
                    **rating_metrics(data['y_train'].iloc[val], candidate.predict(data['X_train'].iloc[val]))})
        original_folds = pd.DataFrame(original_rows)
    ns.update({'initial_model': original, 'settings_model': search.best_estimator_, 'baseline': baseline,
        'search': search, 'SCORING': scoring, 'feature_versions': groups,
        'improvement_configurations': configurations, 'selection_fold_results': folds,
        'improvement_summary': summary, 'feature_fold_results': original_folds.rename(columns={'Feature set': 'Version'}),
        'overlap_reference_model': before, 'overlap_reference_features': list(reference_columns),
        'overlap_reference_validation': reference, 'overlap_reference_threshold': reference_threshold,
        'overlap_review': review, 'model': selected, 'winner': review['winner'],
        'selected_feature_columns': review['columns'], 'selected_parameters': review['parameters'],
        'selected_feature_set': review['winner']['Feature set']})
    if classification:
        ns.update({'selected_threshold': review['threshold'], 'precision_floor': floor,
                   'precision_requirement_met': float(review['winner'].Hit_precision) >= floor - 1e-12})
    assert set(ns['selected_feature_columns']).isdisjoint(data['ID_COLUMNS'] + data['TARGET_COLUMNS'])
    return ns


def show_validation(ns, details=False):
    from IPython.display import display
    classification = ns['task'] == 'success'
    metrics = ['Hit_precision', 'Hit_recall', 'Hit_F1', 'Macro_F1'] if classification else ['MAE', 'RMSE', 'R2']
    row = {m: ns['winner'][m] for m in metrics}
    row['Predictors'] = len(ns['selected_feature_columns'])
    if classification:row['Hit threshold'] = ns['selected_threshold']
    print('Selected using three earlier-to-later folds inside training.' if ns['N_VALIDATION_SPLITS']==3 else f"Selected using {ns['N_VALIDATION_SPLITS']} chronological training folds.")
    display(pd.DataFrame([row]).round(4))
    print('Parameters:', ns['selected_parameters'] or 'Original settings')
    print(ns['overlap_review']['status'])
    if classification and not ns['precision_requirement_met']:
        print('The validation precision requirement was infeasible; the reference is retained and does not meet the goal.')
    if details:
        print('Optional candidate details (selection estimates, not independent evaluation).')
        display(ns['tuning_periods'])
        display(ns['improvement_summary'].head(12).round(4))
        display(ns['overlap_review']['candidates'].sort_values('Predictors').head(12).round(4))


def _metrics_for(ns, estimator, x, actual, threshold):
    if ns['task'] == 'success':
        probability = class_probabilities(estimator, x)
        predicted = predict_classes(probability, threshold)
        return financial_metrics(actual, predicted, probability), predicted
    predicted = estimator.predict(x)
    return rating_metrics(actual, predicted), predicted


def show_results(ns, extra=False, coefficients=False):
    """One default results table, one task-appropriate plot; optional extras."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    classification = ns['task'] == 'success'
    threshold = ns.get('selected_threshold')
    baseline, _ = _metrics_for(ns, ns['baseline'], np.zeros((len(ns['y_test']), 1)), ns['y_test'], ns.get('BASE_THRESHOLD'))
    train, _ = _metrics_for(ns, ns['model'], ns['X_train'], ns['y_train'], threshold)
    test, prediction = _metrics_for(ns, ns['model'], ns['X_test'], ns['y_test'], threshold)
    rows = [{'Model': 'Dummy baseline', 'Split': 'Test', **baseline},
            {'Model': 'Selected', 'Split': 'Training', **train},
            {'Model': 'Selected', 'Split': 'Test', **test}]
    columns = ['Model', 'Split'] + (['Accuracy', 'Macro F1', 'Moderate recall', 'Hit precision', 'Hit recall', 'Hit F1'] if classification else ['MAE', 'RMSE', 'R2'])
    results = pd.DataFrame(rows)
    display(results[columns].round(4))
    fig, ax = plt.subplots(figsize=(6, 4))
    if classification:
        ConfusionMatrixDisplay.from_predictions(ns['y_test'], prediction, labels=CLASSES,
            display_labels=['Bust', 'Moderate', 'Hit'], cmap='Blues', colorbar=False, ax=ax)
        ax.set_title('Selected model: later-test outcomes')
    else:
        ax.scatter(ns['y_test'], prediction, alpha=.2, s=10, color='#2a78d6')
        ax.plot([0, 10], [0, 10], 'k--', linewidth=1)
        ax.set(title='Selected model: later-test ratings', xlabel='Actual rating', ylabel='Predicted rating')
    fig.tight_layout(); plt.show()
    if 'pca' in ns['model'].named_steps:show_pca_comparison(ns)
    if extra:
        print('Optional intermediate references; original, settings-only and before-trim results remain in saved reports.')
        for name, key, value in [('Original','initial_model',ns.get('BASE_THRESHOLD')), ('Settings only','settings_model',ns.get('BASE_THRESHOLD')), ('Before overlap trim','overlap_reference_model',ns['overlap_reference_threshold'])]:
            metrics, _ = _metrics_for(ns, ns[key], ns['X_test'], ns['y_test'], value)
            rows.append({'Model': name, 'Split': 'Test', **metrics})
        display(pd.DataFrame(rows).round(4))
        if not classification and 'RATING_GOALS' in ns:
            display(pd.DataFrame([{'Metric': m, 'Test': test[m], 'Goal': v,
                'Met': test[m]>=v if m=='R2' else test[m]<=v} for m,v in ns['RATING_GOALS'].items()]))
    if coefficients:show_coefficients(ns)
    ns['results'] = results
    return results


def show_pca_comparison(ns):
    """One fixed-settings PCA ablation; validation only, no new test inspection."""
    from IPython.display import display
    rows = []
    for name in ['KNN without PCA', 'KNN with PCA']:
        for fold, (fit, val) in enumerate(ns['tuning_splits'], 1):
            candidate = clone(ns['initial_model'])
            if name == 'KNN without PCA':candidate.set_params(pca='passthrough')
            candidate.fit(ns['X_train'].iloc[fit], ns['y_train'].iloc[fit])
            scores, _ = _metrics_for(ns, candidate, ns['X_train'].iloc[val], ns['y_train'].iloc[val], ns.get('BASE_THRESHOLD'))
            rows.append({'Version': name, 'Fold': fold, **scores})
    ns['pca_ablation'] = pd.DataFrame(rows)
    metrics = ['Macro F1', 'Hit precision', 'Hit F1'] if ns['task']=='success' else ['MAE','RMSE','R2']
    pca = ns['model'].named_steps['pca']
    print(f'PCA retains {pca.n_components_} components / {pca.explained_variance_ratio_.sum():.1%} of training input variance.')
    print('Fixed original settings and full inputs: mean chronological validation scores, with versus without PCA.')
    display(ns['pca_ablation'].groupby('Version', sort=False)[metrics].mean().round(4))


def show_coefficients(ns):
    from IPython.display import display
    estimator = ns['model'].steps[-1][1]
    if not hasattr(estimator, 'coef_'):return
    names = ns['model'][:-1].get_feature_names_out()
    coefficients = np.atleast_2d(estimator.coef_)
    rows=[]
    for i, values in enumerate(coefficients):
        for j in np.argsort(-abs(values))[:10]:
            rows.append({'Class': i if len(coefficients)>1 else 'Rating', 'Feature': names[j], 'Coefficient': values[j]})
    print('Optional largest coefficients. Scales and correlated inputs limit direct importance/causal interpretation.')
    display(pd.DataFrame(rows))


def show_diagnostics_summary(ns):
    """Default: short safeguards + before/after overlap; full tables opt in."""
    from IPython.display import display
    if ns.get('SHOW_DIAGNOSTIC_DETAILS', False):
        d = show_diagnostics(ns)
    else:
        d = diagnose(ns)
        c = d['collinearity']['summary'].iloc[0]
        missing = d['missingness']['Missing training']
        rows = [
            {'Check':'Programmed data/chronology safeguards','Result':'PASS; historical availability still unverified'},
            {'Check':'Remaining overlap','Result':f"{int(d['collinearity']['vif'].VIF.ge(10).sum())} inputs have VIF ≥10; coefficients need cautious interpretation"},
            {'Check':'Missing training inputs','Result':f'{missing.idxmax()}: {int(missing.max())} missing' if missing.max() else 'None'},
            {'Check':'Numerical fitting','Result':d['estimator'].to_dict('records')[0]},
            {'Check':'Generalization limits','Result':'Later movies, missing finances and retrospective histories can differ; use training validation for changes'},
        ]
        display(pd.DataFrame(rows))
        cols=['Stage','Predictors','Transformed inputs','VIF >=10','Maximum VIF','Removed']
        print('Overlap review: fewer redundant inputs; high VIF alone does not invalidate predictions.')
        with pd.option_context('display.max_colwidth', None):display(d['overlap_comparison'][cols].round(3))
        print('Next: compare further experiments only inside chronological training validation. Full technical tables are in diagnostic_results; set SHOW_DIAGNOSTIC_DETAILS=True to display them.')
    ns['diagnostic_results'] = d
    return d


def save_report(ns, notebook_path):
    path=Path(__file__).resolve().parents[2]/'Data_visualization/model_reports.py'
    spec=importlib.util.spec_from_file_location('notebook_model_reports',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.save_model_report(ns,notebook_path)


def compare_director_rating_history(factory, data):
    """Shared fixed-settings experiment; only chronological training folds are used."""
    rows=[]
    groups={'Without director rating history':[c for c in data['FEATURE_COLUMNS'] if c not in DIRECTOR_RATING_FEATURES],
            'With director rating history':data['FEATURE_COLUMNS']}
    for name,columns in groups.items():
        for fold,(fit,val) in enumerate(data['tuning_splits'],1):
            model=factory(columns).fit(data['X_train'].iloc[fit],data['y_train'].iloc[fit])
            metrics=rating_metrics(data['y_train'].iloc[val],model.predict(data['X_train'].iloc[val]))
            rows.append({'Version':name,'Fold':fold,**metrics})
    return pd.DataFrame(rows)
