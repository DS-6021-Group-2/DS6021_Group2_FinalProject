"""Read-only diagnostics shared by all model notebooks (AI-assisted).

Never fits an estimator or selects features/thresholds. VIF/rank are computed
on training predictors only; later-data checks are descriptive evaluations.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from financial_outcomes import CLASSES, CLASS_NAMES, class_probabilities, predict_classes, financial_metrics

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score,
    brier_score_loss, log_loss,
)


def collinearity(matrix, names):
    """Centered, standardized SVD VIF with an implicit intercept.

    Exact dependencies yield infinite VIF for participating columns, not the
    misleading finite values from blindly inverting a singular correlation matrix.
    Constants have undefined VIF and are reported separately.
    """
    x = np.asarray(matrix, dtype=float)
    assert x.ndim == 2 and len(names) == x.shape[1] and np.isfinite(x).all()
    centered = x - x.mean(axis=0)
    sd = centered.std(axis=0, ddof=1)
    active = sd > np.finfo(float).eps * np.maximum(1, np.abs(x).max(axis=0)) * 100
    vif = np.full(x.shape[1], np.nan)
    pairs = []
    if active.any():
        z = centered[:, active] / sd[active]
        u, singular, vt = np.linalg.svd(z, full_matrices=False)
        tol = max(z.shape) * np.finfo(float).eps * singular[0]
        rank = int((singular > tol).sum())
        v = vt.T
        finite_vif = (len(x) - 1) * ((v[:, :rank] / singular[:rank]) ** 2).sum(axis=1)
        # p < n for this project's fitted matrices; complete SVD covers nullspace.
        assert x.shape[1] < len(x), 'Use a dedicated high-dimensional VIF method when p >= n.'
        dependent = np.linalg.norm(v[:, rank:], axis=1) > 1e-7
        finite_vif[dependent] = np.inf
        vif[active] = np.maximum(1, finite_vif)
        condition = float(singular[0] / singular[-1]) if rank == z.shape[1] else np.inf
        corr = (z.T @ z) / (len(z) - 1)
        active_names = np.asarray(names)[active]
        for i, j in zip(*np.where(np.triu(np.abs(corr) >= 0.80, k=1))):
            pairs.append({'Feature 1': active_names[i], 'Feature 2': active_names[j],
                          'Correlation': float(corr[i, j])})
    else:
        rank, condition = 0, np.nan
    table = pd.DataFrame({'Feature': names, 'VIF': vif, 'Constant': ~active})
    table['Interpretation'] = np.select(
        [~active, np.isinf(vif), vif >= 10, vif >= 5],
        ['Undefined: constant', 'Exact linear dependence', 'High (heuristic ≥10)', 'Review (heuristic ≥5)'],
        default='Below review threshold',
    )
    return {
        'summary': pd.DataFrame([{'Inputs': x.shape[1], 'Constants': int((~active).sum()),
                                  'Nonconstant inputs': int(active.sum()), 'Centered rank': rank,
                                  'Standardized condition number': condition,
                                  'Infinite VIF': int(np.isinf(vif).sum()),
                                  'Finite VIF ≥5': int(((vif >= 5) & np.isfinite(vif)).sum())}]),
        'vif': table.sort_values('VIF', ascending=False, na_position='last').reset_index(drop=True),
        'correlations': pd.DataFrame(pairs, columns=['Feature 1', 'Feature 2', 'Correlation']).sort_values(
            'Correlation', key=lambda c: c.abs(), ascending=False),
    }


def _transformed(pipeline, x):
    transformed = pipeline[:-1].transform(x)
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    try:
        names = list(pipeline[:-1].get_feature_names_out())
    except (AttributeError, ValueError):
        names = [f'Input {i+1}' for i in range(transformed.shape[1])]
    return np.asarray(transformed, dtype=float), names


def _safe_correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


def diagnose(ns):
    """Audit the already-selected pipeline in a model notebook namespace."""
    model, movies = ns['model'], ns['movies']
    xtr, xte, ytr, yte = [ns[k] for k in ['X_train', 'X_test', 'y_train', 'y_test']]
    features = ns['selected_feature_columns']
    classification = ns['TARGET_COLUMNS'][0] == 'success'
    forbidden = {'success', 'vote_average', 'budget', 'revenue', 'revenue_to_budget',
                 'tmdb_id', 'imdb_id', 'title_tmdb', 'release_date', 'vote_count', 'popularity'}
    assert not set(features) & forbidden, 'Outcome/identifier leakage in predictors.'
    assert len(ytr) and len(yte) and movies.tmdb_id.is_unique
    assert xtr.index.equals(ytr.index) and xte.index.equals(yte.index)
    assert set(ytr.index).isdisjoint(yte.index)
    assert movies.loc[ytr.index, 'release_date'].max() < movies.loc[yte.index, 'release_date'].min()
    fold_rows = []
    for number, (fit, val) in enumerate(ns['tuning_splits'], 1):
        dates = movies.loc[ytr.index, 'release_date']
        assert dates.iloc[fit].max() < dates.iloc[val].min()
        assert set(fit).isdisjoint(val) and max([*fit, *val]) < len(ytr)
        row = {'Fold': number, 'Training rows': len(fit), 'Validation rows': len(val),
               'Training ends': dates.iloc[fit].max(), 'Validation starts': dates.iloc[val].min()}
        if classification:
            assert set(ytr.iloc[fit]) == set(CLASSES) and set(ytr.iloc[val]) == set(CLASSES)
            row.update({'Training hit fraction': ytr.iloc[fit].eq(2).mean(),
                        'Validation hit fraction': ytr.iloc[val].eq(2).mean()})
        fold_rows.append(row)
    assert len(fold_rows) == ns['N_VALIDATION_SPLITS']
    raw = pd.DataFrame({'Missing training': xtr[features].isna().sum(),
                        'Missing test': xte[features].isna().sum(),
                        'Training unique': xtr[features].nunique()})
    assert not raw['Missing training'].eq(len(xtr)).any(), 'An input is entirely missing in training.'
    preprocessor = model.named_steps['preprocessor']
    numeric = next(cols for name, _, cols in preprocessor.transformers_ if name == 'numeric')
    imputer = preprocessor.named_transformers_['numeric'].named_steps['imputer']
    if imputer.strategy == 'median':
        np.testing.assert_allclose(imputer.statistics_, xtr[numeric].median().to_numpy(), rtol=1e-10, atol=1e-10)
    numeric_pipe = preprocessor.named_transformers_['numeric']
    if 'scaler' in numeric_pipe.named_steps:
        scaler = numeric_pipe.named_steps['scaler']
        imputed = imputer.transform(xtr[numeric])
        assert int(scaler.n_samples_seen_) == len(xtr)
        np.testing.assert_allclose(scaler.mean_, imputed.mean(axis=0), rtol=1e-10, atol=1e-10)
    if 'scaler' in model.named_steps:
        scaler = model.named_steps['scaler']
        encoded = model.named_steps['preprocessor'].transform(xtr)
        assert int(scaler.n_samples_seen_) == len(xtr)
        np.testing.assert_allclose(scaler.mean_, encoded.mean(axis=0), rtol=1e-10, atol=1e-10)
    if 'pca' in model.named_steps:
        assert model.named_steps['pca'].n_samples_ == len(xtr)
        np.testing.assert_allclose(model.named_steps['pca'].mean_, model[:-2].transform(xtr).mean(axis=0), rtol=1e-10, atol=1e-10)
    train_matrix, names = _transformed(model, xtr)
    test_matrix, _ = _transformed(model, xte)
    assert np.isfinite(train_matrix).all() and np.isfinite(test_matrix).all()
    result = {'checks': pd.DataFrame([
        {'Check': 'Movie IDs, target exclusions, chronology, row alignment, fold ordering', 'Result': 'PASS'},
        {'Check': 'Training-only imputer/scaler/PCA state where applicable', 'Result': 'PASS'},
        {'Check': 'Finite transformed predictors', 'Result': 'PASS'},
    ]), 'folds': pd.DataFrame(fold_rows), 'missingness': raw,
        'collinearity': collinearity(train_matrix, names)}
    if 'pca' in model.named_steps:
        before = model[:-2].transform(xtr)
        before_names = list(model[:-2].get_feature_names_out())
        result['before_pca'] = collinearity(before, before_names)
    drift = []
    for col in ns['CATEGORICAL_FEATURES']:
        if col in features:
            unseen = set(xte[col].dropna()) - set(xtr[col].dropna())
            drift.append({'Feature': col, 'Unseen test values': ', '.join(sorted(map(str, unseen))) or 'None',
                          'Affected test rows': int(xte[col].isin(unseen).sum())})
    result['unseen_categories'] = pd.DataFrame(drift)
    shifts = []
    for col in numeric:
        sd = xtr[col].std()
        shifts.append({'Feature': col, 'Training mean': xtr[col].mean(), 'Test mean': xte[col].mean(),
                       'Mean shift / training SD': abs(xte[col].mean() - xtr[col].mean()) / sd if sd > 0 else np.nan,
                       'Missing fraction change': xte[col].isna().mean() - xtr[col].isna().mean()})
    result['distribution_shift'] = pd.DataFrame(shifts).sort_values('Mean shift / training SD', ascending=False)
    estimator = model.steps[-1][1]
    convergence = {'Estimator': type(estimator).__name__}
    if getattr(estimator, 'n_iter_', None) is not None:
        iterations = int(np.max(np.atleast_1d(estimator.n_iter_)))
        limit = getattr(estimator, 'max_iter', None)
        convergence.update({'Iterations': iterations, 'Iteration limit': limit,
                            'Reached iteration limit': iterations >= limit if limit is not None else None})
    if hasattr(estimator, 'dual_gap_'):
        convergence['Dual gap'] = float(np.max(np.atleast_1d(estimator.dual_gap_)))
    if hasattr(estimator, 'coef_'):
        convergence['Nonzero coefficients'] = int(np.count_nonzero(estimator.coef_))
    result['estimator'] = pd.DataFrame([convergence])
    rows = []
    if classification:
        threshold = ns['selected_threshold']
        train_probability = class_probabilities(model, xtr)
        test_probability = class_probabilities(model, xte)
        class_rows = []
        for split, actual, probability in [('Training', ytr, train_probability), ('Test', yte, test_probability)]:
            predicted = predict_classes(probability, threshold)
            constant = np.repeat(int(ytr.mode().iloc[0]), len(actual))
            prevalence = ytr.value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
            baseline_probability = np.tile(prevalence, (len(actual), 1))
            scores = financial_metrics(actual, predicted, probability)
            rows.append({'Split': split, **scores,
                         'Majority baseline accuracy': accuracy_score(actual, constant),
                         'Prevalence baseline Brier': financial_metrics(actual, constant, baseline_probability)['Multiclass Brier']})
            precision, recall, f1, support = precision_recall_fscore_support(
                actual, predicted, labels=CLASSES, zero_division=0)
            for i, label in enumerate(CLASS_NAMES):
                class_rows.append({'Split': split, 'Class': label, 'Precision': precision[i],
                                   'Recall': recall[i], 'F1': f1[i], 'Support': support[i]})
        result['class_metrics'] = pd.DataFrame(class_rows)
        calibration = pd.DataFrame({'Probability': test_probability[:, 2], 'Actual': yte.eq(2).to_numpy()})
        calibration['Bin'] = pd.qcut(calibration.Probability, 5, duplicates='drop')
        result['reliability'] = calibration.groupby('Bin', observed=True).agg(
            Mean_probability=('Probability', 'mean'), Observed_hit_fraction=('Actual', 'mean'), Movies=('Actual', 'size'))
        result['class_balance'] = pd.DataFrame({'Training': ytr.value_counts(), 'Test': yte.value_counts()}).reindex(CLASSES).fillna(0).astype(int).rename(index=dict(zip(CLASSES, CLASS_NAMES)))
        result['validation'] = ns['winner'][[k for k in ['Hit_precision', 'Hit_recall', 'Hit_F1', 'Macro_F1', 'ROC_AUC', 'Score_std'] if k in ns['winner']]].to_frame('Mean CV / score spread')
    else:
        train_prediction, test_prediction = model.predict(xtr), model.predict(xte)
        residual = ytr.to_numpy() - train_prediction
        for split, actual, predicted in [('Training', ytr, train_prediction), ('Test', yte, test_prediction)]:
            assert np.isfinite(predicted).all()
            rows.append({'Split': split, 'MAE': mean_absolute_error(actual, predicted),
                         'RMSE': np.sqrt(mean_squared_error(actual, predicted)), 'R2': r2_score(actual, predicted),
                         'Mean residual (bias)': float(np.mean(actual - predicted)),
                         'Mean-rating baseline MAE': mean_absolute_error(actual, np.repeat(ytr.mean(), len(actual))),
                         'Predictions outside 0–10': int(((predicted < 0) | (predicted > 10)).sum())})
        residuals = pd.DataFrame({'Fitted': train_prediction, 'Residual': residual, 'Absolute residual': abs(residual)})
        residuals['Training fitted bin'] = pd.qcut(residuals.Fitted, 4, duplicates='drop')
        result['training_residuals'] = residuals.groupby('Training fitted bin', observed=True).agg(
            Movies=('Residual', 'size'), Mean_residual=('Residual', 'mean'),
            Residual_SD=('Residual', 'std'), Mean_absolute_residual=('Absolute residual', 'mean'))
        result['residual_patterns'] = pd.DataFrame([{
            'Corr(residual, fitted)': _safe_correlation(residual, train_prediction),
            'Corr(abs residual, fitted)': _safe_correlation(abs(residual), train_prediction)}])
        if type(estimator).__name__ == 'LinearRegression':
            centered = train_matrix - train_matrix.mean(axis=0)
            u, singular, _ = np.linalg.svd(centered, full_matrices=False)
            rank = int((singular > singular[0] * max(centered.shape) * np.finfo(float).eps).sum())
            leverage = (u[:, :rank] ** 2).sum(axis=1) + 1 / len(ytr)
            effective_p = rank + 1
            variance = np.sum(residual ** 2) / (len(ytr) - effective_p)
            cooks = residual ** 2 / (effective_p * variance) * leverage / np.maximum(1 - leverage, 1e-12) ** 2
            result['ols_influence'] = pd.DataFrame({'tmdb_id': movies.loc[ytr.index, 'tmdb_id'].to_numpy(),
                                                  'Leverage': leverage, 'Cooks_distance': cooks,
                                                  'Review_leverage': leverage > 2 * effective_p / len(ytr)})
            result['ols_influence'] = result['ols_influence'].sort_values('Cooks_distance', ascending=False).head(15)
        result['validation'] = ns['winner'][['MAE', 'RMSE', 'R2', 'Score_std']].to_frame('Mean CV / score spread')
    training_patterns = pd.util.hash_pandas_object(pd.DataFrame(train_matrix), index=False)
    test_patterns = pd.util.hash_pandas_object(pd.DataFrame(test_matrix), index=False)
    result['duplicate_patterns'] = pd.DataFrame([{
        'Duplicate transformed training rows': int(training_patterns.duplicated().sum()),
        'Test rows with an identical training predictor pattern': int(test_patterns.isin(set(training_patterns)).sum()),
    }])
    if 'overlap_review' in ns:
        reference_matrix, reference_names = _transformed(ns['overlap_reference_model'], xtr)
        before = collinearity(reference_matrix, reference_names)
        comparison = ns['overlap_review']['comparison'].copy()
        for i, metrics in enumerate([before, result['collinearity']]):
            summary = metrics['summary'].iloc[0]
            comparison.loc[comparison.index[i], 'Transformed inputs'] = int(summary['Inputs'])
            comparison.loc[comparison.index[i], 'VIF >=10'] = int(metrics['vif'].VIF.ge(10).sum())
            comparison.loc[comparison.index[i], 'Maximum VIF'] = metrics['vif'].VIF.max()
        result['overlap_comparison'] = comparison
        result['overlap_before'] = before
    result['performance'] = pd.DataFrame(rows)
    return result


def interpret_diagnostics(ns, d):
    """Plain-English findings and candidate experiments, never automatic changes."""
    rows = []
    def add(status, check, finding, meaning, action):
        rows.append({'Status': status, 'Check': check, 'Your result': finding,
                     'What it means': meaning, 'What to do next': action})
    add('PASS', 'Data safeguards', 'Chronology, IDs, exclusions and preprocessing checks passed',
        'No violation of these programmed checks was found; historical availability is still unverified.',
        'Keep the same split and training-only preprocessing.')
    c = d['collinearity']['summary'].iloc[0]
    vif = d['collinearity']['vif'].VIF
    count = int((vif >= 10).sum())
    estimator = ns['model'].steps[-1][1]
    tree = hasattr(estimator, 'feature_importances_')
    add('REVIEW' if count else 'OK', 'Overlapping inputs (VIF)',
        f'{count}/{int(c["Inputs"])} inputs have VIF >=10; {int(c["Infinite VIF"])} infinite',
        'Inputs repeat information. Coefficients may be unstable; this is not a prediction-quality score.' if not tree else
        'Trees allow correlated inputs, but importance can be shared between them.',
        ('Compare dropping n_genres while retaining genre flags using training validation.' if 'n_genres' in ns['selected_feature_columns'] else 'Genre count already removed. Remaining overlap is retained only after validation; interpret coefficients cautiously.'))
    add('REVIEW' if c['Centered rank'] < c['Nonconstant inputs'] else 'OK', 'Exact redundancy',
        f'Rank {int(c["Centered rank"])}/{int(c["Nonconstant inputs"])} nonconstant inputs',
        'Full rank means no exact dependency detected; high VIF can still occur.',
        'Review any exact dependencies; for PCA, inspect the retained components rather than the full dummy encoding.')
    if 'before_pca' in d:
        add('NOTE', 'PCA representation', f'{len(d["before_pca"]["vif"])} encoded inputs become {int(c["Inputs"])} components',
            'PCA can remove redundant directions. Low VIF does not establish better predictions.',
            'Compare component counts and neighbor settings using chronological validation.')
    missing = d['missingness']['Missing training']
    ntr = len(ns['y_train'])
    worst = missing.idxmax()
    add('REVIEW' if missing.max() else 'OK', 'Missing training information',
        f'{worst}: {int(missing.max())}/{ntr} missing ({missing.max()/ntr:.1%})' if missing.max() else 'No missing raw training inputs',
        'Imputation makes training possible; it cannot recover the unknown values.',
        'Test an explicit missing-budget flag or a model without budget inside training folds; preserve source data.')
    p = d['performance'].set_index('Split')
    if 'MAE' in p:
        train, test = p.loc['Training'], p.loc['Test']
        baseline = test['Mean-rating baseline MAE']
        add('BETTER' if test.MAE < baseline else 'REVIEW', 'Later-test rating error',
            f'MAE {test.MAE:.3f}; mean baseline {baseline:.3f}; training MAE {train.MAE:.3f}',
            f'Average absolute error is {test.MAE:.3f} rating points; improvement over baseline is {baseline-test.MAE:.3f}.',
            'Use training-validation MAE to compare candidates. This test comparison is descriptive, not a tuning target.')
        add('NOTE', 'Rating bias', f'Mean test residual {test["Mean residual (bias)"]:+.3f}; {int(test["Predictions outside 0–10"])} predictions outside 0–10',
            'Residual = actual minus predicted: positive means underprediction; negative means overprediction.',
            'Inspect training residual patterns; test interactions/penalties inside chronological validation.')
    else:
        test = p.loc['Test']
        add('NOTE', 'Three-class performance',
            f'Test accuracy {test.Accuracy:.1%}; majority baseline {test["Majority baseline accuracy"]:.1%}; macro F1 {test["Macro F1"]:.3f}',
            'Macro F1 gives Bust, Moderate and Hit equal weight. Accuracy alone can hide a weak class.',
            'Read each class recall below. Compare class weights/features within training folds and report the precision tradeoff.')
        hit = d['class_metrics'].query("Split == 'Test' and Class == 'Hit'").iloc[0]
        moderate = d['class_metrics'].query("Split == 'Test' and Class == 'Moderate success'").iloc[0]
        add('NOTE', 'Hit and moderate coverage',
            f'Hit precision {hit.Precision:.1%}, recall {hit.Recall:.1%}; moderate recall {moderate.Recall:.1%}',
            'Precision: correct predicted hits. Recall: fraction of the actual class found. A high hit threshold can miss hits.',
            'Select thresholds only on chronological validation; keep all-class macro F1 and moderate recall visible.')
    if type(estimator).__name__.startswith('KNeighbors'):
        add('NOTE', 'KNN training score', 'A training movie can be its own nearest neighbor',
            'Distance weighting can produce near-perfect in-sample results without useful generalization.',
            'Judge chronological validation and later-test scores, not the training score.')
    e = d['estimator'].iloc[0]
    reached = e.get('Reached iteration limit', False)
    add('REVIEW' if reached is True or reached == True else 'OK', 'Numerical fitting',
        f'{type(estimator).__name__}: ' + (f'{e["Iterations"]} iterations; limit {e["Iteration limit"]}' if 'Iterations' in e else 'no iterative convergence counter exposed'),
        'An iteration limit warning can mean the coefficients have not settled; absence alone does not prove optimality.',
        'If flagged, review scaling, tolerance and max_iter using training data.')
    add('LIMITATION', 'Later movies differ', 'Test drift / reliability tables are descriptive',
        'Later release years are expected. Missing budgets and retrospective histories limit generalization.',
        'Document these limitations; do not use the already-inspected test period to choose new settings.')
    return pd.DataFrame(rows)


def show_diagnostics(ns):
    """Start with explanations; opt in to the full technical tables at the top."""
    from IPython.display import display
    d = diagnose(ns)
    d['interpretation'] = interpret_diagnostics(ns, d)
    print('READ THIS FIRST: Status -> your result -> meaning -> next experiment. REVIEW is not an automatic failure.')
    with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
        display(d['interpretation'])
    if 'overlap_comparison' in d:
        print('BEFORE / AFTER OVERLAP REVIEW: validation chose the model; these VIFs describe the training matrix.')
        cols = ['Stage', 'Predictors', 'Transformed inputs', 'VIF >=10', 'Maximum VIF'] + (['Hit_precision', 'Hit_F1', 'Macro_F1'] if 'class_metrics' in d else ['MAE']) + ['Removed']
        with pd.option_context('display.max_colwidth', None): display(d['overlap_comparison'][cols].round(4))
    print('Prediction quality: lower MAE/RMSE is better; higher F1/recall/accuracy is better. Training and test scores are descriptive.')
    if 'class_metrics' in d:
        display(d['performance'][['Split', 'Accuracy', 'Balanced accuracy', 'Macro F1', 'Hit precision', 'Hit recall', 'Hit F1', 'Majority baseline accuracy']].round(3))
        display(d['class_metrics'].round(3))
    else:
        display(d['performance'].round(3))
    print('Overlapping inputs: VIF 1 = no linear overlap; 5–10 = review; >10 = high; infinity = exact dependence. Thresholds are guidelines.')
    top = d['collinearity']['vif'].head(5).copy()
    top['Feature'] = top.Feature.str.replace(r'^(numeric|binary|categorical)__', '', regex=True)
    with pd.option_context('display.max_colwidth', None):display(top)
    print('Next steps are experiments, not automatic changes. Compare them using training-only chronological validation. High VIF alone does not invalidate predictions.')
    if ns.get('SHOW_DIAGNOSTIC_DETAILS', False):
        print('OPTIONAL TECHNICAL DETAILS')
        for key in ['checks', 'folds', 'missingness', 'unseen_categories', 'distribution_shift', 'validation', 'estimator', 'duplicate_patterns', 'class_balance', 'reliability', 'training_residuals', 'residual_patterns', 'ols_influence']:
            if key in d:
                print(key.replace('_', ' ').title()); display(d[key])
        for key in ['collinearity', 'before_pca']:
            if key in d:
                print(key); display(d[key]['summary']); display(d[key]['vif']); display(d[key]['correlations'])
    else:
        print('Set SHOW_DIAGNOSTIC_DETAILS = True in the top controls for all technical tables. All tables remain in diagnostic_results.')
    return d
