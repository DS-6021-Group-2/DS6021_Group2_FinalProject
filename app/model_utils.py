"""Read-only app data access. AI-assisted; verified as described in README.md.

The app loads validated fitted estimators; training happens only in the offline exporter.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'linear_model': 'Linear regression',
    'lasso_model': 'Lasso regression', 'elastic_net_model': 'Elastic net regression',
    'ridge_model': 'Ridge regression',
    'random_forest': 'Random Forest', 'gradient_boosting': 'Gradient Boosting',
    'knn_pca': 'KNN + PCA', 'logistic_regression': 'Logistic regression',
}
VARIANTS = {'Original': 'Original', 'Settings only': 'Settings tuned',
            'Selected': 'Validation-selected'}


def load_movie_catalog():
    """Use existing canonical tables; missing financial labels remain unavailable."""
    rating = pd.read_csv(ROOT / 'data/model_data_rating.csv', parse_dates=['release_date'],
                         usecols=['tmdb_id', 'title_tmdb', 'release_date', 'runtime', 'vote_average'])
    success = pd.read_csv(ROOT / 'data/model_data_success.csv', usecols=['tmdb_id', 'success'])
    assert rating.tmdb_id.is_unique and success.tmdb_id.is_unique
    assert success.success.isin([0, 1]).all()
    catalog = rating.merge(success, on='tmdb_id', how='left', validate='one_to_one')
    manifest = json.loads((ROOT / 'app/artifacts/manifest.json').read_text())
    cutoff = manifest['models']['rating']['cutoff']
    catalog = catalog.loc[catalog.release_date.ge(pd.Timestamp(cutoff))].copy()
    catalog['label'] = (catalog.title_tmdb.fillna('Untitled') + ' ('
                        + catalog.release_date.dt.year.astype(str) + ') · TMDB '
                        + catalog.tmdb_id.astype(str))
    return catalog.sort_values(['title_tmdb', 'tmdb_id'], kind='stable').reset_index(drop=True)


def load_comparisons():
    """Discover every model; exclude missing/stale results instead of inventing scores.

    Reuse project report checks, including dataset, code, and report-writer hashes.
    Cohort IDs distinguish test periods/populations that must not share a chart.
    """
    spec = importlib.util.spec_from_file_location('app_project_reports', ROOT / 'Data_visualization/model_reports.py')
    report_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_tools)
    rows, statuses, cohorts = [], [], {}
    for task in ['rating', 'success']:
        for notebook in sorted((ROOT / f'Modeling_scripts/{task}_models').glob('*.ipynb')):
            key = notebook.stem
            label = LABELS.get(key, key.replace('_', ' ').title())
            try:
                local_report = ROOT / f'Data_visualization/results/{task}/{key}.json'
                if local_report.exists():
                    report = report_tools.load_reports(task, [key], 'Selected')[key]
                else:
                    packaged = json.loads((ROOT / 'app/artifacts/comparisons.json').read_text())
                    report = packaged[f'{task}/{key}']
                    for path, expected in [(ROOT / report['dataset'], report['dataset_sha256']), (ROOT / 'Data_visualization/model_reports.py', report['writer_sha256'])]:
                        if report_tools.file_hash(path) != expected:
                            raise ValueError('Packaged comparison is stale.')
                    if report_tools.notebook_code_hash(notebook) != report['code_sha256']:
                        raise ValueError('Packaged notebook comparison is stale.')
                if 'cohort_id' in report:
                    cohort = report['cohort_id']
                else:
                    signature = [report['dataset_sha256'], report['cutoff'], report['train_count'],
                                 sorted((p['tmdb_id'], p['actual'], p['release_date'])
                                        for p in report['versions']['Selected']['predictions'])]
                    cohort = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:12]
                cohorts[(task, cohort)] = {
                    'Dataset': report['dataset'], 'Training movies': report['train_count'],
                    'Test movies': report['test_count'], 'Test starts': report['cutoff'],
                }
                for variant, result in report['versions'].items():
                    if variant not in VARIANTS:
                        continue
                    rows.append({'Task': task, 'Model': label, 'Version': VARIANTS[variant],
                                 'Cohort': cohort, 'Threshold': result['threshold'],
                                 **result['test_metrics']})
                status = 'Verified saved results'
            except FileNotFoundError:
                status = 'Results unavailable — no saved report'
            except (ValueError, AssertionError, KeyError, TypeError, json.JSONDecodeError) as exc:
                status = f'Results unavailable — report is stale or invalid ({type(exc).__name__})'
            statuses.append({'Task': task, 'Model': label, 'Status': status,
                             'Notebook': str(notebook.relative_to(ROOT))})
    return pd.DataFrame(rows), pd.DataFrame(statuses), cohorts


def load_prediction_models():
    """Load only our packaged estimators after validating provenance and versions."""
    import joblib
    import sklearn
    import nbformat
    artifacts = ROOT / 'app/artifacts'
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    if manifest['sklearn_version'] != sklearn.__version__:
        raise ValueError('Installed scikit-learn differs from the exported models. Use requirements.txt.')
    models = {}
    for task, meta in manifest['models'].items():
        path = artifacts / meta['artifact']
        if path.parent != artifacts or hashlib.sha256(path.read_bytes()).hexdigest() != meta['artifact_sha256']:
            raise ValueError(f'{task} model artifact does not match its manifest.')
        if hashlib.sha256((ROOT / meta['dataset']).read_bytes()).hexdigest() != meta['dataset_sha256']:
            raise ValueError(f'{task} dataset changed since model export.')
        nb = nbformat.read(ROOT / meta['notebook'], as_version=4)
        code = [c.source for c in nb.cells if c.cell_type == 'code']
        if hashlib.sha256(json.dumps(code, ensure_ascii=False).encode()).hexdigest() != meta['code_sha256']:
            raise ValueError(f'{task} notebook changed since model export.')
        assert not set(meta['input_columns']) & {'vote_average', 'success', 'revenue', 'revenue_to_budget', 'tmdb_id', 'imdb_id', 'title_tmdb', 'release_date'}
        frame = pd.read_csv(ROOT / meta['dataset'], parse_dates=['release_date'])
        assert frame.tmdb_id.is_unique
        # Joblib files are trusted project artifacts, never user uploads.
        models[task] = (joblib.load(path), meta, frame)
    return models


def predict_movie(movie_id, models):
    """Predict only eligible held-out movies; never fill in unknown financial labels."""
    result = {}
    for task, (pipeline, meta, frame) in models.items():
        rows = frame.loc[frame.tmdb_id.eq(movie_id)]
        if rows.empty:
            result[task] = {'unavailable': 'Movie is outside the financially eligible cohort.'}
            continue
        if not rows.release_date.ge(pd.Timestamp(meta['cutoff'])).all():
            raise ValueError('The prediction demo is limited to later-test movies.')
        inputs = rows[meta['input_columns']].copy()
        for column in meta['categorical_columns']:
            inputs[column] = inputs[column].map(lambda value: str(value) if pd.notna(value) else float('nan'))
        if task == 'rating':
            result[task] = {'rating': float(pipeline.predict(inputs)[0])}
        else:
            probability = float(pipeline.predict_proba(inputs)[0, list(pipeline.classes_).index(1)])
            result[task] = {'probability': probability, 'label': 'Hit' if probability >= meta['threshold'] else 'Bust', 'threshold': meta['threshold']}
    return result
