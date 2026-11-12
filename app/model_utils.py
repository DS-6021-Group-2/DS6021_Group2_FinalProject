"""Read-only app data access. AI-assisted; verified as described in README.md.

This stage never trains, loads fitted estimators, or generates predictions.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'simple_model': 'Simple model', 'linear_model': 'Linear regression',
    'ridge_model': 'Ridge regression', 'support_vector': 'Support Vector Regression',
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
                report = report_tools.load_reports(task, [key], 'Selected')[key]
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
