"""Offline export only: reproduce existing notebook selection and verify test outputs.

AI-assisted. Never called by the web app. Requires existing verified local reports.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform

import joblib
import nbformat
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'app/artifacts'


def export():
    spec = importlib.util.spec_from_file_location('export_reports', ROOT / 'Data_visualization/model_reports.py')
    reports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reports)
    before = {p: reports.file_hash(p) for p in (ROOT / 'data').glob('*.csv')}
    manifest = {'sklearn_version': sklearn.__version__, 'python_version': platform.python_version(), 'models': {}}
    ARTIFACTS.mkdir(exist_ok=True)
    original_cwd = Path.cwd()
    for task, key in [('rating', 'ridge_model'), ('success', 'logistic_regression')]:
        reference = reports.load_reports(task, [key])[key]
        notebook = ROOT / reference['notebook']
        nb = nbformat.read(notebook, as_version=4)
        ns = {}
        try:
            os.chdir(notebook.parent)
            # Run the existing training/selection code, never refit on the test set.
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in nb.cells[:14]:
                    if cell.cell_type == 'code':
                        exec(compile(cell.source, str(notebook), 'exec'), ns)
        finally:
            os.chdir(original_cwd)
        model = ns['model']
        selected = reference['versions']['Selected']
        ids = ns['movies'].loc[ns['y_test'].index, 'tmdb_id'].tolist()
        expected = pd.DataFrame(selected['predictions']).set_index('tmdb_id').loc[ids]
        np.testing.assert_allclose(ns['y_test'], expected.actual, atol=1e-10, rtol=0)
        assert list(ns['selected_feature_columns']) == selected['features']
        if task == 'rating':
            actual = model.predict(ns['X_test'])
            np.testing.assert_allclose(actual, expected.predicted, atol=1e-9, rtol=0)
        else:
            actual = model.predict_proba(ns['X_test'])[:, list(model.classes_).index(1)]
            np.testing.assert_allclose(actual, expected.hit_probability, atol=1e-9, rtol=0)
            assert ns['selected_threshold'] == selected['threshold']
            np.testing.assert_array_equal((actual >= selected['threshold']).astype(int), expected.predicted)
        file = ARTIFACTS / f'{task}_pipeline.joblib'
        joblib.dump(model, file, compress=3)
        manifest['models'][task] = {
            'artifact': file.name, 'artifact_sha256': reports.file_hash(file),
            'dataset': reference['dataset'], 'dataset_sha256': reference['dataset_sha256'],
            'notebook': reference['notebook'], 'code_sha256': reference['code_sha256'],
            'cutoff': reference['cutoff'], 'input_columns': ns['FEATURE_COLUMNS'],
            'selected_features': ns['selected_feature_columns'],
            'categorical_columns': ns['CATEGORICAL_FEATURES'], 'threshold': selected['threshold'],
            'selected_parameters': ns['selected_parameters'],
            'train_count': len(ns['y_train']), 'test_count': len(ns['y_test']),
            'test_metrics': selected['test_metrics'],
        }
        print(f'{task}: exported pipeline; all {len(ids):,} test predictions match notebook.', flush=True)
    # Compact, portable metrics: no duplicate prediction arrays or fitted models.
    packaged = {}
    for task in ['rating', 'success']:
        for notebook in sorted((ROOT / f'Modeling_scripts/{task}_models').glob('*.ipynb')):
            key = notebook.stem
            report = reports.load_reports(task, [key])[key]
            report['cohort_id'] = hashlib.sha256(json.dumps([
                report['dataset_sha256'], report['cutoff'], report['train_count'],
                sorted((p['tmdb_id'], p['actual'], p['release_date']) for p in report['versions']['Selected']['predictions'])
            ]).encode()).hexdigest()[:12]
            for version in report['versions'].values():
                version.pop('predictions', None)
            report.pop('feature_fold_results', None)
            packaged[f'{task}/{key}'] = report
    (ARTIFACTS / 'comparisons.json').write_text(json.dumps(packaged, indent=2) + '\n')
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(reports.file_hash(p) == h for p, h in before.items())
    print(f'{len(packaged)} comparison reports packaged. All CSVs unchanged.', flush=True)


if __name__ == '__main__':
    export()
