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
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / 'app/artifacts'


def export(rating_only=False):
    spec = importlib.util.spec_from_file_location('export_reports', ROOT / 'Data_visualization/model_reports.py')
    reports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reports)
    from app.model_utils import class_probabilities, predict_classes
    before = {p: reports.file_hash(p) for p in (ROOT / 'data').glob('*.csv')}
    if rating_only:
        # Verify and retain the existing success pipeline without fitting it again.
        from app.model_utils import load_prediction_models
        load_prediction_models()
        manifest = json.loads((ARTIFACTS / 'manifest.json').read_text())
    else:
        manifest = {'sklearn_version': sklearn.__version__, 'python_version': platform.python_version(), 'models': {}}
    ARTIFACTS.mkdir(exist_ok=True)
    original_cwd = Path.cwd()
    selections = [('rating', 'lasso_model')]
    if not rating_only:
        selections.append(('success', 'logistic_regression'))
    for task, key in selections:
        reference = reports.load_reports(task, [key])[key]
        notebook = ROOT / reference['notebook']
        nb = nbformat.read(notebook, as_version=4)
        ns = {}
        try:
            os.chdir(notebook.parent)
            # Run the existing training/selection code, never refit on the test set.
            with contextlib.redirect_stdout(io.StringIO()):
                trained = False
                for cell in nb.cells:
                    if cell.cell_type == 'code':
                        exec(compile(cell.source, str(notebook), 'exec'), ns)
                    if 'model-training' in cell.metadata.get('tags', []):
                        trained = True
                        break
                assert trained, 'Notebook is missing the tagged model-training cell.'
        finally:
            os.chdir(original_cwd)
        ns = ns['run']
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
            probabilities = class_probabilities(model, ns['X_test'])
            actual = probabilities[:, 2]
            np.testing.assert_allclose(probabilities, expected[['bust_probability', 'moderate_probability', 'hit_probability']], atol=1e-9, rtol=0)
            np.testing.assert_allclose(actual, expected.hit_probability, atol=1e-9, rtol=0)
            assert ns['selected_threshold'] == selected['threshold']
            np.testing.assert_array_equal(predict_classes(probabilities, selected['threshold']), expected.predicted)
        file = ARTIFACTS / f'{task}_pipeline.joblib'
        joblib.dump(model, file, compress=3)
        manifest['models'][task] = {
            'model_key': key,
            'outcome_schema': reference['outcome_schema'],
            'dependencies_sha256': reference['dependencies_sha256'],
            'class_labels': {'0': 'Bust', '1': 'Moderate success', '2': 'Hit'} if task == 'success' else None,
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
    package_comparisons(reports)
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(reports.file_hash(p) == h for p, h in before.items())


def package_comparisons(reports=None):
    """Refresh verified metrics only; leave fitted prediction pipelines unchanged."""
    if reports is None:
        spec = importlib.util.spec_from_file_location('export_reports', ROOT / 'Data_visualization/model_reports.py')
        reports = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reports)
    before = {p: reports.file_hash(p) for p in (ROOT / 'data').glob('*.csv')}
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
            report.pop('overlap_fold_results', None)
            packaged[f'{task}/{key}'] = report
    (ARTIFACTS / 'comparisons.json').write_text(json.dumps(packaged, indent=2) + '\n')
    assert all(reports.file_hash(p) == h for p, h in before.items())
    print(f'{len(packaged)} comparison reports packaged. All CSVs unchanged.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--rating-only', action='store_true', help='Keep the verified existing success pipeline unchanged.')
    mode.add_argument('--comparisons-only', action='store_true', help='Package current comparison reports without fitting or rewriting prediction pipelines.')
    args = parser.parse_args()
    if args.comparisons_only:
        package_comparisons()
    else:
        export(rating_only=args.rating_only)
