"""Read-only consistency and UI checks. AI-assisted; run with ds6001 from repo root."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit.testing.v1 import AppTest

from model_utils import class_probabilities, predict_classes, CLASS_LABELS, ROOT, load_movie_catalog, load_comparisons, load_prediction_models


def verify():
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'data').glob('*.csv')}
    inventory = {str(p.relative_to(ROOT)) for p in (ROOT / 'Modeling_scripts').glob('*/*.ipynb')}
    scores, statuses, _ = load_comparisons()
    assert set(statuses.Notebook) == inventory
    assert statuses.Status.eq('Verified saved results').all(), statuses.to_dict('records')
    packaged = json.loads((ROOT / 'app/artifacts/comparisons.json').read_text())
    expected_keys = {f"{Path(path).parent.name.removesuffix('_models')}/{Path(path).stem}" for path in inventory}
    assert set(packaged) == expected_keys, 'Packaged inventory differs from notebook inventory.'
    for key, report in packaged.items():
        task, _ = key.split('/')
        local_file = ROOT / f'Data_visualization/results/{key}.json'
        if local_file.exists():
            current = json.loads(local_file.read_text())
            assert report['code_sha256'] == current['code_sha256']
            assert report['dataset_sha256'] == current['dataset_sha256']
            assert set(report['versions']) == set(current['versions'])
            for version in report['versions']:
                assert report['versions'][version]['test_metrics'] == current['versions'][version]['test_metrics']
                assert report['versions'][version]['features'] == current['versions'][version]['features']
                assert report['versions'][version]['threshold'] == current['versions'][version]['threshold']
    print(f'PASS: all {len(inventory)} notebooks match the comparison inventory and packaged scores.')

    models = load_prediction_models()
    assert type(models['rating'][0].named_steps['regression']).__name__ == 'Lasso'
    assert Path(models['rating'][1]['notebook']).stem == 'lasso_model'
    assert type(models['success'][0].named_steps['classifier']).__name__ == 'LogisticRegression'
    assert list(models['success'][0].classes_) == [0, 1, 2]
    assert models['success'][1]['outcome_schema'] == 'bust-moderate-hit-v1'
    for task, (pipeline, meta, frame) in models.items():
        rows = frame.loc[frame.release_date.ge(meta['cutoff'])]
        assert len(rows) == meta['test_count']
        assert frame.release_date.lt(meta['cutoff']).sum() == meta['train_count']
        inputs = rows[meta['input_columns']].copy()
        for column in meta['categorical_columns']:
            inputs[column] = inputs[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
        key = Path(meta['notebook']).stem
        # Full local reports supply per-movie reference predictions; packaged reports supply metrics.
        reference_file = ROOT / f'Data_visualization/results/{task}/{key}.json'
        if task == 'rating':
            predicted = pipeline.predict(inputs)
            actual = rows.vote_average
            metrics = {'MAE': mean_absolute_error(actual, predicted),
                       'RMSE': np.sqrt(mean_squared_error(actual, predicted)), 'R2': r2_score(actual, predicted)}
            for name, value in metrics.items():
                np.testing.assert_allclose(value, meta['test_metrics'][name], rtol=0, atol=1e-10)
        else:
            probabilities = class_probabilities(pipeline, inputs)
            predicted = probabilities[:, 2]
        if reference_file.exists():
            reference = pd.DataFrame(json.loads(reference_file.read_text())['versions']['Selected']['predictions']).set_index('tmdb_id').loc[rows.tmdb_id]
            target = reference.predicted if task == 'rating' else reference.hit_probability
            np.testing.assert_allclose(predicted, target, rtol=0, atol=1e-9)
            if task == 'success':
                np.testing.assert_allclose(probabilities, reference[['bust_probability', 'moderate_probability', 'hit_probability']], rtol=0, atol=1e-9)
                np.testing.assert_array_equal(predict_classes(probabilities, meta['threshold']), reference.predicted)
        print(f'PASS: {task} pipeline matches recorded outputs on {len(rows):,} test movies.')

    catalog = load_movie_catalog()
    at = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=45).run()
    assert not at.exception, at.exception
    assert any(m.label == 'Models in comparison' and m.value == str(len(inventory)) for m in at.metric)
    assert any('Lasso regression' in item.value for item in at.markdown)
    known = catalog.loc[catalog.success.notna()].iloc[0]
    at.selectbox(key='movie').set_value(int(known.tmdb_id)).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any(m.label == 'Predicted viewer rating' and m.value != '—' for m in at.metric)
    assert any(m.label == 'Estimated hit probability' and m.value != '—' for m in at.metric)
    alpha = models['rating'][1]['selected_parameters'].get('regression__alpha', models['rating'][0].named_steps['regression'].alpha)
    assert any(f'Alpha {alpha:g}' in item.value and 'Lasso' in item.value for item in at.caption)
    at.toggle(key='show_actual').set_value(True).run()
    # Every actual financial class must be displayed with its correct meaning.
    for code, label in CLASS_LABELS.items():
        example = catalog.loc[catalog.success.eq(code)].iloc[0]
        at.selectbox(key='movie').set_value(int(example.tmdb_id)).run()
        assert any(m.label == 'Actual financial outcome' and m.value == label for m in at.metric)
        at.button[0].click().run()
        assert not at.exception, at.exception
        from model_utils import predict_movie
        result = predict_movie(int(example.tmdb_id), models)['success']
        np.testing.assert_allclose(sum(result['probabilities'].values()), 1, atol=1e-8)
        assert result['label'] in CLASS_LABELS.values()
        assert any('Predicted outcome: ' + result['label'] in item.value for item in at.markdown)
    missing = catalog.loc[catalog.success.isna()].iloc[0]
    at.selectbox(key='movie').set_value(int(missing.tmdb_id)).run()
    assert any(m.label == 'Predicted viewer rating' and m.value == '—' for m in at.metric)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any(m.label == 'Predicted viewer rating' and m.value != '—' for m in at.metric)
    assert any(m.label == 'Estimated hit probability' and m.value == '—' for m in at.metric)
    assert any(m.label == 'Actual financial outcome' and m.value == 'Unavailable' for m in at.metric)
    training_id = int(models['rating'][2].loc[lambda d: d.release_date.lt('2017-11-18'), 'tmdb_id'].iloc[0])
    from model_utils import predict_movie
    try:
        predict_movie(training_id, models)
    except ValueError:
        pass
    else:
        raise AssertionError('Training movie must not enter the demo.')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
    print('PASS: real UI predictions, all three financial labels/probabilities, lasso label, state reset, missing-finance handling, training exclusion, and unchanged CSVs.')


if __name__ == '__main__':
    verify()
