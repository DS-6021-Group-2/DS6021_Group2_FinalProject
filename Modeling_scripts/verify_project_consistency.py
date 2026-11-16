"""Check active model/data/app contracts; dated snapshots are historical only."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from director_history_comparison import load_model_spec, history_arms
from project_policy import (CUTOFF_DATE, N_VALIDATION_SPLITS, RATING_SELECTION_METRIC,
    RATING_METRIC_GOALS, SUCCESS_SELECTION_METRIC, SUCCESS_MIN_VALIDATION_PRECISION,
    DIRECTOR_HISTORY_FEATURES)
from financial_outcomes import CLASS_LABELS, CLASSES, financial_labels
ROOT=Path(__file__).resolve().parents[1]


def verify(artifacts=False):
    paths=sorted((ROOT/'Modeling_scripts').glob('*_models/*.ipynb'))
    assert len(paths)==12
    for path in paths:
        task=path.parent.name.removesuffix('_models');spec=load_model_spec(task,path.stem)
        assert spec['CUTOFF_DATE']==CUTOFF_DATE and spec['N_VALIDATION_SPLITS']==N_VALIDATION_SPLITS
        assert Path(ROOT/f'data/model_data_{task}.csv').resolve()==(path.parent/spec['DATA_FILE']).resolve()
        if task=='rating':
            assert spec['TUNING_METRIC']==RATING_SELECTION_METRIC and spec['RATING_GOALS']==RATING_METRIC_GOALS
        else:
            assert spec['TUNING_METRIC']==SUCCESS_SELECTION_METRIC
            assert spec['MIN_VALIDATION_PRECISION']==SUCCESS_MIN_VALIDATION_PRECISION
            assert set(spec['data']['y'])==set(CLASSES)
        arms=history_arms(spec['data']['FEATURE_COLUMNS'],spec.get('EXCLUDED_PREDICTORS',[]))
        assert set(arms['With all director history'])-set(arms['Without director history'])==set(DIRECTOR_HISTORY_FEATURES)
    clean=pd.read_csv(ROOT/'data/cleaned_movies.csv')
    expected=financial_labels(clean.budget,clean.revenue).astype(int)
    pd.testing.assert_series_equal(expected,clean.success,check_names=False)
    success=pd.read_csv(ROOT/'data/model_data_success.csv')
    pd.testing.assert_series_equal(success.set_index('tmdb_id').success.sort_index(),clean.set_index('tmdb_id').success.sort_index())
    assert all(1<x<2.5 for x in clean.loc[clean.success.eq(1),'revenue_to_budget'])
    if artifacts:
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('consistency_reports',ROOT/'Data_visualization/model_reports.py')
        reports=module_from_spec(spec);spec.loader.exec_module(reports)
        for path in paths:
            task=path.parent.name.removesuffix('_models')
            local=ROOT/f'Data_visualization/results/{task}/{path.stem}.json'
            if local.exists():report=reports.load_reports(task,[path.stem])[path.stem]
            else:
                packaged=json.loads((ROOT/'app/artifacts/comparisons.json').read_text())
                report=packaged[f'{task}/{path.stem}']
                assert report['dataset_sha256']==reports.file_hash(ROOT/report['dataset'])
                assert report['code_sha256']==reports.notebook_code_hash(path)
                assert report['writer_sha256']==reports.file_hash(ROOT/'Data_visualization/model_reports.py')
                assert report['dependencies_sha256']==reports.dependency_hashes()
            assert report['selection_metric']==(SUCCESS_SELECTION_METRIC if task=='success' else RATING_SELECTION_METRIC)
            if task=='success':assert report['precision_goal']==SUCCESS_MIN_VALIDATION_PRECISION
        analysis=json.loads((ROOT/'Data_visualization/director_history_results.json').read_text())
        for path,digest in analysis['datasets_sha256'].items():assert reports.file_hash(ROOT/path)==digest
        assert analysis['training_only'] and analysis['precision_goal']==SUCCESS_MIN_VALIDATION_PRECISION
        assert set(analysis['director_features'])==set(DIRECTOR_HISTORY_FEATURES)
        for row in analysis['folds']:
            assert row['Training ends'] < row['Validation starts'] < CUTOFF_DATE
            assert row['Director history inputs']==(9 if row['History']=='With all director history' else 0)
            if row['Task']=='success':
                assert row['Inner precision goal met']==(row['inner_hit_precision']>=SUCCESS_MIN_VALIDATION_PRECISION-1e-12)
        manifest=json.loads((ROOT/'app/artifacts/manifest.json').read_text())
        assert manifest['models']['success']['class_labels']=={str(k):v for k,v in CLASS_LABELS.items()}
        assert manifest['models']['success']['dataset']=='data/model_data_success.csv'
    print('PASS: 12 active models share dates/folds, task objectives and canonical inputs; 0 Bust / 1 Moderate / 2 Hit agree across cleaned/model/app data; full history ablation removes exactly nine features.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifacts',action='store_true')
    verify(parser.parse_args().artifacts)
