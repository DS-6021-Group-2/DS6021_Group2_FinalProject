"""Read-only workflow safeguards using temporary synthetic data, not project CSVs."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modeling_scripts" / "helpers"))

from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from model_workflow import NUMERIC_FEATURES, chronological_folds, load_data, train_models


def verify():
    dates = pd.Series(pd.date_range('2000-01-01', periods=30).repeat(2))
    folds, _ = chronological_folds(dates, 3)
    for fit, val in folds:
        assert dates.iloc[fit].max() < dates.iloc[val].min()
        assert not set(dates.iloc[fit]) & set(dates.iloc[val])
    n = 80
    frame = pd.DataFrame({c:np.arange(n, dtype=float) % 7 + 1 for c in NUMERIC_FEATURES})
    frame['tmdb_id'] = np.arange(n)
    frame['imdb_id'] = [f'tt{i:07d}' for i in range(n)]
    frame['title_tmdb'] = [f'Test {i}' for i in range(n)]
    frame['release_date'] = pd.date_range('2000-01-01', periods=n)
    frame['us_certification'] = 'PG'
    frame['release_month'] = frame.release_date.dt.month
    for c in ['is_sequel','is_english','dir_first_film','dir_no_track_record',
              'backfill_unavailable','dir_no_rating_history','genre_drama']:
        frame[c] = np.arange(n) % 2
    frame['vote_average'] = 6 + .5*np.sin(np.arange(n)/4)
    with TemporaryDirectory(prefix='ds6021-workflow-test-') as directory:
        path=Path(directory)/'synthetic.csv';frame.to_csv(path,index=False)
        data=load_data(path,'rating','2000-03-01',2)
        assert set(data['FEATURE_COLUMNS']).isdisjoint(['vote_average','revenue','success','tmdb_id'])
        def make_model(columns):
            numeric=[c for c in data['NUMERIC_FEATURES'] if c in columns]
            categorical=[c for c in data['CATEGORICAL_FEATURES'] if c in columns]
            binary=[c for c in data['BINARY_FEATURES'] if c in columns]
            preprocessor=ColumnTransformer([
                ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric),
                ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical),
                ('binary','passthrough',binary),
            ])
            return Pipeline([('preprocessor',preprocessor),('regression',Ridge(alpha=1))])
        controls={'DATA_FILE':str(path),'CUTOFF_DATE':'2000-03-01','N_VALIDATION_SPLITS':2,
                  'MODEL_SETTINGS':{'alpha':1},'PARAM_GRID':{'regression__alpha':[1,10]},
                  'TUNING_METRIC':'MAE','FEATURE_GROUPS_TO_COMPARE':['All features'],
                  'OVERLAP_TRIMS':['Drop genre count'],'OVERLAP_MAE_TOLERANCE':.002,
                  'OVERLAP_HIT_F1_TOLERANCE':.005,'OVERLAP_MACRO_F1_TOLERANCE':.01}
        before=train_models(make_model,data,controls,joint_search=True)
        changed=data.copy();changed['y_test']=pd.Series(9.,index=data['y_test'].index)
        after=train_models(make_model,changed,controls,joint_search=True)
        assert before['selected_feature_columns']==after['selected_feature_columns']
        assert before['selected_parameters']==after['selected_parameters']
        np.testing.assert_allclose(before['winner'].MAE,after['winner'].MAE,atol=0,rtol=0)
        np.testing.assert_allclose(before['model'].predict(data['X_train']),after['model'].predict(data['X_train']),atol=0,rtol=0)
        # Invalid source schemas/IDs/targets stop before fitting.
        for invalid in [frame.assign(revenue=100),frame.assign(tmdb_id=0),frame.assign(vote_average=11)]:
            invalid.to_csv(path,index=False)
            try:load_data(path,'rating','2000-03-01',2)
            except AssertionError:pass
            else:raise AssertionError('Invalid input was accepted.')
    print('PASS: equal dates stay together; target/ID exclusions and data checks hold; changing only test outcomes cannot alter selection or fitted predictions.')


if __name__=='__main__':verify()
