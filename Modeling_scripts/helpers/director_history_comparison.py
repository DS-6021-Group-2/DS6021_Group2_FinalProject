"""Nested chronological ablation of ALL nine director-history predictors.

Only the original training period is evaluated; the project's later test set is
never scored or used for selection. Both arms independently retune the same grid.
"""
from pathlib import Path
import os
import json
import warnings
import numpy as np
import pandas as pd
import nbformat
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.exceptions import ConvergenceWarning
from model_workflow import chronological_folds, rating_metrics
from financial_outcomes import CLASSES, HIT_CLASS, class_probabilities, predict_classes, financial_metrics
from project_policy import (DIRECTOR_HISTORY_FEATURES, N_VALIDATION_SPLITS, SUCCESS_MIN_VALIDATION_PRECISION,
                            SUCCESS_THRESHOLDS, BASE_THRESHOLD, check_objective)
ROOT = Path(__file__).resolve().parents[2]


def load_model_spec(task, key):
    path = ROOT / f'Modeling_scripts/{task}_models/{key}.ipynb'
    notebook = nbformat.read(path, 4)
    namespace = {}
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and set(cell.metadata.get('tags', [])) & {'model-controls','model-setup','model-definition'}:
                exec(compile(cell.source, str(path), 'exec'), namespace)
    finally:
        os.chdir(previous)
    check_objective(namespace, task)
    return namespace


def history_arms(features, excluded=()):
    """Keep identical non-history inputs; restore all nine history fields in With."""
    history = set(DIRECTOR_HISTORY_FEATURES)
    assert history <= set(features)
    nonhistory = [c for c in features if c not in history and c not in excluded]
    with_history = [c for c in features if c in history or c in nonhistory]
    assert set(with_history)-set(nonhistory) == history
    return {'Without director history': nonhistory, 'With all director history': with_history}


def _tune_rating(factory, columns, x, y, splits, grid):
    search = GridSearchCV(factory(columns), grid, cv=splits, scoring={
        'MAE':'neg_mean_absolute_error','RMSE':'neg_root_mean_squared_error','R2':'r2'},
        refit=False, error_score='raise', n_jobs=1).fit(x,y)
    i = int(np.argmax(search.cv_results_['mean_test_MAE']))
    return {'parameters':search.cv_results_['params'][i], 'threshold':None,
            'inner_MAE':float(-search.cv_results_['mean_test_MAE'][i]), 'goal_met':None}


def _tune_success(factory, columns, x, y, splits, grid):
    # Include the original reference, even if a user later edits it out of the grid.
    original = factory(columns).get_params()
    keys = sorted({key for params in ParameterGrid(grid) for key in params})
    candidates, signatures = [], set()
    for params in [{}] + list(ParameterGrid(grid)):
        signature = tuple((key,repr(params.get(key,original[key]))) for key in keys)
        if signature not in signatures:
            signatures.add(signature);candidates.append(params)
    rows, failed = [], []
    for candidate_index, params in enumerate(candidates):
        candidate_rows = []
        for fold,(fit,val) in enumerate(splits,1):
            model = factory(columns).set_params(**params)
            # An unconverged candidate is excluded, not silently scored as valid.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model.fit(x.iloc[fit], y.iloc[fit])
            except ConvergenceWarning:
                failed.append(candidate_index);candidate_rows=[];break
            probability = class_probabilities(model, x.iloc[val])
            actual = y.iloc[val]
            for threshold in sorted(set(SUCCESS_THRESHOLDS + (BASE_THRESHOLD,))):
                predicted = predict_classes(probability,threshold)
                precision,recall,f1,_ = precision_recall_fscore_support(actual,predicted,labels=[HIT_CLASS],zero_division=0)
                candidate_rows.append({'Candidate':candidate_index,'Threshold':threshold,'Fold':fold,
                    'Hit_precision':precision[0],'Hit_recall':recall[0],'Hit_F1':f1[0],
                    'Macro_F1':f1_score(actual,predicted,labels=CLASSES,average='macro',zero_division=0)})
        rows.extend(candidate_rows)
    assert rows, 'All classification candidates failed to converge.'
    means=pd.DataFrame(rows).groupby(['Candidate','Threshold'],sort=False).mean(numeric_only=True).reset_index()
    means['Distance']=(means.Threshold-.5).abs()
    eligible=means.loc[means.Hit_precision.ge(SUCCESS_MIN_VALIDATION_PRECISION-1e-12)].sort_values(
        ['Hit_F1','Hit_precision','Distance'],ascending=[False,False,True],kind='stable')
    if len(eligible):winner=eligible.iloc[0]
    else:
        reference=means.loc[means.Candidate.eq(0)&means.Threshold.eq(BASE_THRESHOLD)]
        assert len(reference)==1, 'Reference did not converge; do not substitute an arbitrary fallback.'
        winner=reference.iloc[0]
    return {'parameters':candidates[int(winner.Candidate)],'threshold':float(winner.Threshold),
            'inner_hit_precision':float(winner.Hit_precision),'inner_hit_F1':float(winner.Hit_F1),
            'goal_met':bool(len(eligible)), 'unconverged_candidates_excluded':len(set(failed))}


def run_comparison():
    rows=[]
    for task,key in [('rating','lasso_model'),('success','logistic_regression')]:
        spec=load_model_spec(task,key);data=spec['data'];factory=spec['make_model']
        features=data['FEATURE_COLUMNS'];arms=history_arms(features,spec.get('EXCLUDED_PREDICTORS',[]))
        x,y=data['X_train'],data['y_train']
        dates=data['movies'].loc[y.index,'release_date']
        for arm,columns in arms.items():
            for outer_fold,(fit,val) in enumerate(data['tuning_splits'],1):
                inner, _ = chronological_folds(dates.iloc[fit],N_VALIDATION_SPLITS)
                if task=='success':
                    for inner_fit,inner_val in inner:
                        assert set(y.iloc[fit].iloc[inner_fit])==set(CLASSES)
                        assert set(y.iloc[fit].iloc[inner_val])==set(CLASSES)
                    tuned=_tune_success(factory,columns,x.iloc[fit],y.iloc[fit],inner,spec['PARAM_GRID'])
                else:tuned=_tune_rating(factory,columns,x.iloc[fit],y.iloc[fit],inner,spec['PARAM_GRID'])
                model=factory(columns).set_params(**tuned['parameters']).fit(x.iloc[fit],y.iloc[fit])
                if task=='rating':scores=rating_metrics(y.iloc[val],model.predict(x.iloc[val]))
                else:
                    probability=class_probabilities(model,x.iloc[val])
                    scores=financial_metrics(y.iloc[val],predict_classes(probability,tuned['threshold']),probability)
                rows.append({'Task':task,'Model':key,'History':arm,'Fold':outer_fold,'Predictors':len(columns),
                    'Director history inputs':len(set(columns)&set(DIRECTOR_HISTORY_FEATURES)),
                    'Training movies':len(fit),'Validation movies':len(val),
                    'Training ends':str(dates.iloc[fit].max().date()),'Validation starts':str(dates.iloc[val].min().date()),
                    'Inner precision goal met':tuned['goal_met'],
                    'Threshold':tuned['threshold'],'Parameters':json.dumps(tuned['parameters'],sort_keys=True),
                    **{k:v for k,v in tuned.items() if k not in ['parameters','threshold','goal_met']},**scores})
                print(task,arm,'outer fold',outer_fold,'completed',flush=True)
    return pd.DataFrame(rows)


def summarize(results):
    tables={}
    for task,metrics in [('rating',['MAE','RMSE','R2']),('success',['Macro F1','Moderate recall','Hit precision','Hit recall','Hit F1'])]:
        selected=results.loc[results.Task.eq(task)]
        summary=selected.groupby('History',sort=False)[metrics].mean()
        if task=='success':summary['Precision goal met (folds)']=selected.groupby('History',sort=False)['Inner precision goal met'].sum().astype(int)
        tables[task]=summary
    return tables


def save_comparison(results):
    """Persist audit details as ignored JSON; never rewrite a dataset CSV."""
    import hashlib
    from project_policy import CUTOFF_DATE
    output=ROOT/'Data_visualization/results/director_history_comparison.json'
    payload={'schema_version':1,'training_only':True,'cutoff':CUTOFF_DATE,
        'outer_folds':N_VALIDATION_SPLITS,'inner_folds':N_VALIDATION_SPLITS,
        'precision_goal':SUCCESS_MIN_VALIDATION_PRECISION,
        'director_features':list(DIRECTOR_HISTORY_FEATURES),
        'datasets_sha256':{f'data/model_data_{task}.csv':hashlib.sha256((ROOT/f'data/model_data_{task}.csv').read_bytes()).hexdigest() for task in ['rating','success']},
        'folds':json.loads(results.to_json(orient='records',double_precision=12))}
    output.parent.mkdir(exist_ok=True)
    text=json.dumps(payload,indent=2,allow_nan=False)+'\n'
    output.write_text(text)
    (ROOT/'Data_visualization/director_history_results.json').write_text(text)
    return output
