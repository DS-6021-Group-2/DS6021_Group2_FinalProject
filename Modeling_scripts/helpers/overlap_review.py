"""Bounded, training-validation-only overlap reduction (AI-assisted).

Fixed domain-based exclusions: no correlations, VIF, targets, or test rows are
used to learn a deletion list. Every candidate is refitted within the same folds.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.linalg import qr
from financial_outcomes import class_probabilities, predict_classes, financial_metrics


class RemoveExactRedundancy(TransformerMixin, BaseEstimator):
    """Keep independent nonconstant training directions, including an intercept.

    QR is learned inside each fold. Scaling is only for rank detection; retained
    inputs keep their original values. This is not an iterative high-VIF filter.
    """
    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        assert np.isfinite(values).all()
        self.n_features_in_ = values.shape[1]
        centered = values - values.mean(axis=0)
        sd = centered.std(axis=0, ddof=1)
        active = np.flatnonzero(sd > 100*np.finfo(float).eps*np.maximum(1,np.abs(values).max(axis=0)))
        assert len(active), 'No nonconstant inputs remain.'
        z = centered[:, active] / sd[active]
        _, triangular, pivots = qr(z, mode='economic', pivoting=True)
        tolerance = max(z.shape)*np.finfo(float).eps*abs(triangular[0,0])
        rank = int((abs(np.diag(triangular))>tolerance).sum())
        self.keep_indices_ = np.sort(active[pivots[:rank]])
        return self

    def transform(self, x):
        values = np.asarray(x)
        assert values.shape[1] == self.n_features_in_
        return values[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else
                           [f'x{i}' for i in range(self.n_features_in_)], dtype=object)
        assert len(names)==self.n_features_in_
        return names[self.keep_indices_]


def prepare_trimmed_model(model):
    """Remove exact aliases caused by squaring imputer flags in polynomial trials."""
    if not hasattr(model, 'named_steps') or 'preprocessor' not in model.named_steps:
        return model
    preprocessor = model.named_steps['preprocessor']
    numeric = next((transformer for name,transformer,_ in preprocessor.transformers if name=='numeric'), None)
    if numeric is None or not hasattr(numeric, 'named_steps'):
        return model
    steps = numeric.named_steps
    if ('polynomial' in steps and steps['polynomial'].degree>1
            and steps['imputer'].add_indicator):
        return Pipeline(model.steps[:-1]+[('deduplicate',RemoveExactRedundancy()),model.steps[-1]])
    return model

TRIM_PRESETS = {
    'Drop genre count': ['n_genres'],
    'Lean history': ['n_genres', 'dir_years_active_log1p', 'dir_prior_rated_films_log1p'],
    'Lean history + director count': ['n_genres', 'dir_years_active_log1p',
                                      'dir_prior_rated_films_log1p',
                                      'dir_prior_known_films_log1p', 'imdb_director_count'],
}


def choose_candidate(table, classification, precision_floor, mae_tolerance,
                     hit_f1_tolerance, macro_f1_tolerance):
    """Prefer fewer inputs within explicit validation tolerances; retain reference."""
    baseline = table.loc[table.Trim.eq('Before overlap trim')].iloc[0]
    if classification:
        eligible = table.loc[table.Hit_precision.ge(precision_floor - 1e-12)
                             & table.Macro_F1.ge(baseline.Macro_F1 - macro_f1_tolerance - 1e-12)]
        if eligible.empty:
            return baseline.copy(), 'Original reference retained: precision/macro F1 guard infeasible.'
        best = eligible.Hit_F1.max()
        eligible = eligible.loc[eligible.Hit_F1.ge(best-hit_f1_tolerance-1e-12)]
        chosen = eligible.sort_values(['Predictors', 'Hit_F1', 'Macro_F1', 'Hit_precision', 'Threshold distance'],
            ascending=[True, False, False, False, True], kind='stable').iloc[0]
    else:
        best = table.MAE.min()
        eligible = table.loc[table.MAE.le(best+mae_tolerance+1e-12)]
        chosen = eligible.sort_values(['Predictors', 'MAE', 'RMSE'], kind='stable').iloc[0]
    status = ('Retained reference: trims did not justify a change.' if chosen.Trim == 'Before overlap trim'
              else 'Trim selected using chronological training validation only.')
    return chosen.copy(), status


def review_overlap(factory, x_train, y_train, splits, columns, parameters,
                   reference_scores, configurations, classification=False,
                   thresholds=None, reference_threshold=None, precision_floor=0.70,
                   presets=None, mae_tolerance=0.002, hit_f1_tolerance=0.005,
                   macro_f1_tolerance=0.01):
    """Only training data enter this API. Returns selection metadata, not a model."""
    assert len(splits)>=2 and len(x_train)==len(y_train)
    assert mae_tolerance>=0 and hit_f1_tolerance>=0 and macro_f1_tolerance>=0
    presets = list(TRIM_PRESETS) if presets is None else presets
    assert set(presets)<=set(TRIM_PRESETS)
    base = dict(reference_scores)
    base.update({'Trim':'Before overlap trim', 'Predictors':len(columns),
                 'Removed':'None', 'Settings':'Reference', 'Threshold':reference_threshold})
    base['Threshold distance'] = abs(reference_threshold-.5) if classification else 0
    rows, fold_rows = [base], []
    candidate_columns = {'Before overlap trim': list(columns)}
    parameter_map = {'Reference':dict(parameters)}
    seen = {tuple(columns)}
    for trim in presets:
        retained = [c for c in columns if c not in TRIM_PRESETS[trim]]
        if tuple(retained) in seen:continue
        seen.add(tuple(retained)); candidate_columns[trim]=retained
        removed = ', '.join(c for c in columns if c not in retained)
        for label, config in configurations.items():
            parameter_map[label] = dict(config)
            for fold,(fit,val) in enumerate(splits,1):
                assert set(fit).isdisjoint(val)
                candidate = prepare_trimmed_model(factory(retained).set_params(**config))
                candidate.fit(x_train.iloc[fit], y_train.iloc[fit])
                actual=y_train.iloc[val]
                if classification:
                    probability = class_probabilities(candidate, x_train.iloc[val])
                    for threshold in thresholds:
                        scores=financial_metrics(actual,predict_classes(probability,threshold),probability)
                        fold_rows.append({'Trim':trim,'Settings':label,'Threshold':threshold,'Fold':fold,
                            **{key.replace(' ','_'):value for key,value in scores.items()}})
                else:
                    prediction=candidate.predict(x_train.iloc[val])
                    fold_rows.append({'Trim':trim,'Settings':label,'Fold':fold,
                        'MAE':mean_absolute_error(actual,prediction),
                        'RMSE':np.sqrt(mean_squared_error(actual,prediction)), 'R2':r2_score(actual,prediction)})
    folds=pd.DataFrame(fold_rows)
    metrics=['Hit_F1','Hit_precision','Hit_recall','Macro_F1','Balanced_accuracy','Accuracy','ROC_AUC'] if classification else ['MAE','RMSE','R2']
    keys=['Trim','Settings']+(['Threshold'] if classification else [])
    if len(folds):
        assert folds.groupby(keys).Fold.nunique().eq(len(splits)).all()
        summary=folds.groupby(keys,sort=False).agg(**{m:(m,'mean') for m in metrics},
            Score_std=('Hit_F1' if classification else 'MAE','std')).reset_index()
        for _,row in summary.iterrows():
            row=row.to_dict();trim=row['Trim']
            row.update({'Predictors':len(candidate_columns[trim]),
                'Removed':', '.join(c for c in columns if c not in candidate_columns[trim]),
                'Threshold distance':abs(row['Threshold']-.5) if classification else 0})
            rows.append(row)
    table=pd.DataFrame(rows)
    chosen,status=choose_candidate(table,classification,precision_floor,mae_tolerance,hit_f1_tolerance,macro_f1_tolerance)
    selected_columns=candidate_columns[chosen.Trim]
    chosen['Feature set']=str(reference_scores['Feature set'])+' | '+chosen.Trim
    comparison=pd.DataFrame([base,chosen.to_dict()])
    comparison['Stage']=['Before overlap trim','After overlap review']
    return {'columns':selected_columns,'parameters':parameter_map[chosen.Settings],
        'threshold':float(chosen.Threshold) if classification else None,'winner':chosen,
        'comparison':comparison,'candidates':table,'folds':folds,'status':status,
        'rules':{'mae_tolerance':mae_tolerance,'hit_f1_tolerance':hit_f1_tolerance,
                 'macro_f1_tolerance':macro_f1_tolerance,'precision_floor':precision_floor if classification else None}}
