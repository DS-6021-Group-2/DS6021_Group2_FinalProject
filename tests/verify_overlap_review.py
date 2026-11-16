"""Meaningful guard/row-isolation tests; no project datasets are changed."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modeling_scripts"))

import numpy as np
import pandas as pd
from overlap_review import choose_candidate, review_overlap, RemoveExactRedundancy


def verify():
    rating = pd.DataFrame([
        {'Trim':'Before overlap trim','Predictors':4,'MAE':.500,'RMSE':.65},
        {'Trim':'small','Predictors':3,'MAE':.501,'RMSE':.65},
        {'Trim':'too aggressive','Predictors':2,'MAE':.510,'RMSE':.66},
    ])
    chosen,_=choose_candidate(rating,False,.7,.002,.005,.01)
    assert chosen.Trim=='small'  # A tiny declared tradeoff is allowed, .01 is not.
    rating.loc[1,'MAE']=.51
    assert choose_candidate(rating,False,.7,.002,.005,.01)[0].Trim=='Before overlap trim'
    success=pd.DataFrame([
        {'Trim':'Before overlap trim','Predictors':4,'Hit_F1':.40,'Macro_F1':.35,'Hit_precision':.71,'Threshold distance':.1},
        {'Trim':'precision fails','Predictors':1,'Hit_F1':.45,'Macro_F1':.36,'Hit_precision':.69,'Threshold distance':.1},
        {'Trim':'macro fails','Predictors':1,'Hit_F1':.45,'Macro_F1':.30,'Hit_precision':.72,'Threshold distance':.1},
        {'Trim':'safe trim','Predictors':3,'Hit_F1':.398,'Macro_F1':.349,'Hit_precision':.72,'Threshold distance':.1},
    ])
    chosen,_=choose_candidate(success,True,.7,.002,.005,.01)
    assert chosen.Trim=='safe trim'
    success.loc[:,'Hit_precision']=.60
    assert choose_candidate(success,True,.7,.002,.005,.01)[0].Trim=='Before overlap trim'
    # Recording estimator proves every fit gets only the designated earlier rows.
    class RecordingEstimator:
        fit_rows=[]
        def set_params(self,**params):return self
        def fit(self,x,y):
            self.fit_rows.append(tuple(x.index));self.mean=float(y.mean());return self
        def predict(self,x):return np.repeat(self.mean,len(x))
    x=pd.DataFrame({'n_genres':[1]*12,'runtime':np.arange(12)},index=np.arange(100,112))
    y=pd.Series(np.arange(12),index=x.index)
    splits=[(np.arange(4),np.arange(4,8)),(np.arange(8),np.arange(8,12))]
    scores={'Feature set':'reference','MAE':10.,'RMSE':10.,'R2':0.,'Score_std':0.}
    result=review_overlap(lambda columns:RecordingEstimator(),x,y,splits,list(x),{},scores,{'initial':{}},presets=['Drop genre count'])
    assert RecordingEstimator.fit_rows==[tuple(x.index[:4]),tuple(x.index[:8])]
    assert result['columns']==['runtime'] and 'n_genres' in x  # No data mutation.
    # Squared binary flags and constant encodings have exact aliases with an intercept.
    flag=np.tile([0.,1.],30);z=np.column_stack([np.arange(60),flag,flag**2,np.ones(60)])
    reducer=RemoveExactRedundancy().fit(z)
    reduced=reducer.transform(z)
    assert reduced.shape==(60,2)
    from model_diagnostics import collinearity
    assert collinearity(reduced,reducer.get_feature_names_out())['summary'].iloc[0]['Infinite VIF']==0
    # Validation input never refits or changes the learned column selection.
    before=reducer.keep_indices_.copy();reducer.transform(z[:2]);np.testing.assert_array_equal(before,reducer.keep_indices_)
    print('PASS: MAE tolerance, hit precision/macro F1 guards, reference fallback, fold isolation, and unchanged inputs.')


if __name__=='__main__':verify()
