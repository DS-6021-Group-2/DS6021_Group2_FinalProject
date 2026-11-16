"""Read-only boundary, decision-rule, metric and dataset consistency checks."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modeling_scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from financial_outcomes import financial_labels, predict_classes, financial_metrics


def verify():
    # Both boundaries and immediately adjacent values; unknown != Bust.
    labels = financial_labels([100]*6 + [0, 100, -1, np.nan],
                             [99, 100, 100.001, 249.999, 250, 251, 50, 0, 50, 100])
    assert labels.iloc[:6].tolist() == [0, 0, 1, 1, 2, 2]
    assert labels.iloc[6:].isna().all()
    p = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .2, .7], [.2, .35, .45]])
    np.testing.assert_array_equal(predict_classes(p, .5), [0, 1, 2, 1])
    np.testing.assert_array_equal(predict_classes(p, None), [0, 1, 2, 2])
    np.testing.assert_array_equal(predict_classes(p, .4), [0, 1, 2, 2])
    actual = np.array([0, 1, 2, 2]);predicted = predict_classes(p, .5)
    scores = financial_metrics(actual, predicted, p)
    np.testing.assert_allclose(scores['Hit precision'], precision_score(actual==2, predicted==2))
    np.testing.assert_allclose(scores['Macro F1'], f1_score(actual, predicted, average='macro'))
    np.testing.assert_allclose(scores['ROC AUC'], roc_auc_score(actual==2, p[:, 2]))
    # Class codes are NOT averaged; perfect one-hot probabilities have Brier zero.
    perfect = financial_metrics([0, 1, 2], [0, 1, 2], np.eye(3))
    assert perfect['Macro F1'] == 1 and perfect['Multiclass Brier'] == 0
    root = Path(__file__).resolve().parents[1]
    cleaned = pd.read_csv(root/'data/cleaned_movies.csv').set_index('tmdb_id')
    success = pd.read_csv(root/'data/model_data_success.csv').set_index('tmdb_id')
    rating = pd.read_csv(root/'data/model_data_rating.csv').set_index('tmdb_id')
    assert cleaned.index.is_unique and success.index.is_unique and rating.index.is_unique
    expected = financial_labels(cleaned.budget, cleaned.revenue)
    pd.testing.assert_series_equal(cleaned.success.astype('Int64'), expected, check_names=False)
    pd.testing.assert_series_equal(success.success.sort_index(), cleaned.success.sort_index())
    assert set(success.success)=={0,1,2} and 'success' not in rating
    pd.testing.assert_frame_equal(success.drop(columns='success').sort_index(),
        rating.reindex(success.index).sort_index(), check_dtype=False)
    assert success.dir_prior_success_rate_smoothed.between(0,1).all()
    print('PASS: exact boundaries, unknown finances, three-class decisions, metrics, shared predictors, and hit-history range.')


if __name__=='__main__':verify()
