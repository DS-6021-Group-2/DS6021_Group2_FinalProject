"""Shared financial labels and three-class metrics (AI-assisted).

These codes are categories, not a continuous success score. Histories count hits.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             f1_score, roc_auc_score, log_loss)

CLASSES = [0, 1, 2]
CLASS_NAMES = ['Bust', 'Moderate success', 'Hit']
CLASS_LABELS = dict(zip(CLASSES, CLASS_NAMES))
HIT_CLASS = 2


def financial_labels(budget, revenue):
    """0: r/b <=1; 1: 1<r/b<2.5; 2: r/b>=2.5. Unknown stays missing."""
    budget = pd.to_numeric(pd.Series(budget), errors='coerce')
    revenue = pd.to_numeric(pd.Series(revenue), errors='coerce')
    usable = budget.gt(0) & revenue.gt(0) & np.isfinite(budget) & np.isfinite(revenue)
    labels = pd.Series(pd.NA, index=budget.index, dtype='Int64', name='success')
    labels.loc[usable] = np.select(
        [revenue.loc[usable].le(budget.loc[usable]), revenue.loc[usable].lt(2.5 * budget.loc[usable])],
        [0, 1], default=2,
    )
    return labels


def class_probabilities(estimator, x):
    """Align estimator outputs to the documented [Bust, Moderate, Hit] order."""
    assert set(estimator.classes_) == set(CLASSES), 'Fit all three financial classes.'
    result = estimator.predict_proba(x)[:, [list(estimator.classes_).index(c) for c in CLASSES]]
    validate_probabilities(result)
    return result


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-8)
    return p


def predict_classes(probabilities, hit_threshold=0.5):
    """Precision-oriented decision: hit above threshold, otherwise stronger non-hit.

    None uses ordinary three-class argmax. A thresholded hit can differ from argmax;
    probabilities remain unchanged, and moderate success remains a real third class.
    """
    p = validate_probabilities(probabilities)
    if hit_threshold is None:
        return np.argmax(p, axis=1)
    assert 0 < hit_threshold < 1
    return np.where(p[:, HIT_CLASS] >= hit_threshold, HIT_CLASS, np.argmax(p[:, :2], axis=1))


def financial_metrics(actual, predicted, probabilities):
    p = validate_probabilities(probabilities)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    assert np.isin(actual, CLASSES).all() and np.isin(predicted, CLASSES).all()
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0,
    )
    hit_actual = actual == HIT_CLASS
    # Keep ROC AUC explicitly hit-vs-rest; add multiclass macro OvR separately.
    result = {'Accuracy': float(accuracy_score(actual, predicted)),
              'Balanced accuracy': float(balanced_accuracy_score(actual, predicted)),
              'Macro F1': float(f1_score(actual, predicted, labels=CLASSES, average='macro', zero_division=0)),
              'ROC AUC': float(roc_auc_score(hit_actual, p[:, HIT_CLASS])),
              'Multiclass ROC AUC': float(roc_auc_score(actual, p, labels=CLASSES, multi_class='ovr', average='macro')),
              'Log loss': float(log_loss(actual, p, labels=CLASSES)),
              'Multiclass Brier': float(np.mean(np.sum((p - np.eye(3)[actual.astype(int)]) ** 2, axis=1)))}
    for i, label in enumerate(['Bust', 'Moderate', 'Hit']):
        result.update({f'{label} precision': float(precision[i]), f'{label} recall': float(recall[i]),
                       f'{label} F1': float(f1[i])})
    return result
