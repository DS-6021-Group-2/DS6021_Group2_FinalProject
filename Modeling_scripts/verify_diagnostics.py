"""Numerical regression checks for diagnostics; no project datasets are changed."""
import numpy as np

from model_diagnostics import collinearity


def verify():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(300, 4))
    x[:, 1] = 0.8 * x[:, 0] + 0.2 * x[:, 1]
    actual = collinearity(x, list('abcd'))['vif'].set_index('Feature')
    # Independent definition: regress each column on all others and an intercept.
    for j, name in enumerate('abcd'):
        others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        residual = x[:, j] - others @ np.linalg.lstsq(others, x[:, j], rcond=None)[0]
        expected = np.sum((x[:, j] - x[:, j].mean()) ** 2) / np.sum(residual ** 2)
        np.testing.assert_allclose(actual.loc[name, 'VIF'], expected, rtol=1e-10)
    # VIF should not depend on units or the origin of a variable's scale.
    rescaled = collinearity(x * [0.01, 100, 2, 5] + [3, 7, -2, 10], list('abcd'))
    np.testing.assert_allclose(actual.VIF.sort_index(),
                               rescaled['vif'].set_index('Feature').VIF.sort_index(), rtol=1e-10)
    redundant = np.column_stack([x[:, 0], 2 * x[:, 0], x[:, 2], np.ones(len(x))])
    actual = collinearity(redundant, ['a', 'twice_a', 'c', 'constant'])
    vif = actual['vif'].set_index('Feature').VIF
    assert np.isinf(vif.loc[['a', 'twice_a']]).all()
    assert np.isfinite(vif.loc['c']) and np.isnan(vif.loc['constant'])
    assert actual['summary'].iloc[0]['Centered rank'] == 2
    # Pairwise correlations alone cannot identify this three-column dependency.
    actual = collinearity(np.column_stack([x[:, 0], x[:, 2], x[:, 0] + x[:, 2]]),
                          ['a', 'b', 'sum'])
    assert np.isinf(actual['vif'].VIF).all()
    constant = collinearity(np.ones((30, 2)), ['one', 'also_one'])
    assert constant['vif'].VIF.isna().all()
    assert constant['summary'].iloc[0]['Constants'] == 2
    # Orthogonal components should have VIF 1.
    q, _ = np.linalg.qr(x - x.mean(axis=0))
    np.testing.assert_allclose(collinearity(q, list('abcd'))['vif'].VIF, 1, atol=1e-12)
    print('PASS: VIF matches independent OLS, is scale invariant, handles constants / exact dependencies, and recognizes orthogonal inputs.')


if __name__ == '__main__':
    verify()
