import pytest
import numpy as np
from scipy.optimize import check_grad, approx_fprime

from cblearn.datasets import make_random_triplets
from cblearn.embedding import STE, TSTE
from cblearn.embedding._ste import _ste_x_grad


@pytest.mark.parametrize('n,d', [(20, 1), (50, 2), (100, 3)])
@pytest.mark.parametrize('heavy_tailed', [False, True])
def test_ste_gradient(n, d, heavy_tailed):
    """ Test the gradient of the STE loss against a finite difference approximation.

    The loss is -log(P) with P = kernel_ij / (kernel_ij + kernel_ik), so the outer
    derivative is -(1 - P). Weighting the summands with the derivative of P itself,
    P * (1 - P), scales every triplet by P and gives a relative error near one.
    """
    def fun(x, *args):
        return _ste_x_grad(x, *args)[0]

    def grad(x, *args):
        return _ste_x_grad(x, *args)[1]

    random_state = np.random.RandomState(n * d)
    X = random_state.randn(n, d)
    T = make_random_triplets(X, size=int(2 * d * n * np.log(n)), result_format='list-order')
    for i in range(5):  # test at 5 different points in the param space
        # Keep the points close together, so that exp(-distance) stays well above
        # the epsilon in the denominator of P and the loss remains smooth.
        init = random_state.rand(n, d) - 0.5
        args = [init.ravel(), X.shape, T, heavy_tailed]
        numeric_norm = np.linalg.norm(approx_fprime(init.ravel(), fun, 1.49e-8, *args[1:]))
        assert check_grad(fun, grad, *args) / numeric_norm < 1e-4


@pytest.mark.parametrize('estimator', [STE, TSTE])
def test_ste_recovers_noiseless_embedding(estimator):
    """ Test that the optimization recovers an embedding that explains the triplets.

    A damped gradient still points roughly downhill, so it produces an embedding
    that is better than chance but stops far short of the noiseless optimum.
    """
    n, d = 15, 2
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, d)
    T = make_random_triplets(X, size=1000, result_format='list-order', random_state=random_state)

    scores = []
    for repeat in range(5):
        estimated = estimator(n_components=d, random_state=repeat).fit(T, n_objects=n)
        scores.append(estimated.score(T))

    assert np.mean(scores) > 0.9, f"mean triplet accuracy {np.mean(scores)} is too low"
