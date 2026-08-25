import numpy as np
import pytest

from cblearn.datasets import make_random_triplets
from cblearn.embedding import MLDS


def test_mlds_rejects_multiple_components():
    """ Test that MLDS rejects n_components != 1, which it cannot estimate.

    scikit-learn requires that __init__ stores the parameters unaltered and
    raises nothing, so the check has to happen in fit.
    """
    X = np.sort(np.random.uniform(1, 2, (10, 1)), axis=0)
    T = make_random_triplets(X, size=100, monotonic=True, result_format='list-order')

    MLDS(n_components=2)  # constructing must not raise
    with pytest.raises(ValueError, match="MLDS expects n_components=1"):
        MLDS(n_components=2).fit(T)

    assert MLDS(n_components=1).fit(T).embedding_.shape == (10, 1)
