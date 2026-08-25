import pytest

from cblearn.embedding.wrapper._r_base import RWrapperMixin


def _forget_r_state():
    """ Drop the cached rpy2 handles, so the next call re-runs init_r. """
    for attribute in ('robjects', 'rpackages'):
        if hasattr(RWrapperMixin, attribute):
            delattr(RWrapperMixin, attribute)


@pytest.mark.parametrize('error', [TypeError, AttributeError])
def test_init_r_falls_back_to_activate_on_old_rpy2(monkeypatch, error):
    """ Test that init_r falls back to numpy2ri.activate if the Converter API is missing.

    Old rpy2 versions have no conversion.Converter to register a single direction
    with, so init_r has to fall back to the global activate().
    """
    pytest.importorskip('rpy2', reason='rpy2 is not installed')
    from rpy2.robjects import conversion, numpy2ri

    def raise_error(*args, **kwargs):
        raise error("simulated old rpy2")

    activate_calls = []
    monkeypatch.setattr(conversion, 'Converter', raise_error)
    monkeypatch.setattr(numpy2ri, 'activate', lambda: activate_calls.append(True))

    # init_r only runs when no handles are cached yet.
    _forget_r_state()
    try:
        RWrapperMixin.init_r()
        assert activate_calls == [True], "Expects the activate fallback to run exactly once"
        assert hasattr(RWrapperMixin, 'robjects')
    finally:
        # The stubbed activate did not register any converter, so leaving the
        # handles in place would make later R wrapper calls skip init_r and fail
        # with a py2rpy error. Drop them to force a clean re-initialization.
        _forget_r_state()
