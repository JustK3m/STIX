import numpy as np
import pytest
from astropy.modeling.models import Polynomial1D

from process.fitting.fitters.LevMarCstatFitter import LevMarCstatFitter


# ---------------------------------------------------------------------------
# _cstat / _cstat_residuals
# ---------------------------------------------------------------------------

def test_cstat_is_zero_when_model_matches_observation():
    observed = np.array([5.0, 10.0, 20.0])
    assert LevMarCstatFitter._cstat(observed, observed) == pytest.approx(0.0, abs=1e-10)


def test_cstat_is_positive_when_model_differs_from_observation():
    observed = np.array([5.0, 10.0, 20.0])
    model = np.array([6.0, 9.0, 25.0])
    assert LevMarCstatFitter._cstat(observed, model) > 0


def test_cstat_residuals_sum_of_squares_equals_cstat():
    observed = np.array([1.0, 5.0, 0.0, 20.0])
    model = np.array([2.0, 4.0, 1.0, 18.0])

    residuals = LevMarCstatFitter._cstat_residuals(observed, model)
    cstat = LevMarCstatFitter._cstat(observed, model)

    assert np.sum(residuals ** 2) == pytest.approx(cstat, rel=1e-8)


def test_cstat_residuals_sign_matches_direction_of_deviation():
    observed = np.array([10.0, 10.0])
    model = np.array([5.0, 15.0])  # model under-predicts, then over-predicts

    residuals = LevMarCstatFitter._cstat_residuals(observed, model)
    # implementation convention: sign is negative when observed >= model
    assert residuals[0] < 0   # observed > model
    assert residuals[1] > 0   # observed < model


# ---------------------------------------------------------------------------
# _get_bounds
# ---------------------------------------------------------------------------

def test_get_bounds_defaults_to_infinite_when_unset():
    model = Polynomial1D(1, c0=1.0, c1=1.0)
    lower, upper = LevMarCstatFitter._get_bounds(model)

    assert np.all(np.isneginf(lower))
    assert np.all(np.isposinf(upper))


def test_get_bounds_reflects_min_max_and_skips_fixed_params():
    model = Polynomial1D(1, c0=1.0, c1=1.0)
    model.c0.min = -5.0
    model.c0.max = 5.0
    model.c1.fixed = True

    lower, upper = LevMarCstatFitter._get_bounds(model)

    assert lower.shape == (1,)
    assert lower[0] == -5.0
    assert upper[0] == 5.0


# ---------------------------------------------------------------------------
# __call__ (end-to-end fit)
# ---------------------------------------------------------------------------

def test_call_recovers_known_linear_parameters():
    fitter = LevMarCstatFitter()
    rng = np.random.default_rng(0)

    x = np.linspace(1, 10, 50)
    true_model = Polynomial1D(1, c0=5.0, c1=2.0)
    y = rng.poisson(np.clip(true_model(x), 1, None)).astype(float)

    model = Polynomial1D(1, c0=1.0, c1=1.0)
    fitted = fitter(model, x, y)

    assert fitter.fit_info["success"] is True
    assert fitted.c0.value == pytest.approx(5.0, rel=0.2)
    assert fitted.c1.value == pytest.approx(2.0, rel=0.2)


def test_call_rejects_negative_observed_counts():
    fitter = LevMarCstatFitter()
    model = Polynomial1D(1, c0=1.0, c1=1.0)

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, -2.0, 3.0])

    with pytest.raises(ValueError):
        fitter(model, x, y)


def test_call_raises_when_all_parameters_fixed():
    fitter = LevMarCstatFitter()
    model = Polynomial1D(1, c0=1.0, c1=1.0)
    model.c0.fixed = True
    model.c1.fixed = True

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        fitter(model, x, y)


def test_call_respects_parameter_bounds():
    fitter = LevMarCstatFitter()
    rng = np.random.default_rng(1)

    x = np.linspace(1, 10, 50)
    true_model = Polynomial1D(1, c0=5.0, c1=2.0)
    y = rng.poisson(np.clip(true_model(x), 1, None)).astype(float)

    model = Polynomial1D(1, c0=1.0, c1=0.25)
    model.c1.min = 0.0
    model.c1.max = 0.5  # force away from the true value of 2.0

    fitted = fitter(model, x, y)

    assert fitted.c1.value <= 0.5 + 1e-8
    assert fitted.c1.value >= 0.0 - 1e-8
