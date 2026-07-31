import numpy as np
import pytest

from process.fitting.methods.ForwardFolded import (
    BrokenPowerLaw,
    ExpPowerLaw,
    PowerLaw,
    PowerLawCutoffFix,
    VTH,
    VTHPlusPowerLaw,
    integrate_flux,
)


# ---------------------------------------------------------------------------
# integrate_flux
# ---------------------------------------------------------------------------

def test_integrate_flux_of_constant_function_equals_constant():
    result = integrate_flux(1.0, 5.0, lambda E: np.full_like(E, 3.0))
    assert result == pytest.approx(3.0)


def test_integrate_flux_of_linear_function_equals_average():
    # trapz(a + b*E) / (e2 - e1) == a + b * midpoint
    result = integrate_flux(2.0, 6.0, lambda E: 10.0 + 2.0 * E, n_points=50)
    expected = 10.0 + 2.0 * 4.0  # midpoint of [2, 6] is 4
    assert result == pytest.approx(expected, rel=1e-3)


def test_integrate_flux_rejects_non_increasing_bounds():
    with pytest.raises(ValueError):
        integrate_flux(5.0, 1.0, lambda E: E)


# ---------------------------------------------------------------------------
# Shared fixtures: a trivial identity SRM so folded == true photon flux
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_srm():
    e_low = np.array([1.0, 2.0, 3.0])
    e_high = np.array([2.0, 3.0, 4.0])
    matrix = np.eye(3)
    exposure = 2.0
    return e_low, e_high, matrix, exposure


# ---------------------------------------------------------------------------
# PowerLaw
# ---------------------------------------------------------------------------

def test_power_law_flat_spectrum_folded_through_identity(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    model = PowerLaw(e_low, e_high, matrix, exposure, E_pivot=1.0)

    # alpha=0 -> Phi(E) = amplitude everywhere -> folded == amplitude / exposure
    folded = model.evaluate(np.zeros(3), amplitude=1.0, alpha=0.0)
    np.testing.assert_allclose(folded, np.full(3, 0.5), rtol=1e-6)


def test_power_law_output_is_positive_and_finite(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    model = PowerLaw(e_low, e_high, matrix, exposure)

    folded = model.evaluate(np.zeros(3), amplitude=1e-2, alpha=2.0)
    assert np.all(np.isfinite(folded))
    assert np.all(folded > 0)


@pytest.mark.parametrize("exposure", [0.0, -1.0])
def test_power_law_rejects_non_positive_exposure(identity_srm, exposure):
    e_low, e_high, matrix, _ = identity_srm
    with pytest.raises(ValueError):
        PowerLaw(e_low, e_high, matrix, exposure)


@pytest.mark.parametrize("e_pivot", [0.0, -10.0])
def test_power_law_rejects_non_positive_e_pivot(identity_srm, e_pivot):
    e_low, e_high, matrix, exposure = identity_srm
    with pytest.raises(ValueError):
        PowerLaw(e_low, e_high, matrix, exposure, E_pivot=e_pivot)


# ---------------------------------------------------------------------------
# BrokenPowerLaw
# ---------------------------------------------------------------------------

def test_broken_power_law_matches_power_law_when_indices_equal(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm

    broken = BrokenPowerLaw(e_low, e_high, matrix, exposure)
    folded_broken = broken.evaluate(
        np.zeros(3), amplitude=1.0, E_break=2.5, alpha_1=2.0, alpha_2=2.0)

    plain = PowerLaw(e_low, e_high, matrix, exposure, E_pivot=2.5)
    folded_plain = plain.evaluate(np.zeros(3), amplitude=1.0, alpha=2.0)

    np.testing.assert_allclose(folded_broken, folded_plain, rtol=1e-6)


# ---------------------------------------------------------------------------
# VTH
# ---------------------------------------------------------------------------

def test_vth_output_is_positive_and_finite(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    model = VTH(e_low, e_high, matrix, exposure)

    folded = model.evaluate(np.zeros(3), EM=1e48, T=1.0)
    assert np.all(np.isfinite(folded))
    assert np.all(folded > 0)


def test_vth_scales_linearly_with_emission_measure(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    model = VTH(e_low, e_high, matrix, exposure)

    low = model.evaluate(np.zeros(3), EM=1e48, T=2.0)
    high = model.evaluate(np.zeros(3), EM=2e48, T=2.0)
    np.testing.assert_allclose(high, 2.0 * low, rtol=1e-6)


# ---------------------------------------------------------------------------
# VTHPlusPowerLaw
# ---------------------------------------------------------------------------

def test_vth_plus_power_law_equals_sum_of_components(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm

    combo = VTHPlusPowerLaw(e_low, e_high, matrix, exposure, E_pivot=1.0)
    folded_combo = combo.evaluate(
        np.zeros(3), EM=1e48, T=2.0, amplitude=1.0, alpha=0.0)

    vth = VTH(e_low, e_high, matrix, exposure)
    folded_vth = vth.evaluate(np.zeros(3), EM=1e48, T=2.0)

    pl = PowerLaw(e_low, e_high, matrix, exposure, E_pivot=1.0)
    folded_pl = pl.evaluate(np.zeros(3), amplitude=1.0, alpha=0.0)

    np.testing.assert_allclose(folded_combo, folded_vth + folded_pl, rtol=1e-6)


# ---------------------------------------------------------------------------
# ExpPowerLaw
# ---------------------------------------------------------------------------

def test_exp_power_law_output_is_finite(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    model = ExpPowerLaw(e_low, e_high, matrix, exposure)

    folded = model.evaluate(np.zeros(3), p0=1.0, p1=-2.0, p2=20.0, e3=1.0, e4=10.0)
    assert np.all(np.isfinite(folded))


# ---------------------------------------------------------------------------
# PowerLawCutoffFix
# ---------------------------------------------------------------------------

def test_power_law_cutoff_zeroes_flux_below_cutoff(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    # Cutoff above all bins -> everything folded to (near) zero.
    model = PowerLawCutoffFix(e_low, e_high, matrix, exposure, E_cut=100.0, E_pivot=1.0)

    folded = model.evaluate(np.zeros(3), amplitude=1.0, alpha=2.0)
    np.testing.assert_allclose(folded, np.zeros(3), atol=1e-12)


def test_power_law_cutoff_matches_power_law_when_cut_below_all_bins(identity_srm):
    e_low, e_high, matrix, exposure = identity_srm
    cutoff = PowerLawCutoffFix(e_low, e_high, matrix, exposure, E_cut=0.0, E_pivot=1.0)
    folded_cutoff = cutoff.evaluate(np.zeros(3), amplitude=1.0, alpha=2.0)

    plain = PowerLaw(e_low, e_high, matrix, exposure, E_pivot=1.0)
    folded_plain = plain.evaluate(np.zeros(3), amplitude=1.0, alpha=2.0)

    np.testing.assert_allclose(folded_cutoff, folded_plain, rtol=1e-6)
