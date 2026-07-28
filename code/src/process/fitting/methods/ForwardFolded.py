import numpy as np
from astropy.modeling import FittableModel, Parameter


def integrate_flux(e1, e2, model_func, n_points=10):
    """
    Computes the average photon flux of a model over an energy bin
    using the trapezoidal method.

    Parameters
    ----------
    e1 : float
        Lower bound of the energy bin (keV).
    e2 : float
        Upper bound of the energy bin (keV).
    model_func : callable
        Photon flux function: E -> Phi(E)
        [photons cm-2 s-1 keV-1].
    n_points : int, optional
        Number of quadrature points (default: 10).

    Returns
    -------
    float
        Average flux over the bin [photons cm-2 s-1 keV-1],
        i.e. trapz(Phi, E) / (e2 - e1).
    """
    if e1 >= e2:
        raise ValueError(f"e1 ({e1}) must be strictly < e2 ({e2})")

    energies = np.linspace(e1, e2, n_points)
    fluxes = model_func(energies)
    return np.trapezoid(fluxes, energies) / (e2 - e1)


# ══════════════════════════════════════════════════════════
#  0 — Power Law
# ══════════════════════════════════════════════════════════
class PowerLaw(FittableModel):
    """
    Simple power law convolved with the SRM.

    Phi(E) = amplitude * (E / E_pivot)^(-alpha)

    Parameters (astropy)
    --------------------
    amplitude : float, default 1e-2
        Flux normalisation at the pivot energy
        [photons cm-2 s-1 keV-1].
    alpha : float, default 2.0
        Spectral index (dimensionless).

    Parameters (__init__)
    ---------------------
    e_low_true : array-like
        Lower bounds of the true-energy bins (keV), shape (N,).
    e_high_true : array-like
        Upper bounds of the true-energy bins (keV), shape (N,).
    matrix : ndarray
        Instrument response matrix (SRM), shape (N, M).
    exposure : float
        Exposure time (s).
    E_pivot : float, optional
        Pivot energy (keV), default 100.0.

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Modelled count rate per measured channel [counts s-1].
    """

    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter(default=1e-2)
    alpha = Parameter(default=2.0)

    def __init__(self, e_low_true, e_high_true, matrix, exposure,
                 E_pivot=100.0, **kwargs):
        super().__init__(**kwargs)

        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix

        if float(exposure) <= 0:
            raise ValueError(f"exposure must be > 0, got: {exposure}")
        self.exposure = float(exposure)

        if float(E_pivot) <= 0:
            raise ValueError(f"E_pivot must be > 0, got: {E_pivot}")
        self.E_pivot = float(E_pivot)

    def evaluate(self, x, amplitude, alpha):

        def model_func(E):
            E = np.asarray(E, dtype=float)
            E_safe = np.maximum(E, 1e-12)
            return amplitude * (E_safe / self.E_pivot) ** (-alpha)

        true_fluxes = np.array([
            integrate_flux(e1, e2, model_func)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])

        folded = np.dot(true_fluxes, self.matrix) / self.exposure

        folded = np.nan_to_num(folded, nan=1e-30, posinf=0.0, neginf=0.0)
        return np.clip(folded, 1e-30, None)


# ══════════════════════════════════════════════════════════
#  1 — Broken Power Law
# ══════════════════════════════════════════════════════════
class BrokenPowerLaw(FittableModel):
    """
    Broken power law convolved with the SRM.

    Phi(E) = amplitude * (E/E_break)^(-alpha_1)  if E < E_break
           = amplitude * (E/E_break)^(-alpha_2)  if E >= E_break

    Parameters (astropy)
    --------------------
    amplitude : float, default 1e-2
        Normalisation at the break energy [photons cm-2 s-1 keV-1].
    E_break : float, default 10.0
        Break energy (keV).
    alpha_1 : float, default 2.0
        Spectral index below E_break.
    alpha_2 : float, default 3.0
        Spectral index above E_break.

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bounds of the true-energy bins (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Exposure time (s).

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Modelled count rate [counts s-1].
    """
    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter(default=1e-2)
    E_break = Parameter(default=10.0)
    alpha_1 = Parameter(default=2.0)
    alpha_2 = Parameter(default=3.0)

    def __init__(self, e_low_true, e_high_true, matrix, exposure, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure

    def evaluate(self, x, amplitude, E_break, alpha_1, alpha_2):
        def model_func(E):
            return amplitude * np.where(E < E_break, (E / E_break) ** (-alpha_1), (E / E_break) ** (-alpha_2))

        true_fluxes = np.array([
            integrate_flux(e1, e2, model_func)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])
        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded


# ══════════════════════════════════════════════════════════
#  3 — VTH
# ══════════════════════════════════════════════════════════
class VTH(FittableModel):
    """
    Optically thin thermal bremsstrahlung (Variable Thermal),
    convolved with the SRM.

    Phi(E) = (A_ff * EM) / (E * sqrt(T)) * exp(-E / T)
    with A_ff = 1.07e-42 * g_ff, g_ff = 1.2 (mean Gaunt factor).

    Parameters (astropy)
    --------------------
    EM : float, default 1e48, bounds [1e44, 1e52]
        Emission measure (cm-3), EM = integral(ne^2 dV).
    T : float, default 1.0, bounds [0.1, 50.0]
        Plasma temperature (keV).

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bounds of the true-energy bins (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Exposure time (s).

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Modelled count rate [counts s-1].

    Notes
    -----
    T is clamped to a 1e-3 keV minimum to avoid division by zero.
    """
    n_inputs = 1
    n_outputs = 1

    EM = Parameter(default=1e48, bounds=(1e44, 1e52))
    T = Parameter(default=1.0, bounds=(0.1, 50.0))

    def __init__(self, e_low_true, e_high_true, matrix, exposure, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure

    def evaluate(self, x, EM, T):
        # Constants
        gff = 1.2
        A = 1.07e-42 * gff

        # Safety: avoid division by zero or negative values
        safe_T = max(1e-3, T)

        def thermal_model(E):
            return (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T)

        true_fluxes = np.array([
            integrate_flux(e1, e2, thermal_model)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])

        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded


# ══════════════════════════════════════════════════════════
#  2 — Single Power Law Times an Exponential (ExpPowerLaw)
# ══════════════════════════════════════════════════════════
class ExpPowerLaw(FittableModel):
    """
    Power law multiplied by an exponential (empirical model),
    convolved with the SRM.

    Phi(E) = p0 * (E / p2)^p1 * exp(e3 - E / e4)

    Parameters (astropy)
    --------------------
    p0 : float, default 1.0
        Normalisation.
    p1 : float, default -2.0
        Spectral index.
    p2 : float, default 20.0
        Reference energy (keV).
    e3 : float, default 1.0
        Exponential offset (dimensionless).
    e4 : float, default 10.0
        Energy scale of the exponential cutoff (keV).

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bounds of the true-energy bins (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Exposure time (s).

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Modelled count rate [counts s-1].
    """
    n_inputs = 1
    n_outputs = 1

    p0 = Parameter(default=1.0)
    p1 = Parameter(default=-2.0)
    p2 = Parameter(default=20.0)
    e3 = Parameter(default=1.0)
    e4 = Parameter(default=10.0)

    def __init__(self, e_low_true, e_high_true, matrix, exposure, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure

    def evaluate(self, x, p0, p1, p2, e3, e4):
        def model_func(E):
            safe_E = np.where(E <= 0, 1e-6, E)
            return (p0 * (safe_E / p2) ** p1) * np.exp(e3 - safe_E / e4)

        true_fluxes = np.array([
            integrate_flux(e1, e2, model_func)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])
        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded


# ══════════════════════════════════════════════════════════
#  4 — V_TH + Power Law
# ══════════════════════════════════════════════════════════
class VTHPlusPowerLaw(FittableModel):
    """
    Additive superposition of a thermal component (VTH) and a power
    law, convolved with the SRM.

    Phi(E) = Phi_VTH(E) + Phi_PL(E)
    with Phi_VTH(E) = (A_ff * EM) / (E * sqrt(T)) * exp(-E / T)
    and  Phi_PL(E)  = amplitude * (E / E_pivot)^(-alpha)

    Parameters (astropy)
    --------------------
    EM : float, default 1e48, bounds [1e44, 1e52]
        Emission measure (cm-3).
    T : float, default 1.0, bounds [0.1, 50.0]
        Plasma temperature (keV).
    amplitude : float, default 1e-2
        Normalisation of the non-thermal component.
    alpha : float, default 2.0
        Spectral index of the power law.

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bounds of the true-energy bins (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Exposure time (s).
    E_pivot : float, optional
        Pivot energy of the power law (keV), default 100.0.

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Modelled count rate [counts s-1].
    """
    n_inputs = 1
    n_outputs = 1

    # VTH parameters
    EM = Parameter(default=1e48, bounds=(1e44, 1e52))
    T = Parameter(default=1.0, bounds=(0.1, 50.0))

    # Power Law parameters
    amplitude = Parameter(default=1e-2)
    alpha = Parameter(default=2.0)

    def __init__(self, e_low_true, e_high_true, matrix, exposure, E_pivot=100.0, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure
        self.E_pivot = E_pivot

    def evaluate(self, x, EM, T, amplitude, alpha):
        # Gaunt constant
        gff = 1.2
        A = 1.07e-42 * gff
        safe_T = max(1e-3, T)

        def model_total(E):
            # Thermal component
            thermal = (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T)
            # Power-law component
            power = amplitude * (E / self.E_pivot) ** (-alpha)
            return thermal + power

        # Integrate the photon flux within each SRM bin
        true_fluxes = np.array([
            integrate_flux(e1, e2, model_total)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])

        # Forward-folding through the SRM
        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded


# ══════════════════════════════════════════════════════════
#  5 — Power Law Cutoff Fix
# ══════════════════════════════════════════════════════════
class PowerLawCutoffFix(FittableModel):
    """
    Power law with a fixed hard cutoff (E_cut not fitted),
    convolved with the SRM.

    Phi(E) = amplitude * E^(-alpha)  if E >= E_cut
           = 0                        if E < E_cut

    E_cut is an instance attribute set at construction time; it can
    be changed by the user via the Set Function window but is not a
    free parameter of the fit.

    Parameters (astropy)
    --------------------
    amplitude : float, default 1e-2
        Normalisation [photons cm-2 s-1 keV-1].
    alpha : float, default 2.0
        Spectral index.

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bounds of the true-energy bins (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Exposure time (s).
    E_cut : float, optional
        Fixed cutoff energy (keV), default 10.0.
    E_pivot : float, optional
        Pivot energy (keV), default 100.0.

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Modelled count rate [counts s-1].
    """
    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter(default=1e-2)
    alpha = Parameter(default=2.0, min=1e-5)

    def __init__(self, e_low_true, e_high_true, matrix, exposure, E_cut=10.0, E_pivot=100.0, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.E_cut = E_cut
        self.exposure = exposure
        self.E_pivot = E_pivot

    def evaluate(self, x, amplitude, alpha):
        model_func = lambda E: np.where(E >= self.E_cut, amplitude * (E / self.E_pivot) ** (-alpha), 0.0)

        true_fluxes = np.array([
            integrate_flux(e1, e2, model_func)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])

        folded = np.dot(true_fluxes, self.matrix) / self.exposure

        return folded
