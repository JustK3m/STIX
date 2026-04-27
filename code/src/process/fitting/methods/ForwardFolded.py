import numpy as np
from astropy.modeling import FittableModel, Parameter


# function to calculate the flux
def integrate_flux(e1, e2, model_func, n_points=10):
    """
    Calcule le flux photonique moyen d'un modèle sur un bin en énergie
    par la méthode des trapèzes.

    Parameters
    ----------
    e1 : float
        Borne inférieure du bin en énergie (keV).
    e2 : float
        Borne supérieure du bin en énergie (keV).
    model_func : callable
        Fonction du flux photonique : E -> Phi(E)
        [photons cm-2 s-1 keV-1].
    n_points : int, optional
        Nombre de points de quadrature (défaut : 10).

    Returns
    -------
    float
        Flux moyen sur le bin [photons cm-2 s-1 keV-1],
        soit trapz(Phi, E) / (e2 - e1).
    """
    if e1 >= e2:
        raise ValueError(f"e1 ({e1}) doit être strictement < e2 ({e2})")

    energies = np.linspace(e1, e2, n_points)
    fluxes = model_func(energies)
    return np.trapezoid(fluxes, energies) / (e2 - e1)


# ══════════════════════════════════════════════════════════
#  0 — Power Law
# ══════════════════════════════════════════════════════════
class PowerLaw(FittableModel):
    """
    Loi de puissance simple convoluée par la SRM.

    Phi(E) = amplitude * (E / E_pivot)^(-alpha)

    Parameters (astropy)
    --------------------
    amplitude : float, défaut 1e-2
        Normalisation du flux à l'énergie pivot
        [photons cm-2 s-1 keV-1].
    alpha : float, défaut 2.0
        Indice spectral (sans dimension).

    Parameters (__init__)
    ---------------------
    e_low_true : array-like
        Bornes inférieures des bins en énergie vraie (keV), shape (N,).
    e_high_true : array-like
        Bornes supérieures des bins en énergie vraie (keV), shape (N,).
    matrix : ndarray
        Matrice de réponse instrumentale SRM, shape (N, M).
    exposure : float
        Temps d'exposition (s).
    E_pivot : float, optional
        Énergie pivot (keV), défaut 100.0.

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Taux de comptage modélisé par canal mesuré [coups s-1].
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
            raise ValueError(f"exposure doit être > 0, reçu : {exposure}")
        self.exposure = float(exposure)

        if float(E_pivot) <= 0:
            raise ValueError(f"E_pivot doit être > 0, reçu : {E_pivot}")
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
    Loi de puissance brisée convoluée par la SRM.

    Phi(E) = amplitude * (E/E_break)^(-alpha_1)  si E < E_break
           = amplitude * (E/E_break)^(-alpha_2)  si E >= E_break

    Parameters (astropy)
    --------------------
    amplitude : float, défaut 1e-2
        Normalisation à l'énergie de cassure [photons cm-2 s-1 keV-1].
    E_break : float, défaut 10.0
        Énergie de cassure (keV).
    alpha_1 : float, défaut 2.0
        Indice spectral sous E_break.
    alpha_2 : float, défaut 3.0
        Indice spectral au-dessus de E_break.

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bornes des bins en énergie vraie (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Temps d'exposition (s).

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Taux de comptage modélisé [coups s-1].
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
#  2 — Exp Power Law
# ══════════════════════════════════════════════════════════
class VTH(FittableModel):
    """
    Bremsstrahlung thermique optiquement mince (Variable Thermal),
    convolué par la SRM.

    Phi(E) = (A_ff * EM) / (E * sqrt(T)) * exp(-E / T)
    avec A_ff = 1.07e-42 * g_ff, g_ff = 1.2 (facteur de Gaunt moyen).

    Parameters (astropy)
    --------------------
    EM : float, défaut 1e48, bornes [1e44, 1e52]
        Mesure d'émission (cm-3), EM = integral(ne^2 dV).
    T : float, défaut 1.0, bornes [0.1, 50.0]
        Température du plasma (keV).

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bornes des bins en énergie vraie (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Temps d'exposition (s).

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Taux de comptage modélisé [coups s-1].

    Notes
    -----
    T est clampé à 1e-3 keV minimum pour éviter les divisions par zéro.
    """
    n_inputs = 1
    n_outputs = 1

    # T = Parameter(default=10.0)      # Température en keV
    # EM = Parameter(default=1e49)     # Emission Measure en cm^-3

    EM = Parameter(default=1e48, bounds=(1e44, 1e52))
    T = Parameter(default=1.0, bounds=(0.1, 50.0))

    def __init__(self, e_low_true, e_high_true, matrix, exposure, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure

    def evaluate(self, x, EM, T):
        # Constantes
        gff = 1.2
        A = 1.07e-42 * gff

        # Sécurité : éviter division par zéro ou valeurs négatives
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
#  3 — V_TH
# ══════════════════════════════════════════════════════════
class ExpPowerLaw(FittableModel):
    """
    Loi de puissance multipliée par une exponentielle (modèle empirique),
    convoluée par la SRM.

    Phi(E) = p0 * (E / p2)^p1 * exp(e3 - E / e4)

    Parameters (astropy)
    --------------------
    p0 : float, défaut 1.0
        Normalisation.
    p1 : float, défaut -2.0
        Indice spectral.
    p2 : float, défaut 20.0
        Énergie de référence (keV).
    e3 : float, défaut 1.0
        Offset exponentiel (sans dimension).
    e4 : float, défaut 10.0
        Échelle d'énergie de la coupure exponentielle (keV).

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bornes des bins en énergie vraie (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Temps d'exposition (s).

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Taux de comptage modélisé [coups s-1].
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
    Superposition additive d'une composante thermique (VTH) et d'une
    loi de puissance, convoluée par la SRM.

    Phi(E) = Phi_VTH(E) + Phi_PL(E)
    avec Phi_VTH(E) = (A_ff * EM) / (E * sqrt(T)) * exp(-E / T)
    et  Phi_PL(E)  = amplitude * (E / E_pivot)^(-alpha)

    Parameters (astropy)
    --------------------
    EM : float, défaut 1e48, bornes [1e44, 1e52]
        Mesure d'émission (cm-3).
    T : float, défaut 1.0, bornes [0.1, 50.0]
        Température du plasma (keV).
    amplitude : float, défaut 1e-2
        Normalisation de la composante non-thermique.
    alpha : float, défaut 2.0
        Indice spectral de la loi de puissance.

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bornes des bins en énergie vraie (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Temps d'exposition (s).
    E_pivot : float, optional
        Énergie pivot de la loi de puissance (keV), défaut 100.0.

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Taux de comptage modélisé [coups s-1].
    """
    n_inputs = 1
    n_outputs = 1

    # Paramètres VTH
    EM = Parameter(default=1e48, bounds=(1e44, 1e52))
    T = Parameter(default=1.0, bounds=(0.1, 50.0))

    # Paramètres Power Law
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
        # Constante de Gaunt
        gff = 1.2
        A = 1.07e-42 * gff
        safe_T = max(1e-3, T)

        def model_total(E):
            # Thermal component
            thermal = (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T)
            # Power-law component
            power = amplitude * (E / self.E_pivot) ** (-alpha)
            return thermal + power

        # Intégration du flux photonique dans chaque bin SRM
        true_fluxes = np.array([
            integrate_flux(e1, e2, model_total)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])

        # Forward-folding via SRM
        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded


# ══════════════════════════════════════════════════════════
#  5 — Power Law Cutoff Fix
# ══════════════════════════════════════════════════════════
class PowerLawCutoffFix(FittableModel):
    """
    Loi de puissance avec coupure dure fixe (E_cut non ajusté),
    convoluée par la SRM.

    Phi(E) = amplitude * E^(-alpha)  si E >= E_cut
           = 0                        si E < E_cut

    E_cut est un attribut d'instance fixé à la construction ; il peut
    être modifié par l'utilisateur via la fenêtre Set Function mais
    n'est pas un paramètre libre de l'ajustement.

    Parameters (astropy)
    --------------------
    amplitude : float, défaut 1e-2
        Normalisation [photons cm-2 s-1 keV-1].
    alpha : float, défaut 2.0
        Indice spectral.

    Parameters (__init__)
    ---------------------
    e_low_true, e_high_true : array-like
        Bornes des bins en énergie vraie (keV).
    matrix : ndarray
        SRM, shape (N, M).
    exposure : float
        Temps d'exposition (s).
    E_cut : float, optional
        Énergie de coupure fixe (keV), défaut 10.0.

    Returns (evaluate)
    ------------------
    ndarray, shape (M,)
        Taux de comptage modélisé [coups s-1].
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


# # ══════════════════════════════════════════════════════════
# #  6 — Power Law Cutoff Free
# # ══════════════════════════════════════════════════════════
# class PowerLawCutoffFree(FittableModel):
#     """
#     Power law avec cutoff libre (E_c est un Parameter ajusté).
#
#     P(E) = A * E^(-alpha)  si E >= E_cut
#            0               sinon  (coupure dure)
#
#     Parameters
#     ----------
#     amplitude : float
#         Normalisation du flux (> 0).
#     alpha     : float
#         Indice spectral (> 0).
#     E_cut     : float
#         Énergie de coupure en keV (> 0).
#     """
#
#     n_inputs = 1
#     n_outputs = 1
#
#     amplitude = Parameter(default=1e-2)  # flux > 0 obligatoire
#     alpha = Parameter(default=2.0, min=1e-5)  # indice physique
#
#     def __init__(self, e_low_true, e_high_true, matrix, exposure, E_pivot=100.0, **kwargs):
#         super().__init__(**kwargs)
#         self.e_low_true = np.asarray(e_low_true, dtype=float)
#         self.e_high_true = np.asarray(e_high_true, dtype=float)
#         self.matrix = np.asarray(matrix, dtype=float)
#         self.exposure = exposure
#         self.E_pivot = E_pivot
#         self.E_cut = 10
#
#     def evaluate(self, x, amplitude, alpha):
#         model_func = lambda E: np.where(E >= self.E_cut, amplitude * (E / self.E_pivot) ** (-alpha), 0.0)
#
#         true_fluxes = np.array([
#             integrate_flux(e1, e2, model_func)
#             for e1, e2 in zip(self.e_low_true, self.e_high_true)
#         ])
#
#         folded = np.dot(true_fluxes, self.matrix) / self.exposure
#
#         return folded
#
#
# # ══════════════════════════════════════════════════════════
# #  7 — V_TH x Cutoff Fix
# # ══════════════════════════════════════════════════════════
# class VTHPowerLawCutoffFix(FittableModel):
#     n_inputs = 1
#     n_outputs = 1
#
#     # Paramètres VTH
#     EM = Parameter(default=1e48, bounds=(1e44, 1e52))
#     T = Parameter(default=1.0, bounds=(0.1, 50.0))
#
#     # Paramètres Power Law
#     amplitude = Parameter(default=1e-2)
#     alpha = Parameter(default=2.0, min=1e-5)
#
#     def __init__(self, e_low_true, e_high_true, matrix, exposure, E_cut=10.0, E_pivot=100, **kwargs):
#         super().__init__(**kwargs)
#         self.e_low_true = e_low_true
#         self.e_high_true = e_high_true
#         self.matrix = matrix
#         self.exposure = exposure
#         self.E_cut = E_cut
#         self.E_pivot = E_pivot
#
#     def evaluate(self, x, EM, T, amplitude, alpha):
#         # Constante de Gaunt
#         gff = 1.2
#         A = 1.07e-42 * gff
#         safe_T = max(1e-3, T)
#
#         def model_total(E):
#             # Power-law component
#             power = np.where(E >= self.E_cut, amplitude * (E / self.E_pivot) ** (-alpha),
#                              (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T))
#             return power
#
#         # Intégration du flux photonique dans chaque bin SRM
#         true_fluxes = np.array([
#             integrate_flux(e1, e2, model_total)
#             for e1, e2 in zip(self.e_low_true, self.e_high_true)
#         ])
#
#         # Forward-folding via SRM
#         folded = np.dot(true_fluxes, self.matrix) / self.exposure
#         return folded
#
#
# # ══════════════════════════════════════════════════════════
# #  8 — V_TH x Cutoff Free
# # ══════════════════════════════════════════════════════════
# class VTHPowerLawCutoffFree(FittableModel):
#     n_inputs = 1
#     n_outputs = 1
#
#     # Paramètres VTH
#     EM = Parameter(default=1e48, bounds=(1e44, 1e52))
#     T = Parameter(default=1.0, bounds=(0.1, 50.0))
#
#     # Paramètres Power Law
#     amplitude = Parameter(default=1e-2)
#     alpha = Parameter(default=2.0, min=1e-5)
#     E_cut = Parameter(default=10, min=4, max=120)
#
#     def __init__(self, e_low_true, e_high_true, matrix, exposure, E_pivot=100, **kwargs):
#         super().__init__(**kwargs)
#         self.e_low_true = e_low_true
#         self.e_high_true = e_high_true
#         self.matrix = matrix
#         self.exposure = exposure
#         self.E_pivot = E_pivot
#
#     def evaluate(self, x, EM, T, amplitude, alpha, E_cut):
#         # Constante de Gaunt
#         gff = 1.2
#         A = 1.07e-42 * gff
#         safe_T = max(1e-3, T)
#
#         def model_total(E):
#             # Power-law component
#             power = np.where(E >= E_cut, amplitude * (E / self.E_pivot) ** (-alpha),
#                              (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T))
#             return power
#
#         # Intégration du flux photonique dans chaque bin SRM
#         true_fluxes = np.array([
#             integrate_flux(e1, e2, model_total)
#             for e1, e2 in zip(self.e_low_true, self.e_high_true)
#         ])
#
#         # Forward-folding via SRM
#         folded = np.dot(true_fluxes, self.matrix) / self.exposure
#         return folded


# === Power Law Hard Cutoff ===
class PowerLawHardCutoff(FittableModel):
    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter(default=1e-2)
    alpha = Parameter(default=2.0)
    E_cut = Parameter(default=3.0)  # coupure dure

    def __init__(self, e_low_true, e_high_true, matrix, exposure, E_pivot=100.0, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure
        self.E_pivot = E_pivot

    def evaluate(self, x, amplitude, alpha, E_cut):
        # Power law avec coupure dure
        model_func = lambda E: np.where(
            E >= E_cut,
            amplitude * (E / self.E_pivot) ** (-alpha),
            0.0
        )
        true_fluxes = np.array([
            integrate_flux(e1, e2, model_func)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])
        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded


# === Power Law Cutoff ===
class PowerLawCutoff(FittableModel):
    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter(default=1e-2)
    alpha = Parameter(default=2.0)
    E_cut = Parameter(default=10.0)  # nouveau paramètre

    def __init__(self, e_low_true, e_high_true, matrix, exposure, E_pivot=100.0, **kwargs):
        super().__init__(**kwargs)
        self.e_low_true = e_low_true
        self.e_high_true = e_high_true
        self.matrix = matrix
        self.exposure = exposure
        self.E_pivot = E_pivot

    def evaluate(self, x, amplitude, alpha, E_cut):
        model_func = lambda E: amplitude * (E / self.E_pivot) ** (-alpha) * np.exp(-E / E_cut)
        true_fluxes = np.array([
            integrate_flux(e1, e2, model_func)
            for e1, e2 in zip(self.e_low_true, self.e_high_true)
        ])
        folded = np.dot(true_fluxes, self.matrix) / self.exposure
        return folded
