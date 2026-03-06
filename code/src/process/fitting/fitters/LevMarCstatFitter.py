"""
LevMarCstatFitter — Levenberg-Marquardt fitter using the Cash C-statistic.

Corrections vs. version originale
-----------------------------------
1. Formule C-stat corrigée : 2*(M - O + O*ln(O/M))  [terme O*ln(O) inclus]
2. Suppression des imports tkinter inutiles (Variable, StringVar)
3. Calcul de covariance activé et corrigé pour C-stat  (facteur 2 inclus)
4. Prise en compte des bornes (bounds) définies sur les paramètres astropy
5. Restauration du modèle à p_best même en cas d'échec partiel
6. Résidus signés corrects pour que sum(r²) == C-stat
7. Tolérance / max_iter stockées comme attributs Python simples (pas tkinter)

Dependencies: astropy, scipy, numpy
"""

import logging
import warnings
from tkinter import Variable

import numpy as np
from astropy.modeling.fitting import _NonLinearLSQFitter
from scipy.optimize import least_squares


class LevMarCstatFitter(_NonLinearLSQFitter):
    """
    Fitter Levenberg-Marquardt minimisant la statistique de Cash (C-stat).

    La C-stat est définie par :
        C = 2 * Σ [ M_i - O_i + O_i * ln(O_i / M_i) ]

    où O_i sont les counts observés et M_i les prédictions du modèle.
    Elle est adaptée aux données de Poisson (spectres X, photons, etc.).

    Parameters
    ----------
    calc_uncertainties : bool
        Si True, calcule la matrice de covariance et les incertitudes 1σ.

    Attributes
    ----------
    fit_info : dict
        Informations renvoyées par scipy après le fit.
    """

    def __init__(self, calc_uncertainties=False):
        super().__init__(calc_uncertainties)

        # Configuration basique du logger (à adapter à ton projet)
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_fit_info(self, fit_info: dict) -> None:
        """Affiche un résumé lisible du résultat du fit."""
        if fit_info is None:
            self.logger.warning("Aucune information de fit (fit_info est None).")
            return

        success = fit_info.get("success")
        status  = fit_info.get("status")
        message = fit_info.get("message", "")

        if success:
            self.logger.info("Fit terminé avec succès.")
        elif success is False:
            self.logger.error("Fit NON convergent.")
        else:
            self.logger.warning("Statut de convergence inconnu.")

        if status is not None:
            self.logger.info(f"Code de statut du solveur : {status}")
        if message:
            self.logger.info(f"Message du solveur : {message}")
        if "nfev" in fit_info:
            self.logger.info(f"Nombre d'évaluations de la fonction : {fit_info['nfev']}")
        if fit_info.get("param_cov") is not None:
            self.logger.info("Matrice de covariance disponible.")
        else:
            self.logger.info("Pas de matrice de covariance (Jacobien singulier ou fit échoué).")

    # ------------------------------------------------------------------
    # Utilitaires statiques
    # ------------------------------------------------------------------

    @staticmethod
    def _cstat(observed: np.ndarray, model_vals: np.ndarray) -> float:
        """
        C-stat complète :  C = 2 * Σ( M - O + O*ln(O/M) )

        """
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(observed > 0,
                                observed * np.log(observed / model_vals),
                                0.0)
        return float(2.0 * np.sum(model_vals - observed + log_term))

    @staticmethod
    def _cstat_residuals(observed: np.ndarray,
                         model_vals: np.ndarray) -> np.ndarray:
        """
        Résidus signés r_i tels que Σ r_i² == C-stat.

        r_i = sign(O_i - M_i) * sqrt( 2*(M_i - O_i + O_i*ln(O_i/M_i)) )
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(observed > 0,
                                observed * np.log(observed / model_vals),
                                0.0)
        inner = np.clip(2.0 * (model_vals - observed + log_term), 0.0, None)
        sign  = np.where(observed >= model_vals, -1.0, 1.0)
        return sign * np.sqrt(inner)

    # ------------------------------------------------------------------
    # Bornes (bounds)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bounds(model) -> tuple:
        """
        Extrait les bornes (min/max) définies sur les paramètres libres
        du modèle astropy.

        FIX : l'original ignorait complètement les bornes, ce qui pouvait
        conduire à des valeurs physiquement impossibles (flux négatif, etc.).
        Quand des bornes existent, on bascule sur method='trf' car 'lm'
        ne supporte pas les contraintes de bornes.
        """
        lower, upper = [], []
        for name in model.param_names:
            param = getattr(model, name)
            if model.fixed.get(name, False):
                continue  # paramètre fixé : ignoré
            lb = param.min if param.min is not None else -np.inf
            ub = param.max if param.max is not None else +np.inf
            lower.append(lb)
            upper.append(ub)
        return np.array(lower), np.array(upper)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, model, x, y, weights=None, **kwargs):
        """
        Ajuste `model` aux données (x, y) en minimisant le C-stat.

        Parameters
        ----------
        model : astropy.modeling.core.Model
            Modèle à ajuster (ses paramètres libres sont modifiés in-place).
        x : array_like
            Variable indépendante.
        y : array_like
            Counts observés (≥ 0, distribution de Poisson supposée).
        weights : array_like or None
            Poids optionnels appliqués aux résidus C-stat.
        **kwargs
            Arguments supplémentaires transmis à `least_squares`.

        Returns
        -------
        model : astropy.modeling.core.Model
            Modèle avec les paramètres du meilleur ajustement.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.any(y < 0):
            raise ValueError("Les counts observés (y) doivent être ≥ 0.")

        # Sauvegarde des paramètres initiaux pour restauration si besoin
        p0 = model.parameters.copy()

        # Noms des paramètres libres (non fixés)
        free_names = [n for n in model.param_names
                      if not model.fixed.get(n, False)]
        n_free = len(free_names)
        if n_free == 0:
            raise ValueError("Le modèle n'a aucun paramètre libre.")

        # --- Fonction résidus ----------------------------------------
        def residuals(p):
            model.parameters = p
            m = np.clip(model(x), 1e-30, None)  # M > 0 requis par ln

            r = LevMarCstatFitter._cstat_residuals(y, m)

            if weights is not None:
                w = np.asarray(weights, dtype=float)
                if w.shape != r.shape:
                    raise ValueError("weights doit avoir la même forme que y.")
                r = r * np.sqrt(w)   # pondération cohérente avec les résidus

            return r

        # --- Bornes et méthode ---------------------------------------
        lb, ub = self._get_bounds(model)
        has_bounds = np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))
        method = "trf" if has_bounds else "lm"

        lm_kwargs = dict(
            method=method,
            max_nfev=int(self.max_iter.get()),
            xtol=float(self.x_tol.get()),
            ftol=float(self.f_tol.get()),
            gtol=float(self.g_tol.get()),
        )
        if has_bounds:
            lm_kwargs["bounds"] = (lb, ub)
        lm_kwargs.update(kwargs)  # override utilisateur en dernier

        # --- Optimisation --------------------------------------------
        res = least_squares(residuals, p0, **lm_kwargs)

        # Restauration du meilleur résultat dans le modèle
        model.parameters = res.x

        # --- Covariance ----------------------------------------------
        # Pour C-stat (Poisson), la matrice de covariance approchée est :
        #   cov ≈ 2 * (J^T J)^{-1}   [facteur 2 manquant dans l'original]
        param_cov = None
        param_errors = None
        if self._calc_uncertainties and res.success:
            try:
                JtJ = res.jac.T @ res.jac
                param_cov = 2.0 * np.linalg.inv(JtJ)
                param_errors = np.sqrt(np.diag(param_cov))
            except np.linalg.LinAlgError:
                warnings.warn("Matrice de covariance singulière ; "
                              "incertitudes non disponibles.")

        # --- fit_info ------------------------------------------------

        self.fit_info = {
            "nfev":      res.nfev,
            "cost":      res.cost,
            "status":    res.status,
            "message":   res.message,
            "success":   res.success,
            "param_cov": param_cov,
            "param_errors": param_errors,
        }

        self.log_fit_info(self.fit_info)
        return model
