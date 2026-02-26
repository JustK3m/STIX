import logging

import numpy as np
from astropy.modeling import models
from astropy.modeling.fitting import _NonLinearLSQFitter
from scipy.optimize import least_squares


class LevMarCstatFitter(_NonLinearLSQFitter):

    def __init__(self, calc_uncertainties=False):
        super().__init__(calc_uncertainties)

        # Configuration basique du logger (à adapter à ton projet)
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(message)s"
        )

        self.logger = logging.getLogger(__name__)


    def log_fit_info(self, fit_info):
        """
        Affiche des logs dépendamment du contenu de fit_info.

        Parameters
        ----------
        fit_info : dict
            Dictionnaire retourné par le fitter (par ex. CStatLevMarFitter.fit_info).
        """

        if fit_info is None:
            self.logger.warning("Aucune information de fit (fit_info est None).")
            return

        # 1) Statut global du fit
        success = fit_info.get("success", None)
        status = fit_info.get("status", None)
        message = fit_info.get("message", "")

        if success:
            self.logger.info("Fit terminé avec succès.")
        elif success is False:
            self.logger.error("Fit NON convergent.")
        else:
            self.logger.warning("Statut de convergence inconnu (champ 'success' manquant).")

        if status is not None:
            self.logger.info(f"Code de statut du solveur : {status}")

        if message:
            self.logger.info(f"Message du solveur : {message}")

        # 2) Infos numériques si disponibles
        if "nfev" in fit_info:
            self.logger.info(f"Nombre d'évaluations de la fonction : {fit_info['nfev']}")

        if "cost" in fit_info:
            self.logger.info(f"Valeur finale du coût (cost) : {fit_info['cost']}")

        if "param_cov" in fit_info and fit_info["param_cov"] is not None:
            self.logger.info("Matrice de covariance des paramètres disponible.")
        else:
            self.logger.info("Pas de matrice de covariance des paramètres.")

        # 3) Champ générique pour debug (ex: jacobienne, résidus, etc.)
        reste = {k: v for k, v in fit_info.items()
                 if k not in ("success", "status", "message", "nfev", "cost", "param_cov")}
        if reste:
            self.logger.debug(f"Autres informations dans fit_info : {list(reste.keys())}")


    def smart_init(self, model, x, y):
        """Estimation rapide des paramètres à partir des données"""
        if isinstance(model, models.Gaussian1D):
            return [np.max(y), np.mean(x), np.std(x)]
        # etc.
        return model.parameters


    def __call__(self, model, x, y, weights=None, **kwargs):
        x, y = np.asarray(x), np.asarray(y)
        n_points = len(x)
        p0 = self.smart_init(model, x, y)

        def residuals(p):
            model.parameters = p
            m = model(x)
            m = np.clip(m, 1e-12, None)
            return np.sqrt(2.0 * np.abs(m - y * np.log(m)))  # vectorisé

        # LM moderne + Jacobienne
        res = least_squares(
            residuals, p0,
            method='lm',
            jac='2-point',
            xtol=1e-6,  # tolérances moins strictes
            ftol=1e-6,
            max_nfev=3000,
            **kwargs
        )

        model.parameters = res.x

        # stocker des infos comme fit_info dans LevMarLSQFitter
        self.fit_info = {
            "nfev": res.nfev,
            "cost": res.cost,
            "status": res.status,
            "message": res.message,
            "success": res.success,
            "param_cov": np.linalg.inv(res.jac.T.dot(res.jac)) * res.cost if res.success else None
        }
        self.log_fit_info(self.fit_info)
        return model