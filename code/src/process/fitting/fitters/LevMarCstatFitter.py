import logging
import warnings
from tkinter import Variable

import numpy as np
from astropy.modeling.fitting import _NonLinearLSQFitter
from scipy.optimize import least_squares


class LevMarCstatFitter(_NonLinearLSQFitter):
    """
    Levenberg-Marquardt fitter minimising the Cash statistic (C-stat)
    rather than the classic chi-square, suited to low-count data
    (Poisson regime).

    C = 2 * sum_i | M_i - D_i * ln(M_i) |

    Inherits from astropy.modeling.fitting._NonLinearLSQFitter.

    Attributes
    ----------
    fit_info : dict
        Diagnostic dictionary filled after each call:
        {'nfev', 'cost', 'status', 'message', 'success'}.
    logger : logging.Logger
        Internal logger for convergence messages.
    """

    def __init__(self, calc_uncertainties=False):
        """
        Parameters
        ----------
        calc_uncertainties : bool, optional
            Whether to compute parameter uncertainties (default: False).
            Passed through to the parent class _NonLinearLSQFitter.
        """
        super().__init__(calc_uncertainties)

        # Basic logger configuration (adapt to your project as needed)
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_fit_info(self, fit_info: dict) -> None:
        """
        Logs the fitter's convergence result from the fit_info dictionary.

        Parameters
        ----------
        fit_info : dict or None
            Dictionary returned after the fit. Expected keys:
            - 'success' (bool) : whether convergence was reached.
            - 'status' (int)   : return code from the scipy solver.
            - 'message' (str)  : text message from the solver.
            - 'nfev' (int)     : number of cost function evaluations.
            - 'cost' (float)   : final value of C-stat / 2.
            - 'param_cov'      : covariance matrix (if available).

        Returns
        -------
        None
            Logs via self.logger (INFO / WARNING / ERROR / DEBUG).
        """
        if fit_info is None:
            self.logger.warning("No fit information available (fit_info is None).")
            return

        success = fit_info.get("success")
        status  = fit_info.get("status")
        message = fit_info.get("message", "")

        if success:
            self.logger.info("Fit completed successfully.")
        elif success is False:
            self.logger.error("Fit did NOT converge.")
        else:
            self.logger.warning("Unknown convergence status.")

        if status is not None:
            self.logger.info(f"Solver status code: {status}")
        if message:
            self.logger.info(f"Solver message: {message}")
        if "nfev" in fit_info:
            self.logger.info(f"Number of function evaluations: {fit_info['nfev']}")
        if fit_info.get("param_cov") is not None:
            self.logger.info("Covariance matrix available.")
        else:
            self.logger.info("No covariance matrix (singular Jacobian or failed fit).")

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cstat(observed: np.ndarray, model_vals: np.ndarray) -> float:
        """
        Full C-stat:  C = 2 * Σ( M - O + O*ln(O/M) )

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
        Signed residuals r_i such that Σ r_i² == C-stat.

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
    # Bounds
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bounds(model) -> tuple:
        """
        Extracts the bounds (min/max) defined on the astropy model's
        free parameters.
        """
        lower, upper = [], []
        for name in model.param_names:
            param = getattr(model, name)
            if model.fixed.get(name, False):
                continue  # fixed parameter: skipped
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
        Runs the model fit by minimising the C-stat residual vector via
        scipy.optimize.least_squares (method 'lm').

        The vectorised residual function is:
            r_i = 2 * | M_i(theta) - D_i * ln(M_i(theta)) |
        with M_i clamped to 1e-12 to avoid ln(0).

        Parameters
        ----------
        model : astropy FittableModel
            Model to fit. Its parameters are modified in place by the
            fitter.
        x : array-like
            Independent variable (dummy zero vector for ForwardFolded
            models).
        y : array-like
            Observed data D_i (counts s-1 or counts depending on unit).
        weights : array-like or None, optional
            Weights applied to the residual vector. Must have the same
            shape as y. If None, all points have equal weight.
        **kwargs
            Extra arguments passed through to scipy.least_squares.

        Returns
        -------
        model : astropy FittableModel
            Model with its parameters updated to the optimal solution.

        Side effects
        ------------
        self.fit_info is updated with the convergence metrics.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.any(y < 0):
            raise ValueError("Observed counts (y) must be ≥ 0.")

        # Save the initial parameters in case they need to be restored
        p0 = model.parameters.copy()

        # Names of the free (non-fixed) parameters
        free_names = [n for n in model.param_names
                      if not model.fixed.get(n, False)]
        n_free = len(free_names)
        if n_free == 0:
            raise ValueError("The model has no free parameters.")

        # --- Residual function ----------------------------------------
        def residuals(p):
            model.parameters = p
            m = np.clip(model(x), 1e-30, None)  # M > 0 required by ln

            r = LevMarCstatFitter._cstat_residuals(y, m)

            if weights is not None:
                w = np.asarray(weights, dtype=float)
                if w.shape != r.shape:
                    raise ValueError("weights must have the same shape as y.")
                r = r * np.sqrt(w)   # weighting consistent with the residuals

            return r

        # --- Bounds and method ---------------------------------------
        lb, ub = self._get_bounds(model)
        has_bounds = np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))
        method = "trf" if has_bounds else "lm"

        lm_kwargs = dict(
            method=method,
        )
        if has_bounds:
            lm_kwargs["bounds"] = (lb, ub)
        lm_kwargs.update(kwargs)  # user override takes precedence

        # --- Optimisation --------------------------------------------
        res = least_squares(residuals, p0, **lm_kwargs)

        # Restore the best result into the model
        model.parameters = res.x

        # --- Covariance ----------------------------------------------
        # For C-stat (Poisson), the approximate covariance matrix is:
        #   cov ≈ 2 * (J^T J)^{-1}
        # (the factor of 2 comes from C-stat being 2x the log-likelihood,
        # unlike the classic chi-square case)
        param_cov = None
        param_errors = None
        if self._calc_uncertainties and res.success:
            try:
                JtJ = res.jac.T @ res.jac
                param_cov = 2.0 * np.linalg.inv(JtJ)
                param_errors = np.sqrt(np.diag(param_cov))
            except np.linalg.LinAlgError:
                warnings.warn("Singular covariance matrix; "
                              "uncertainties unavailable.")

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
