# fitting/fitters/LevMarCstatFitter.py

`LevMarCstatFitter` is a custom Astropy-compatible fitter: it implements the same callable contract as `astropy.modeling.fitting.LevMarLSQFitter` (`fitter(model, x, y, weights=None) -> model`), but minimizes the Cash statistic (C-stat) instead of chi-square. `fit_all.py`'s "Set statistics" menu swaps `self.fitter` between the two - every model in `fitting/methods/ForwardFolded.py` works with either, since neither the model classes nor `fit_all.py`'s fitting helpers know which fitter is active (see [../README.md](../README.md) for why this is possible).

## Why C-stat

Chi-square weighting (`1/sigma^2`) assumes Gaussian errors and becomes a poor approximation when the number of counts per energy channel is small (the Poisson regime - common at high energies or for weak flares, where a channel might have single-digit counts). C-stat is the exact Poisson maximum-likelihood statistic instead:

```
C = 2 * sum_i | M_i - D_i * ln(M_i) |
```

where `M_i` is the model prediction and `D_i` the observed data in channel `i`.

## Why a custom fitter is needed at all

Astropy ships fitters built around least-squares residual vectors (`scipy.optimize.least_squares` under the hood for `LevMarLSQFitter`), not around an arbitrary scalar statistic. To reuse the same Levenberg-Marquardt machinery for C-stat, `LevMarCstatFitter` reformulates it as a **signed residual vector** whose sum of squares equals C-stat:

```python
r_i = sign(O_i - M_i) * sqrt(2 * (M_i - O_i + O_i * ln(O_i / M_i)))
```

(`_cstat_residuals`) - note this is the *full* C-stat form `2*(M - O + O*ln(O/M))`, which reduces to `_cstat`'s `2*|M - O*ln(M)|` up to the constant `O*ln(O)` term (a `O*ln(O)` offset that does not depend on the model and therefore does not affect where the minimum is, but does make `_cstat`'s value not directly comparable to `_cstat_residuals`' sum of squares as an absolute number - keep this in mind if ever cross-checking the two). `residuals(p)` inside `__call__` builds this vector for the current parameter guess `p`, optionally multiplied by `sqrt(weights)` (consistent with how squaring a weighted residual should scale the statistic), and hands it to `scipy.optimize.least_squares`.

## Bounds handling

Astropy models carry `.min`/`.max` on each `Parameter`, but plain Levenberg-Marquardt (`method="lm"` in `scipy.optimize.least_squares`) does not support bounds. `_get_bounds(model)` collects the free (non-fixed) parameters' bounds into `(lower, upper)` arrays (`-inf`/`+inf` where unset); `__call__` then picks `method = "trf"` (Trust Region Reflective, which does support bounds) whenever any finite bound is present, and `method = "lm"` otherwise. This is why `fit_all.py` can pass either bounded or unbounded models to the same fitter instance without special-casing anything on its side.

## `__call__` flow

1. Validates `y >= 0` (Cash statistic is undefined for negative "observed counts") and records the free parameter names (raises if the model has none free).
2. `residuals(p)` (the closure passed to `least_squares`): writes `p` into `model.parameters`, evaluates `model(x)`, clips the result to a `1e-30` floor (`ln(0)` guard - this floor, not the parameter bounds, is what actually prevents the optimizer from producing an invalid model output during the search), computes the signed C-stat residual vector, and applies `weights` if given.
3. Runs `scipy.optimize.least_squares(residuals, p0, method=..., bounds=...)`, then writes the result back onto `model.parameters` - the same "mutate the model in place, also return it" convention Astropy's own fitters use.
4. If `calc_uncertainties=True` (constructor flag, default `False`) and the fit reports success, approximates the parameter covariance as `2 * inv(J^T J)` from the solver's final Jacobian - the factor of 2 comes from C-stat being *twice* the Poisson log-likelihood, so this mirrors how chi-square fitters derive covariance from `(J^T J)^-1` directly. Wrapped in a `try/except np.linalg.LinAlgError`, since a singular Jacobian (e.g. a parameter with no local sensitivity in-range) makes the matrix non-invertible; a `UserWarning` is emitted and `param_cov`/`param_errors` are left `None` rather than raising.
5. Populates `self.fit_info` (`nfev`, `cost`, `status`, `message`, `success`, `param_cov`, `param_errors`) and logs a short summary through `self.logger` (`log_fit_info`) - `fit_all.py` does not currently read `fit_info` after a fit, so today this is diagnostic-only (visible via `print`/logging output, not surfaced in the UI).

## Practical notes for extending this

- Unlike `fit_all.py`'s own `fit_unconstrained_then_bounded` / `fit_with_bounds_check` helpers (which run *multiple* fits and pick a result), `LevMarCstatFitter.__call__` always runs exactly one `least_squares` call per invocation - the "try unconstrained, then bounded" strategy lives entirely in `fit_all.py`, one layer up.
- `calc_uncertainties` defaults to `False`; `fit_all.py` constructs `LevMarCstatFitter()` with no arguments, so uncertainty/covariance computation is currently never enabled in the app, even though the machinery for it exists.
