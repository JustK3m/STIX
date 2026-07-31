# fitting/methods/ - what gets fitted

Two independent approaches to reconstructing a photon spectrum from observed counts: physical forward-folded models (`ForwardFolded.py`) and a pre-trained neural network (`NeuralNetwork.py`). `process/fit_all.py`'s "Set method" control picks between them (see [../../fit_all.md](../../fit_all.md)).

## `ForwardFolded.py` - physical spectral models

Every model here is an `astropy.modeling.FittableModel` whose `evaluate()` does the same three steps: build a photon-flux function `Phi(E)` from the current parameter values, integrate it over each true-energy bin of the response matrix, then fold the result through the matrix and divide by exposure to get a modelled count rate.

**`integrate_flux(e1, e2, model_func, n_points=10)`** is the shared building block: it evaluates `model_func` at `n_points` (default 10) linearly-spaced energies between `e1` and `e2` and returns the trapezoidal-rule average (`np.trapezoid(fluxes, energies) / (e2 - e1)`). Every model class calls this once per true-energy bin inside `evaluate()`, so a fit with many true-energy bins does many small trapezoidal integrations per residual evaluation - this is the main cost driver of a forward-folding fit.

Each class takes `e_low_true`, `e_high_true`, `matrix`, `exposure` (and sometimes `E_pivot`/`E_cut`) as constructor arguments (not `Parameter`s - these describe the instrument/observation, not the source, and are never fitted), plus its physical `Parameter`s:

| Class | Photon flux Phi(E) | Free parameters | Notes |
|---|---|---|---|
| `PowerLaw` | `A * (E/E_pivot)^-alpha` | amplitude, alpha | Used by model listbox entry "PowerLaw1D". `E_pivot` fixed at construction. |
| `BrokenPowerLaw` | `A*(E/E_break)^-alpha_1` below break, `A*(E/E_break)^-alpha_2` above | amplitude, E_break, alpha_1, alpha_2 | |
| `VTH` | `(A_ff*EM)/(E*sqrt(T)) * exp(-E/T)`, `A_ff = 1.07e-42 * 1.2` | EM, T | Optically-thin thermal bremsstrahlung. `T` is clamped to a `1e-3` keV floor inside `evaluate()` to avoid division by zero, independent of the parameter's own `(0.1, 50.0)` bound. |
| `ExpPowerLaw` | `p0 * (E/p2)^p1 * exp(e3 - E/e4)` | p0, p1, p2, e3, e4 | Empirical model, listed as "Single Power Law Times an Exponential". |
| `VTHPlusPowerLaw` | `Phi_VTH(E) + Phi_PowerLaw(E)` | EM, T, amplitude, alpha | Additive combination of `VTH` and `PowerLaw`'s formulas, re-implemented inline rather than composing the other two classes. |
| `PowerLawCutoffFix` | `A * E^-alpha` for `E >= E_cut`, else 0 | amplitude, alpha | `E_cut` is a plain instance attribute set at construction (default 10 keV), not a `Parameter` - it can be changed via `Set Function` in `fit_all.py` but is never fitted by this class itself. |

There is **no `PowerLawCutoffFree` class** - the free-cutoff behavior (fitting `E_cut` itself) is implemented in `fit_all.py` by wrapping `PowerLawCutoffFix` in a `scipy.optimize.minimize_scalar` search over the mutable `E_cut` attribute (see the idx 6 / idx 8 branches in [../../fit_all.md](../../fit_all.md)). Likewise, "V_TH x PowerLawCutoffFix/Free" in the model listbox are not combined model classes; `fit_all.py` fits `PowerLawCutoffFix` and `VTH` separately over complementary energy sub-ranges and plots both curves.

All `evaluate()` implementations except `PowerLaw`'s clip only implicitly (via `np.where`/`max()` guards on their own inputs); `PowerLaw.evaluate()` is the one that explicitly clamps its own output to `[1e-30, inf)` and replaces non-finite values, because it is the model exercised by `fit_unconstrained_then_bounded`'s *unconstrained* first pass, which is the most likely to wander into numerically unstable parameter regions before bounds are applied.

## `NeuralNetwork.py` - CNN-style reconstruction

`NeuralNetModel` reconstructs the photon spectrum `Phi(E)` directly from a vector of detector counts, without fitting any physical model or explicit forward folding at inference time. It is trained entirely offline on simulated data; `fit_all.py` only ever calls `.predict()` on a checkpoint loaded from disk (`data/nn_powerlaw_150k.pt`, path hardcoded - see [../../fit_all.md](../../fit_all.md)).

### Architecture

A two-branch MLP (`PhotonMLP`):

```
counts  (n_det)          ─────────────────────────────────────┐
                                                                cat -> MLP -> Softplus -> Phi (n_true)
SRM flat (n_true*n_det)  -> Linear(srm_dim) + LayerNorm + ReLU ┘
```

The SRM (STIX: 1028 true-energy bins x 30 detector channels, ~30840 values when flattened) is compressed by a small linear encoder to `srm_dim` (default 64) before being concatenated with the raw counts - without this, the 30-value counts vector would be numerically drowned out by the much larger flattened SRM in the first MLP layer. The MLP body is 3 hidden layers (default `[256, 512, 256]`, BatchNorm + ReLU + Dropout each) ending in a `Softplus` output, which guarantees the reconstructed flux is strictly positive everywhere - a physical requirement for a photon flux that a plain linear output would not enforce.

### Training data

`generate_power_law_dataset()` never touches real observations. For each simulated sample it draws a random power-law `Phi(E) = A * E^-alpha` (log-uniform `A`, uniform `alpha`), perturbs the reference SRM with multiplicative log-normal noise plus Gaussian smoothing along the true-energy axis (`_generate_synthetic_srm_batch`, vectorized over a whole batch at once), forward-folds the photon spectrum through that *synthetic* SRM to get simulated counts, and optionally adds Poisson noise. Passing a different, perturbed SRM with every sample (rather than always the reference SRM) is what lets the trained network generalize to SRM variation at inference time instead of memorizing one fixed instrument response - `predict()` accepts an explicit `srm=` argument for exactly this reason, falling back to the reference SRM loaded at construction time if none is given.

### `LogStandardScaler`

A `log1p` (not plain `log`, to stay finite at zero) plus zero-mean/unit-variance normalization, fit independently for counts, flattened SRM, and photon flux (three separate `LogStandardScaler` instances on `NeuralNetModel`). `fit_subsample()` fits the SRM scaler on a random 5% subsample (with a 500-sample floor) rather than the full flattened array, since the mean/std of ~30k-dimensional data converges well before every row is seen - this is purely a training-time speed optimization.

### `train()`

Generates `n_samples` simulated pairs (default 10,000), splits 70/20/10 train/val/test, fits the three scalers on the training split only, then runs a standard Adam + `CosineAnnealingLR` training loop with MSE loss. Two things worth knowing when touching this:

- **Best-checkpoint restore**: the network's weights after the *last* epoch are not what ends up in `self.net` - `train()` deep-copies the state dict whenever validation loss improves and restores that best state at the end, so early stopping (`patience`/`min_delta`, default `patience=20`) does not lose the best result to a later, worse epoch.
- **Retraining is from scratch every call**: there is no incremental/resume training; `.save()`/`.load()` exist specifically so a trained model can be reused across process runs instead of retraining on every app launch. The shipped checkpoint (`code/data/nn_powerlaw_150k.pt`) was produced by running `NeuralNetwork.py` directly (see the `if __name__ == "__main__":` block at the bottom of the file, which trains on 350,000 samples against `data/stx_srm_2303197888.fits`).

### `predict()`

Accepts a single counts vector `(n_det,)` or a batch `(N, n_det)`, and an optional `srm` (a single `(n_true, n_det)` matrix broadcast to every sample, or a per-sample batch `(N, n_true, n_det)`); normalizes both through the fitted scalers, runs a forward pass in `eval()` mode (`torch.no_grad()`), and inverse-transforms the output back into physical flux units via `scaler_Y.inverse_transform_output`. Raises `RuntimeError` if called before `train()` or `load()`.

### Known gaps

- `save()`/`load()` persist model weights, scaler parameters, training/validation history, and the held-out test split, but **not** the random seed or the exact `alpha_range`/`amp_range` used to generate the training data - reproducing a checkpoint's training conditions from the `.pt` file alone is not fully possible.
- There is no test coverage for this module (unlike `ForwardFolded.py`, which has model-level tests).
