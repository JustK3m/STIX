"""
NeuralNetwork.py — Neural network-based photon spectrum reconstruction (PyTorch).

Overview
--------
This module implements a supervised learning approach to solve the STIX X-ray
spectral inversion problem: given a vector of detector counts c (measured), it
reconstructs the underlying photon spectrum Φ(E) [ph cm⁻² s⁻¹ keV⁻¹] without
requiring an explicit forward-folding fit.

The inversion is learned entirely on simulated data, making the model fast at
inference time (single forward pass, < 1 ms) and robust to SRM uncertainties.

Method
------
Training data are generated on-the-fly from parametric power-law spectra:

    Φ(E) = A · E^(−α)       A ∈ [1e-3, 1e12],  α ∈ [1.5, 10.0]  (log-uniform / uniform,
                                                                  train()'s defaults)

Simulated counts are obtained by forward-folding through a perturbed SRM:

    c = SRM_i @ Φ     (+ optional Poisson noise)

where SRM_i is a physically realistic synthetic variant of the reference STIX
SRM (multiplicative log-normal noise + Gaussian smoothing along the true-energy
axis). Providing the SRM used for each sample at training time allows the
network to learn an SRM-conditioned inversion, i.e. it adapts its reconstruction
to the specific instrument response provided at inference.

Network Architecture
--------------------
Two-branch MLP with an SRM encoder:

    counts  (n_det)          ──────────────────────────────────────┐
                                                                    cat → MLP → Softplus → Φ (n_true)
    SRM flat (n_true*n_det)  → Linear(srm_dim) + LayerNorm + ReLU ┘

The SRM encoder compresses the 30 840-dimensional flattened SRM down to
srm_dim (default 64) before concatenation, reducing the first MLP layer
from ~8 M to ~500 k parameters and speeding up training by ×3.

MLP body: 3 hidden layers [256, 512, 256] with BatchNorm + ReLU + Dropout.
Output activation: Softplus — guarantees strictly positive flux values.

Training features:
  - Best-model checkpointing in memory (val_loss criterion).
  - Early stopping with configurable patience and min_delta threshold.
  - CosineAnnealingLR schedule with warm restart capability.
  - Separate LogStandardScaler (log1p + z-score) for counts, SRM, and photons.

STIX Dimensions
---------------
    n_det  =   30   detector energy channels (measured)
    n_true = 1028   true photon energy bins
    SRM    : (1028, 30)  loaded from a FITS file

Dependencies
------------
    numpy, astropy, torch, scipy

Quick Start
-----------
    from process.fitting.methods.NeuralNetwork import NeuralNetModel

    model = NeuralNetModel("data/stx_srm_2303197888.fits")
    model.train(n_samples=10_000, n_epochs=150, batch_size=128,
                patience=20, min_delta=1e-6)
    photons = model.predict(counts_vector)               # (n_true,)  array

    # With an explicit SRM at inference (e.g. observation-specific response):
    photons = model.predict(counts_vector, srm=my_srm)

    # Save / reload
    model.save("checkpoints/nn_model.pt")
    model = NeuralNetModel.load("checkpoints/nn_model.pt")

Integration in fit_all.py (idx == N)
-------------------------------------
    from process.fitting.methods.NeuralNetwork import NeuralNetModel

    model = NeuralNetModel(Fitting.rname)
    model.train()
    photon_spectrum = model.predict(counts[fit_mask])

Notes
-----
- The model is retrained from scratch each time .train() is called; call
  .save() / .load() to persist a trained model across sessions.
- srm_dim can be tuned: use 32 for faster training, 128 if the SRM varies
  significantly between observations.
- Setting patience=None disables early stopping (trains for all n_epochs).
- The Softplus output ensures Φ(E) > 0 everywhere, which is physically
  required for a photon flux spectrum.
"""
import os
import numpy as np
from astropy.io import fits

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
import time


# ══════════════════════════════════════════════════════════════════════════════
#  1. Loading and preparing the real SRM
# ══════════════════════════════════════════════════════════════════════════════

def load_srm(srm_path: str) -> dict:
    """
    Loads the STIX SRM from a FITS file and returns the useful arrays.

    Parameters
    ----------
    srm_path : str
        Path to the SRM FITS file.

    Returns
    -------
    dict with keys:
        'matrix'   : ndarray (n_true, n_det)  — normalised response matrix.
        'energ_lo' : ndarray (n_true,)         — lower bounds in true energy (keV).
        'energ_hi' : ndarray (n_true,)         — upper bounds in true energy (keV).
        'e_min'    : ndarray (n_det,)           — lower bounds of detector channels (keV).
        'e_max'    : ndarray (n_det,)           — upper bounds of detector channels (keV).
        'e_true'   : ndarray (n_true,)          — centres of the true-energy bins (keV).
    """
    with fits.open(srm_path) as hdul:
        hdu_m = hdul[1]
        hdu_e = hdul[2]

        matrix   = hdu_m.data['MATRIX'].astype(np.float32)   # (n_true, n_det)
        energ_lo = hdu_m.data['ENERG_LO'].astype(np.float32)
        energ_hi = hdu_m.data['ENERG_HI'].astype(np.float32)
        e_min    = hdu_e.data['E_MIN'].astype(np.float32)
        e_max    = hdu_e.data['E_MAX'].astype(np.float32)

    e_true = 0.5 * (energ_lo + energ_hi)

    return {
        'matrix':   matrix,
        'energ_lo': energ_lo,
        'energ_hi': energ_hi,
        'e_min':    e_min,
        'e_max':    e_max,
        'e_true':   e_true,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. Generating physically realistic synthetic SRMs
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_srm_batch(srm_ref: np.ndarray,
                                   n_batch: int,
                                   noise_level: float,
                                   smooth_sigma: float,
                                   rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised version of generate_synthetic_srm for an entire batch.

    Generates n_batch synthetic SRMs at once, with no Python loop:
    the multiplicative noise is drawn for the whole batch in one go, and the
    Gaussian smoothing is applied to the whole array (n_batch, n_true, n_det)
    in a single scipy call (axis=1), instead of one call per channel and per sample.

    Returns
    -------
    ndarray (n_batch, n_true, n_det)
    """
    n_true, n_det = srm_ref.shape

    log_noise = rng.normal(0.0, noise_level,
                            size=(n_batch, n_true, n_det)).astype(np.float32)
    srm_syn = srm_ref[np.newaxis, :, :] * np.exp(log_noise)   # (n_batch, n_true, n_det)

    # Gaussian smoothing along the true-energy axis (axis=1), for the whole
    # batch and all detector channels in a single call
    srm_syn = gaussian_filter1d(srm_syn, sigma=smooth_sigma, axis=1)

    # Restore the zero mask (broadcast over the batch)
    srm_syn[:, srm_ref == 0.0] = 0.0
    np.clip(srm_syn, 0.0, None, out=srm_syn)

    return srm_syn

# ══════════════════════════════════════════════════════════════════════════════
#  3. Generating the training data (power law)
# ══════════════════════════════════════════════════════════════════════════════

def generate_power_law_dataset(n_samples: int,
                               e_true: np.ndarray,
                               srm_ref: np.ndarray,
                               alpha_range: tuple = (1.5, 20.0),
                               amp_range:   tuple = (1e-3, 1e12),
                               noise_level_srm: float = 0.2,
                               poisson_noise: bool = True,
                               rng: np.random.Generator = None,
                               chunk_size: int = 2000) -> tuple:
    """
    Generates N simulated (counts, photons) pairs under a power law —
    chunk-vectorised version (same signature and same output as the
    original version, but with no per-sample Python loop).

    chunk_size : int
        Size of the batches processed at once. A speed/memory trade-off:
        larger = faster but more RAM used at once.
        Adjust to your machine (2000 is a good starting point for
        n_true=1028, n_det=30).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_true, n_det = srm_ref.shape
    n_srm_flat    = n_true * n_det

    counts_all  = np.zeros((n_samples, n_det),      dtype=np.float32)
    photons_all = np.zeros((n_samples, n_true),     dtype=np.float32)
    params_all  = np.zeros((n_samples, 2),          dtype=np.float32)
    srms_all    = np.zeros((n_samples, n_srm_flat), dtype=np.float32)

    log_amp_lo = np.log10(amp_range[0])
    log_amp_hi = np.log10(amp_range[1])

    for start in range(0, n_samples, chunk_size):
        end     = min(start + chunk_size, n_samples)
        n_batch = end - start

        # ── Vectorised draw of parameters (A, alpha) ──────────────────
        A     = 10 ** rng.uniform(log_amp_lo, log_amp_hi, size=n_batch)
        alpha = rng.uniform(alpha_range[0], alpha_range[1], size=n_batch)

        # phi_batch : (n_batch, n_true) — broadcast over e_true
        phi_batch = (A[:, None] * e_true[None, :] ** (-alpha[:, None])).astype(np.float32)

        # ── Vectorised synthetic SRMs ─────────────────────────────────
        srm_batch = _generate_synthetic_srm_batch(
            srm_ref, n_batch, noise_level_srm, smooth_sigma=1.0, rng=rng)  # (n_batch, n_true, n_det)

        # ── Projection counts = SRM.T @ phi, vectorised (einsum) ────────
        counts_batch = np.einsum('nij,ni->nj', srm_batch, phi_batch)

        if poisson_noise:
            counts_batch = rng.poisson(np.maximum(counts_batch, 0)).astype(np.float32)

        counts_all[start:end]  = counts_batch
        photons_all[start:end] = phi_batch
        params_all[start:end]  = np.stack([A, alpha], axis=1)
        srms_all[start:end]    = srm_batch.reshape(n_batch, n_srm_flat)

    return counts_all, photons_all, params_all, srms_all

# ══════════════════════════════════════════════════════════════════════════════
#  4. Neural network architecture
# ══════════════════════════════════════════════════════════════════════════════

class PhotonMLP(nn.Module):
    """
    Dual-branch neural network for photon spectrum reconstruction.

    Architecture:
        counts (n_det)         ──────────────────────────────────────────────┐
                                                                              cat → MLP → Softplus → Φ (n_true)
        SRM (n_true * n_det)   → Linear(srm_dim) → LayerNorm → ReLU ────────┘

    The two inputs are processed separately before fusion:
      - Counts branch : raw vector of n_det values (already normalised).
      - SRM branch    : linear encoder that compresses the flattened SRM
                        (n_true * n_det ≈ 30 840 dims) into an embedding of
                        srm_dim (default 64) before concatenation.
                        Without this encoder, the counts (n_det = 30) would
                        only represent 0.1% of the input and their gradient
                        signal would be drowned out by the SRM.

    MLP body : hidden_dims layers [BN → ReLU → Dropout].
    Output   : Softplus → guaranteed positivity.

    Parameters
    ----------
    n_det : int
        Number of measured detector channels.
    n_true : int
        Number of true-energy bins (output dimension).
    hidden_dims : list of int
        Size of the MLP hidden layers (default: [256, 512, 256]).
    dropout : float
        Dropout rate (default: 0.1).
    srm_dim : int
        Dimension of the SRM embedding produced by the encoder (default: 64).
    """

    def __init__(self,
                 n_det: int,
                 n_true: int,
                 hidden_dims: list = None,
                 dropout: float = 0.1,
                 srm_dim: int = 128):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 1028, 512]

        self.n_det = n_det

        # ── SRM branch: compresses n_true*n_det → srm_dim ─────────────────
        self.srm_encoder = nn.Sequential(
            nn.Linear(n_true * n_det, srm_dim),
            nn.LayerNorm(srm_dim),
            nn.ReLU(),
        )

        # ── Main MLP: fuses counts + SRM embedding ────────────────
        layers = []
        in_dim = n_det + srm_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h

        layers += [
            nn.Linear(in_dim, n_true),
            nn.Softplus(),
        ]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch_size, n_det + n_true*n_det)
            Concatenation of the normalised counts and the normalised flattened SRM.

        Returns
        -------
        Tensor (batch_size, n_true)
            Reconstructed photon spectrum (values > 0).
        """
        counts   = x[:, :self.n_det]
        srm_flat = x[:, self.n_det:]
        srm_emb  = self.srm_encoder(srm_flat)
        merged   = torch.cat([counts, srm_emb], dim=1)
        return self.mlp(merged)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Normalisation (log + standardisation)
# ══════════════════════════════════════════════════════════════════════════════

class LogStandardScaler:
    """
    log1p normalisation + standardisation (μ=0, σ=1).

    log1p is used instead of log to be robust to zeros
    (common in high-energy or low-signal spectra).

    Attributes
    ----------
    mean_ : ndarray
        Mean computed on the training data after log1p.
    std_ : ndarray
        Standard deviation computed on the training data after log1p.
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> 'LogStandardScaler':
        log_X      = np.log1p(np.maximum(X, 0))
        self.mean_ = log_X.mean(axis=0)
        self.std_  = log_X.std(axis=0) + 1e-8
        return self

    def fit_subsample(self, X: np.ndarray, subsample_frac: float = 0.05,
                       min_samples: int = 500,
                       rng: np.random.Generator = None) -> 'LogStandardScaler':
        """
        Fits on a random subsample of X rather than on all of X.

        Useful for huge arrays (e.g. flattened SRM) where the statistic
        (mean/std) converges well before every row has been seen.

        Parameters
        ----------
        subsample_frac : float
            Fraction of X to use for the fit (default: 5%).
        min_samples : int
            Minimum sample floor, to keep the estimate stable even when
            X is small or subsample_frac is very low
            (default: 500). If X has fewer rows than this floor,
            all of X is used.
        rng : np.random.Generator or None
        """
        n     = X.shape[0]
        n_sub = max(int(n * subsample_frac), min_samples)

        if n_sub >= n:
            return self.fit(X)

        generator = rng if rng is not None else np.random.default_rng()
        idx = generator.choice(n, size=n_sub, replace=False)
        return self.fit(X[idx])

    def transform(self, X: np.ndarray) -> np.ndarray:
        log_X = np.log1p(np.maximum(X, 0))
        return ((log_X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit + transform in a single pass over log1p(X) (instead of two)."""
        log_X      = np.log1p(np.maximum(X, 0))
        self.mean_ = log_X.mean(axis=0)
        self.std_  = log_X.std(axis=0) + 1e-8
        return ((log_X - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform_output(self, Y_norm: np.ndarray) -> np.ndarray:
        """Inverse for the output (photon spectrum)."""
        return np.expm1(Y_norm * self.std_ + self.mean_)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Main class — training and inference
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetModel:
    """
    Main interface for NN-based photon spectrum reconstruction.

    Workflow:
        model = NeuralNetModel("data/stx_srm_2303197888.fits")
        model.train(n_samples=10000, n_epochs=150)
        photons = model.predict(counts_vector)

    Parameters
    ----------
    srm_path : str
        Path to the reference SRM FITS file.
    hidden_dims : list of int, optional
        Architecture of the MLP hidden layers.
    dropout : float, optional
        Dropout rate (default: 0.1).
    srm_dim : int, optional
        Dimension of the SRM embedding produced by the encoder (default: 64).
    device : str or None, optional
        'cpu', 'cuda', or None for automatic detection.

    Attributes
    ----------
    srm_data : dict
        Loaded SRM data (matrix, e_true, e_min, e_max, ...).
    net : PhotonMLP
        PyTorch neural network.
    scaler_X : LogStandardScaler
        Normaliser for the counts (input).
    scaler_S : LogStandardScaler
        Normaliser for the flattened SRM.
    scaler_Y : LogStandardScaler
        Normaliser for the photons (output).
    is_trained : bool
        True if the model has been trained.
    train_history : list of float
        History of the MSE loss per epoch.
    """

    def __init__(self,
                 srm_path: str,
                 hidden_dims: list = None,
                 dropout: float = 0.1,
                 srm_dim: int = 64,
                 device: str = None):

        self.srm_path    = srm_path
        self.srm_data    = load_srm(srm_path)
        self.hidden_dims = hidden_dims or [256, 512, 256]
        self.dropout     = dropout
        self.srm_dim     = srm_dim
        self.device      = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        n_det  = self.srm_data['matrix'].shape[1]
        n_true = self.srm_data['matrix'].shape[0]

        # Separate scalers: counts, flattened SRM, photons
        self.scaler_X     = LogStandardScaler()   # counts (n_det,)
        self.scaler_S     = LogStandardScaler()   # flattened SRM (n_true*n_det,)
        self.scaler_Y     = LogStandardScaler()   # photons (n_true,)

        self.net = PhotonMLP(n_det, n_true, self.hidden_dims, self.dropout, self.srm_dim)
        self.net.to(self.device)

        self.is_trained   = False
        self.train_history = []

        print(f"[NeuralNetModel] device={self.device} | "
              f"n_det={n_det} | n_true={n_true} | srm_dim={srm_dim} | "
              f"counts branch: {n_det} | SRM branch: {n_true * n_det} → {srm_dim} | "
              f"merged: {n_det + srm_dim}")

    def train(self,
              n_samples:       int   = 10000,
              n_epochs:        int   = 150,
              batch_size:      int   = 128,
              lr:              float = 1e-3,
              weight_decay:    float = 1e-5,
              alpha_range:     tuple = (1.5, 10.0),
              amp_range:       tuple = (1e-3, 1e12),
              noise_level_srm: float = 0.2,
              poisson_noise:   bool  = True,
              val_frac:        float = 0.2,
              test_frac:       float = 0.1,
              seed:            int   = 42,
              verbose:         bool  = True,
              patience:        int   = 20,
              min_delta:       float = 1e-6) -> None:
        """
        Generates the simulated data and trains the network.

        Automatically saves the weights of the best model (minimal
        val_loss) in memory, and stops training early if the val_loss
        does not improve by more than `min_delta` for `patience`
        consecutive epochs (early stopping).

        Parameters
        ----------
        n_samples : int
            Number of simulated spectra for training (default: 10,000).
        n_epochs : int
            Maximum number of epochs (default: 150).
        batch_size : int
            Mini-batch size (default: 128).
        lr : float
            Initial Adam learning rate (default: 1e-3).
        weight_decay : float
            L2 weight regularisation (default: 1e-5).
        alpha_range : tuple
            Range of the spectral index α.
        amp_range : tuple
            Log-uniform range of the amplitude A.
        noise_level_srm : float
            Relative noise on the synthetic SRMs (default: 20%).
        poisson_noise : bool
            Whether to add Poisson noise to the simulated counts.
        val_frac : float
            Validation fraction (default: 20%).
        test_frac : float
            Test fraction (default: 10%).
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Whether to display progress.
        patience : int
            Number of epochs without improvement before early stopping
            (default: 20). Set to None to disable early stopping.
        min_delta : float
            Minimum val_loss improvement considered as progress
            (default: 1e-6). Below this threshold, the epoch is counted
            as stagnation.
        """
        import copy

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        srm_ref = self.srm_data['matrix']
        e_true  = self.srm_data['e_true']

        start = time.time()
        # ── Data generation ─────────────────────────────────────────
        if verbose:
            print(f"[Train] Generating {n_samples} simulated spectra...")

        counts_all, photons_all, params_all, srms_all = generate_power_law_dataset(
            n_samples       = n_samples,
            e_true          = e_true,
            srm_ref         = srm_ref,
            alpha_range     = alpha_range,
            amp_range       = amp_range,
            noise_level_srm = noise_level_srm,
            poisson_noise   = poisson_noise,
            rng             = rng,
        )

        # ── Train / validation / test split (70 / 20 / 10) ───────────────────
        if verbose:
            print(f"[Train] Splitting train / validation / test")

        n_test  = int(n_samples * test_frac)
        n_val   = int(n_samples * val_frac)
        n_train = n_samples - n_val - n_test

        X_train = counts_all[:n_train]
        X_val   = counts_all[n_train:n_train + n_val]
        X_test  = counts_all[n_train + n_val:]

        Y_train = photons_all[:n_train]
        Y_val   = photons_all[n_train:n_train + n_val]
        Y_test  = photons_all[n_train + n_val:]

        S_train = srms_all[:n_train]
        S_val   = srms_all[n_train:n_train + n_val]
        S_test  = srms_all[n_train + n_val:]

        P_train = params_all[:n_train]
        P_val = params_all[n_train:n_train + n_val]
        P_test = params_all[n_train + n_val:]

        # ── Normalisation ──────────────────────────────────────────────────
        if verbose:
            print(f"[Train] Normalising...")
        X_train_n = self.scaler_X.fit_transform(X_train)
        X_val_n = self.scaler_X.transform(X_val)
        self.scaler_S.fit_subsample(S_train, subsample_frac=0.05, rng=rng)
        S_train_n = self.scaler_S.transform(S_train)
        S_val_n = self.scaler_S.transform(S_val)
        Y_train_n = self.scaler_Y.fit_transform(Y_train)
        Y_val_n = self.scaler_Y.transform(Y_val)

        # Concatenate counts + flattened SRM → network input
        XS_train_n = np.concatenate([X_train_n, S_train_n], axis=1)
        XS_val_n   = np.concatenate([X_val_n,   S_val_n],   axis=1)

        # ── DataLoaders ────────────────────────────────────────────────────
        if verbose:
            print(f"[Train] Creating data loaders...")
        def to_loader(XS, Y, shuffle):
            ds = TensorDataset(
                torch.tensor(XS, dtype=torch.float32),
                torch.tensor(Y,  dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        loader_train = to_loader(XS_train_n, Y_train_n, shuffle=True)
        loader_val   = to_loader(XS_val_n,   Y_val_n,   shuffle=False)

        # ── Optimizer + scheduler ─────────────────────────────────────────
        if verbose:
            print(f"[Train] Optimizer and scheduler...")
        optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        loss_fn = nn.MSELoss()

        # ── Early stopping state & best checkpoint ──────────────────────────
        best_val_loss   = float('inf')
        best_epoch      = 0
        best_state_dict = None          # weights of the best model (in memory)
        stagnation      = 0             # counter of epochs without progress

        elapsed = time.time() - start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"[Train] Data generation. Execution time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


        # ── Training loop ──────────────────────────────────────────
        self.train_history = []
        self.val_history   = []
        self.lr_history    = []
        if verbose:
            es_info = (f"patience={patience}, min_delta={min_delta:.0e}"
                       if patience else "early stopping disabled")
            print(f"[Train] Starting — {n_epochs} max epochs | "
                  f"device={self.device} | lr={lr} | {es_info}")

        for epoch in range(1, n_epochs + 1):
            # ── Training pass ───────────────────────────────────────
            self.net.train()
            train_loss = 0.0
            for xb, yb in loader_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= n_train

            # ── Validation pass ────────────────────────────────────────
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += loss_fn(self.net(xb), yb).item() * len(xb)
            val_loss /= n_val

            current_lr = optimizer.param_groups[0]['lr']
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            self.lr_history.append(current_lr)
            scheduler.step()

            # ── Best model checkpoint ──────────────────────────────────────
            improved = val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss   = val_loss
                best_epoch      = epoch
                best_state_dict = copy.deepcopy(self.net.state_dict())
                stagnation      = 0
                tag = ' ★'          # visual improvement marker
            else:
                stagnation += 1
                tag = f' ({stagnation}/{patience})' if patience else ''

            if verbose and (epoch % 10 == 0 or epoch == 1 or improved):
                print(f"  Epoch {epoch:4d}/{n_epochs} | "
                      f"train={train_loss:.4e} | val={val_loss:.4e}{tag}")

            # ── Early stopping ─────────────────────────────────────────────
            if patience and stagnation >= patience:
                if verbose:
                    print(f"\n[Train] Early stopping triggered at epoch {epoch} "
                          f"(no improvement for {patience} epochs).")
                break

        # ── Restoring the best state ──────────────────────────────────
        if best_state_dict is not None:
            self.net.load_state_dict(best_state_dict)
            if verbose:
                print(f"[Train] Best model restored → "
                      f"epoch {best_epoch} | val_loss={best_val_loss:.4e}")

        self.best_epoch    = best_epoch
        self.best_val_loss = best_val_loss
        self.is_trained    = True
        self.X_test        = X_test
        self.Y_test        = Y_test
        self.S_test        = S_test
        self.P_test        = P_test
        if verbose:
            print("[Train] Training complete.")

    def predict(self, counts: np.ndarray,
                srm: np.ndarray = None) -> np.ndarray:
        """
        Reconstructs the photon spectrum from a counts vector and its
        associated SRM.

        Parameters
        ----------
        counts : ndarray (n_det,) or (N, n_det)
            Counts observed by the detector.
        srm : ndarray (n_true, n_det) or (N, n_true, n_det), optional
            Instrument response matrix used for the acquisition.
            If None, uses the reference SRM loaded from the FITS file.

        Returns
        -------
        ndarray (n_true,) or (N, n_true)
            Reconstructed photon spectrum [photons cm-2 s-1 keV-1].

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError(
                "The model is not trained. Call .train() first.")

        single = counts.ndim == 1
        if single:
            counts = counts[np.newaxis, :]   # (1, n_det)

        N = counts.shape[0]

        # ── Preparing the SRM ─────────────────────────────────────────
        if srm is None:
            # Use the reference SRM for all samples
            srm_flat = self.srm_data['matrix'].ravel()[np.newaxis, :]  # (1, n_true*n_det)
            srm_flat = np.repeat(srm_flat, N, axis=0)                  # (N, n_true*n_det)
        else:
            if srm.ndim == 2:
                # Single SRM (n_true, n_det) → replicate for each sample
                srm_flat = srm.ravel()[np.newaxis, :]
                srm_flat = np.repeat(srm_flat, N, axis=0)
            else:
                # Batch of SRMs (N, n_true, n_det)
                srm_flat = srm.reshape(N, -1)

        srm_flat = srm_flat.astype(np.float32)

        # ── Normalisation ─────────────────────────────────────────────────
        X_n  = self.scaler_X.transform(counts)
        S_n  = self.scaler_S.transform(srm_flat)
        XS_n = np.concatenate([X_n, S_n], axis=1)

        X_t = torch.tensor(XS_n, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            Y_n = self.net(X_t).cpu().numpy()

        # Denormalisation
        photons = self.scaler_Y.inverse_transform_output(Y_n)

        return photons[0] if single else photons

    def save(self, path: str, srm_path: str) -> None:
        """
        Saves the trained model (weights + scalers + history + test set)
        to disk.

        Parameters
        ----------
        path : str
            Path of the save file (.pt recommended).
        srm_path : str
            Path of the reference SRM FITS file, stored in the checkpoint
            so .load() can reload the matching SRM.
        """
        torch.save({
            'state_dict': self.net.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_std': self.scaler_X.std_,
            'scaler_S_mean': self.scaler_S.mean_,
            'scaler_S_std': self.scaler_S.std_,
            'scaler_Y_mean': self.scaler_Y.mean_,
            'scaler_Y_std': self.scaler_Y.std_,
            'srm_path': srm_path,
            'n_det': self.srm_data['matrix'].shape[1],
            'n_true': self.srm_data['matrix'].shape[0],
            'best_epoch': getattr(self, 'best_epoch', None),
            'best_val_loss': getattr(self, 'best_val_loss', None),
            'srm_dim': getattr(self, 'srm_dim', 64),
            # ── Additions: training history ──────────────────────────
            'train_history': getattr(self, 'train_history', []),
            'val_history': getattr(self, 'val_history', []),
            'lr_history': getattr(self, 'lr_history', []),
            # ── Additions: test set (for plots without retraining) ──────
            'X_test': getattr(self, 'X_test', None),
            'Y_test': getattr(self, 'Y_test', None),
            'S_test': getattr(self, 'S_test', None),
            'P_test': getattr(self, 'P_test', None),
        }, path)
        print(f"[NeuralNetModel] Model saved → {path}")


    @classmethod
    def load(cls, path: str, device: str = None) -> 'NeuralNetModel':
        """
        Loads a previously saved model, along with its training history
        and test set (if present in the checkpoint).

        Parameters
        ----------
        path : str
            Path of the .pt file generated by .save().
        device : str or None
            Force a specific device ('cpu', 'cuda').

        Returns
        -------
        NeuralNetModel
            Instance ready for .predict() and for diagnostic plots.
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        instance = cls.__new__(cls)
        instance.srm_path = ckpt['srm_path']
        instance.srm_data = load_srm(ckpt['srm_path'])
        instance.hidden_dims = ckpt['hidden_dims']
        instance.dropout = ckpt['dropout']
        instance.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        n_det, n_true = ckpt['n_det'], ckpt['n_true']
        instance.srm_dim = ckpt.get('srm_dim', 64)
        instance.net = PhotonMLP(n_det, n_true, ckpt['hidden_dims'],
                                 ckpt['dropout'], instance.srm_dim)
        instance.net.load_state_dict(ckpt['state_dict'])
        instance.net.to(instance.device)

        instance.scaler_X = LogStandardScaler()
        instance.scaler_X.mean_ = ckpt['scaler_X_mean']
        instance.scaler_X.std_ = ckpt['scaler_X_std']

        instance.scaler_S = LogStandardScaler()
        instance.scaler_S.mean_ = ckpt['scaler_S_mean']
        instance.scaler_S.std_ = ckpt['scaler_S_std']

        instance.scaler_Y = LogStandardScaler()
        instance.scaler_Y.mean_ = ckpt['scaler_Y_mean']
        instance.scaler_Y.std_ = ckpt['scaler_Y_std']

        instance.is_trained = True
        instance.best_epoch = ckpt.get('best_epoch', None)
        instance.best_val_loss = ckpt.get('best_val_loss', None)

        # ── Restoring: training history ─────────────────────────────────────
        instance.train_history = ckpt.get('train_history', [])
        instance.val_history = ckpt.get('val_history', [])
        instance.lr_history = ckpt.get('lr_history', [])

        # ── Restoring: test set ────────────────────────────────────────
        instance.X_test = ckpt.get('X_test', None)
        instance.Y_test = ckpt.get('Y_test', None)
        instance.S_test = ckpt.get('S_test', None)
        instance.P_test = ckpt.get('P_test', None)

        print(f"[NeuralNetModel] Model loaded from {path} | device={instance.device}")
        return instance

if __name__ == "__main__":
    nb_samples = 350000
    NeuralNetModel = NeuralNetModel(os.path.join('..', '..', '..', '..', 'data', 'stx_srm_2303197888.fits'))
    NeuralNetModel.train(n_samples=nb_samples)
    NeuralNetModel.save(os.path.join('..', '..', '..', '..', 'data', 'nn_powerlaw.pt'), "../../../../data/stx_srm_2303197888.fits")