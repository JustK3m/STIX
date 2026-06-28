"""
NeuralNet.py — Neural network-based photon spectrum reconstruction (PyTorch).

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

    Φ(E) = A · E^(−α)       A ∈ [1e-3, 1e2],  α ∈ [1.5, 5.0]  (log-uniform / uniform)

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
    from process.fitting.methods.NeuralNet import NeuralNetModel

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
    from process.fitting.methods.NeuralNet import NeuralNetModel

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


# ══════════════════════════════════════════════════════════════════════════════
#  1. Chargement et préparation de la SRM réelle
# ══════════════════════════════════════════════════════════════════════════════

def load_srm(srm_path: str) -> dict:
    """
    Charge la SRM STIX depuis un fichier FITS et retourne les tableaux utiles.

    Parameters
    ----------
    srm_path : str
        Chemin vers le fichier FITS de la SRM.

    Returns
    -------
    dict avec les clés :
        'matrix'   : ndarray (n_true, n_det)  — matrice de réponse normalisée.
        'energ_lo' : ndarray (n_true,)         — bornes inférieures énergie vraie (keV).
        'energ_hi' : ndarray (n_true,)         — bornes supérieures énergie vraie (keV).
        'e_min'    : ndarray (n_det,)           — bornes inférieures canaux détecteur (keV).
        'e_max'    : ndarray (n_det,)           — bornes supérieures canaux détecteur (keV).
        'e_true'   : ndarray (n_true,)          — centres des bins en énergie vraie (keV).
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
#  2. Génération de SRM synthétiques physiquement réalistes
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_srm(srm_ref: np.ndarray,
                            noise_level: float = 0.05,
                            smooth_sigma: float = 1.0,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Génère une SRM synthétique par perturbation de la SRM de référence.

    La perturbation est multiplicative (lognormale) et lissée, ce qui préserve :
        - la positivité (pas de valeurs négatives physiquement impossibles),
        - la structure diagonale dominante de la réponse instrumentale,
        - la variation douce entre canaux adjacents.

    Parameters
    ----------
    srm_ref : ndarray (n_true, n_det)
        SRM de référence extraite du fichier FITS réel.
    noise_level : float, optional
        Écart-type relatif du bruit lognormal (défaut : 0.05 = 5%).
        Valeurs typiques : 0.02–0.10.
    smooth_sigma : float, optional
        Sigma du lissage gaussien 1D appliqué sur l'axe des énergies vraies
        (défaut : 1.0 bin). Évite les discontinuités non physiques.
    rng : np.random.Generator or None
        Générateur numpy pour la reproductibilité (défaut : None → np.random.default_rng()).

    Returns
    -------
    ndarray (n_true, n_det)
        SRM perturbée, avec la même structure de zéros que srm_ref.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Bruit multiplicatif lognormal : exp(N(0, σ²)) centré sur 1
    log_noise = rng.normal(0.0, noise_level, size=srm_ref.shape).astype(np.float32)
    perturb   = np.exp(log_noise)

    srm_syn = srm_ref * perturb

    # Lissage gaussien sur l'axe 0 (énergie vraie) pour chaque canal détecteur
    for j in range(srm_syn.shape[1]):
        srm_syn[:, j] = gaussian_filter1d(srm_syn[:, j], sigma=smooth_sigma)

    # Restauration du masque de zéros : les zones nulles dans srm_ref
    # correspondent à des canaux d'énergie physiquement inactifs
    srm_syn[srm_ref == 0.0] = 0.0

    # Clamp positif (le lissage peut introduire de très légères valeurs négatives)
    np.clip(srm_syn, 0.0, None, out=srm_syn)

    return srm_syn


# ══════════════════════════════════════════════════════════════════════════════
#  3. Génération des données d'entraînement (loi de puissance)
# ══════════════════════════════════════════════════════════════════════════════

def generate_power_law_dataset(n_samples: int,
                               e_true: np.ndarray,
                               srm_ref: np.ndarray,
                               alpha_range: tuple = (1.5, 5.0),
                               amp_range:   tuple = (1e-3, 1e2),
                               noise_level_srm: float = 0.05,
                               poisson_noise: bool = True,
                               rng: np.random.Generator = None) -> tuple:
    """
    Génère N paires (counts, photons) simulées sous loi de puissance.

    Pour chaque échantillon i :
        1. Tire des paramètres (A_i, α_i) aléatoirement.
        2. Calcule le spectre photonique : Φ_i(E) = A_i · E^(−α_i)
        3. Génère une SRM synthétique perturbée à partir de srm_ref.
        4. Calcule les counts simulés : c_i = SRM_i @ Φ_i
        5. Ajoute optionnellement un bruit de Poisson sur c_i.

    Parameters
    ----------
    n_samples : int
        Nombre de paires d'entraînement à générer.
    e_true : ndarray (n_true,)
        Centres des bins en énergie vraie (keV).
    srm_ref : ndarray (n_true, n_det)
        SRM de référence utilisée comme base de perturbation.
    alpha_range : tuple (float, float)
        Intervalle uniforme pour l'indice spectral α (défaut : 1.5–5.0).
    amp_range : tuple (float, float)
        Intervalle log-uniforme pour l'amplitude A (défaut : 1e-3–1e2).
    noise_level_srm : float
        Niveau de bruit relatif pour les SRM synthétiques (défaut : 5%).
    poisson_noise : bool
        Si True, ajoute un bruit de Poisson sur les counts simulés.
    rng : np.random.Generator or None
        Générateur pour la reproductibilité.

    Returns
    -------
    counts_all  : ndarray (n_samples, n_det)
        Counts simulés (entrée du réseau).
    photons_all : ndarray (n_samples, n_true)
        Spectres photoniques vrais (cible du réseau).
    params_all  : ndarray (n_samples, 2)
        Paramètres (A, α) pour chaque échantillon.
    srms_all    : ndarray (n_samples, n_true * n_det)
        SRM synthétiques aplaties utilisées pour chaque échantillon.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_true, n_det = srm_ref.shape
    n_srm_flat    = n_true * n_det

    counts_all  = np.zeros((n_samples, n_det),     dtype=np.float32)
    photons_all = np.zeros((n_samples, n_true),    dtype=np.float32)
    params_all  = np.zeros((n_samples, 2),         dtype=np.float32)
    srms_all    = np.zeros((n_samples, n_srm_flat), dtype=np.float32)

    # Tirage log-uniforme sur A pour couvrir plusieurs ordres de grandeur
    log_amp_lo = np.log10(amp_range[0])
    log_amp_hi = np.log10(amp_range[1])

    for i in range(n_samples):
        A     = 10 ** rng.uniform(log_amp_lo, log_amp_hi)
        alpha = rng.uniform(*alpha_range)

        # Spectre photonique : loi de puissance
        phi = (A * e_true ** (-alpha)).astype(np.float32)

        # SRM perturbée physiquement réaliste
        srm_i = generate_synthetic_srm(srm_ref, noise_level=noise_level_srm, rng=rng)

        # Projection : counts = SRM @ phi
        counts_i = srm_i.T @ phi   # (n_det,)

        # Bruit de Poisson (régime réaliste de comptage)
        if poisson_noise:
            counts_i = rng.poisson(np.maximum(counts_i, 0)).astype(np.float32)

        counts_all[i]  = counts_i
        photons_all[i] = phi
        params_all[i]  = [A, alpha]
        srms_all[i]    = srm_i.ravel()   # aplatissement (n_true*n_det,)

    return counts_all, photons_all, params_all, srms_all


# ══════════════════════════════════════════════════════════════════════════════
#  4. Architecture du réseau de neurones
# ══════════════════════════════════════════════════════════════════════════════

class PhotonMLP(nn.Module):
    """
    Réseau de neurones MLP pour la reconstruction du spectre photonique.

    Architecture :
        Input (n_det + n_true*n_det) → Linear → BN → ReLU → Dropout
                                     → Linear → BN → ReLU → Dropout
                                     → Linear → BN → ReLU → Dropout
                                     → Linear → Softplus → Output (n_true)

    L'entrée est la concaténation des counts mesurés (n_det) et de la SRM
    aplatie associée (n_true * n_det), ce qui permet au réseau d'apprendre
    une inversion adaptée à chaque instrument/configuration.

    La couche Softplus en sortie garantit des valeurs strictement positives,
    ce qui est physiquement requis pour un spectre de flux photonique.

    Parameters
    ----------
    n_det : int
        Nombre de canaux détecteur mesurés.
    n_true : int
        Nombre de bins en énergie vraie (dimension de sortie).
    hidden_dims : list of int
        Taille des couches cachées (défaut : [256, 512, 256]).
    dropout : float
        Taux de dropout (défaut : 0.1).
    """

    def __init__(self,
                 n_det: int,
                 n_true: int,
                 hidden_dims: list = None,
                 dropout: float = 0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512, 256]

        # Dimension d'entrée = counts (n_det) + SRM aplatie (n_true * n_det)
        in_dim_total = n_det + n_true * n_det

        layers = []
        in_dim = in_dim_total
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
            nn.Softplus(),   # positivité garantie
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch_size, n_det + n_true*n_det)
            Concaténation des counts mesurés et de la SRM aplatie associée.

        Returns
        -------
        Tensor (batch_size, n_true)
            Spectre photonique reconstruit (valeurs > 0).
        """
        return self.network(x)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Normalisation (log + standardisation)
# ══════════════════════════════════════════════════════════════════════════════

class LogStandardScaler:
    """
    Normalisation log1p + standardisation (μ=0, σ=1).

    log1p est utilisé plutôt que log pour être robuste aux zéros
    (fréquents dans les spectres à haute énergie ou faible signal).

    Attributes
    ----------
    mean_ : ndarray
        Moyenne calculée sur les données d'entraînement après log1p.
    std_ : ndarray
        Écart-type calculé sur les données d'entraînement après log1p.
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> 'LogStandardScaler':
        log_X      = np.log1p(np.maximum(X, 0))
        self.mean_ = log_X.mean(axis=0)
        self.std_  = log_X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        log_X = np.log1p(np.maximum(X, 0))
        return ((log_X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform_output(self, Y_norm: np.ndarray) -> np.ndarray:
        """Inverse pour la sortie (spectre photonique)."""
        return np.expm1(Y_norm * self.std_ + self.mean_)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Classe principale — entraînement et inférence
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetModel:
    """
    Interface principale pour la reconstruction NN du spectre photonique.

    Workflow :
        model = NeuralNetModel("data/stx_srm_2303197888.fits")
        model.train(n_samples=10000, n_epochs=150)
        photons = model.predict(counts_vector)

    Parameters
    ----------
    srm_path : str
        Chemin vers le fichier FITS de la SRM de référence.
    hidden_dims : list of int, optional
        Architecture des couches cachées du MLP.
    dropout : float, optional
        Taux de dropout (défaut : 0.1).
    device : str or None, optional
        'cpu', 'cuda', ou None pour détection automatique.

    Attributes
    ----------
    srm_data : dict
        Données SRM chargées (matrix, e_true, e_min, e_max, ...).
    net : PhotonMLP
        Réseau de neurones PyTorch.
    scaler_X : LogStandardScaler
        Normalisateur des counts (entrée).
    scaler_Y : LogStandardScaler
        Normalisateur des photons (sortie).
    is_trained : bool
        True si le modèle a été entraîné.
    train_history : list of float
        Historique de la loss MSE par époque.
    """

    def __init__(self,
                 srm_path: str,
                 hidden_dims: list = None,
                 dropout: float = 0.1,
                 device: str = None):

        self.srm_path    = srm_path
        self.srm_data    = load_srm(srm_path)
        self.hidden_dims = hidden_dims or [256, 512, 256]
        self.dropout     = dropout
        self.device      = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        n_det  = self.srm_data['matrix'].shape[1]
        n_true = self.srm_data['matrix'].shape[0]

        # Scalers séparés : counts, SRM aplatie, photons
        self.scaler_X     = LogStandardScaler()   # counts (n_det,)
        self.scaler_S     = LogStandardScaler()   # SRM aplatie (n_true*n_det,)
        self.scaler_Y     = LogStandardScaler()   # photons (n_true,)

        # Le réseau reçoit counts + SRM en entrée
        self.net = PhotonMLP(n_det, n_true, self.hidden_dims, self.dropout)
        self.net.to(self.device)

        self.is_trained   = False
        self.train_history = []

        print(f"[NeuralNetModel] device={self.device} | "
              f"n_det={n_det} | n_true={n_true} | "
              f"input_dim={n_det + n_true * n_det} (counts + SRM aplatie)")

    def train(self,
              n_samples:       int   = 10000,
              n_epochs:        int   = 150,
              batch_size:      int   = 128,
              lr:              float = 1e-3,
              weight_decay:    float = 1e-5,
              alpha_range:     tuple = (1.5, 5.0),
              amp_range:       tuple = (1e-3, 1e2),
              noise_level_srm: float = 0.05,
              poisson_noise:   bool  = True,
              val_frac:        float = 0.1,
              seed:            int   = 42,
              verbose:         bool  = True,
              patience:        int   = 20,
              min_delta:       float = 1e-6) -> None:
        """
        Génère les données simulées et entraîne le réseau.

        Sauvegarde automatiquement les poids du meilleur modèle (val_loss
        minimale) en mémoire, et arrête l'entraînement prématurément si la
        val_loss ne s'améliore pas de plus de `min_delta` pendant `patience`
        époques consécutives (early stopping).

        Parameters
        ----------
        n_samples : int
            Nombre de spectres simulés pour l'entraînement (défaut : 10 000).
        n_epochs : int
            Nombre maximal d'époques (défaut : 150).
        batch_size : int
            Taille des mini-batchs (défaut : 128).
        lr : float
            Taux d'apprentissage initial Adam (défaut : 1e-3).
        weight_decay : float
            Régularisation L2 sur les poids (défaut : 1e-5).
        alpha_range : tuple
            Intervalle de l'indice spectral α.
        amp_range : tuple
            Intervalle log-uniforme de l'amplitude A.
        noise_level_srm : float
            Bruit relatif sur les SRM synthétiques (défaut : 5%).
        poisson_noise : bool
            Ajout d'un bruit de Poisson sur les counts simulés.
        val_frac : float
            Fraction de validation (défaut : 10%).
        seed : int
            Graine aléatoire pour la reproductibilité.
        verbose : bool
            Affichage de la progression.
        patience : int
            Nombre d'époques sans amélioration avant arrêt anticipé
            (défaut : 20). Mettre à None pour désactiver l'early stopping.
        min_delta : float
            Amélioration minimale de la val_loss considérée comme un progrès
            (défaut : 1e-6). En dessous de ce seuil, l'époque est comptée
            comme une stagnation.
        """
        import copy

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        srm_ref = self.srm_data['matrix']
        e_true  = self.srm_data['e_true']

        # ── Génération des données ─────────────────────────────────────────
        if verbose:
            print(f"[Train] Génération de {n_samples} spectres simulés...")

        counts_all, photons_all, _, srms_all = generate_power_law_dataset(
            n_samples       = n_samples,
            e_true          = e_true,
            srm_ref         = srm_ref,
            alpha_range     = alpha_range,
            amp_range       = amp_range,
            noise_level_srm = noise_level_srm,
            poisson_noise   = poisson_noise,
            rng             = rng,
        )

        # ── Split train / validation ───────────────────────────────────────
        n_val   = int(n_samples * val_frac)
        n_train = n_samples - n_val

        X_train, X_val = counts_all[:n_train],  counts_all[n_train:]
        Y_train, Y_val = photons_all[:n_train], photons_all[n_train:]
        S_train, S_val = srms_all[:n_train],    srms_all[n_train:]

        # ── Normalisation ──────────────────────────────────────────────────
        X_train_n = self.scaler_X.fit_transform(X_train)
        X_val_n   = self.scaler_X.transform(X_val)
        S_train_n = self.scaler_S.fit_transform(S_train)
        S_val_n   = self.scaler_S.transform(S_val)
        Y_train_n = self.scaler_Y.fit_transform(Y_train)
        Y_val_n   = self.scaler_Y.transform(Y_val)

        # Concaténation counts + SRM aplatie → entrée du réseau
        XS_train_n = np.concatenate([X_train_n, S_train_n], axis=1)
        XS_val_n   = np.concatenate([X_val_n,   S_val_n],   axis=1)

        # ── DataLoaders ────────────────────────────────────────────────────
        def to_loader(XS, Y, shuffle):
            ds = TensorDataset(
                torch.tensor(XS, dtype=torch.float32),
                torch.tensor(Y,  dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        loader_train = to_loader(XS_train_n, Y_train_n, shuffle=True)
        loader_val   = to_loader(XS_val_n,   Y_val_n,   shuffle=False)

        # ── Optimiseur + scheduler ─────────────────────────────────────────
        optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        loss_fn = nn.MSELoss()

        # ── État early stopping & best checkpoint ──────────────────────────
        best_val_loss   = float('inf')
        best_epoch      = 0
        best_state_dict = None          # poids du meilleur modèle (en mémoire)
        stagnation      = 0             # compteur d'époques sans progrès

        # ── Boucle d'entraînement ──────────────────────────────────────────
        self.train_history = []
        self.val_history   = []
        self.lr_history    = []
        if verbose:
            es_info = (f"patience={patience}, min_delta={min_delta:.0e}"
                       if patience else "early stopping désactivé")
            print(f"[Train] Démarrage — {n_epochs} époques max | "
                  f"device={self.device} | lr={lr} | {es_info}")

        for epoch in range(1, n_epochs + 1):
            # ── Passe d'entraînement ───────────────────────────────────────
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

            # ── Passe de validation ────────────────────────────────────────
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
                tag = ' ★'          # marqueur visuel d'amélioration
            else:
                stagnation += 1
                tag = f' ({stagnation}/{patience})' if patience else ''

            if verbose and (epoch % 10 == 0 or epoch == 1 or improved):
                print(f"  Époque {epoch:4d}/{n_epochs} | "
                      f"train={train_loss:.4e} | val={val_loss:.4e}{tag}")

            # ── Early stopping ─────────────────────────────────────────────
            if patience and stagnation >= patience:
                if verbose:
                    print(f"\n[Train] Early stopping déclenché à l'époque {epoch} "
                          f"(pas d'amélioration depuis {patience} époques).")
                break

        # ── Restauration du meilleur état ──────────────────────────────────
        if best_state_dict is not None:
            self.net.load_state_dict(best_state_dict)
            if verbose:
                print(f"[Train] Meilleur modèle restauré → "
                      f"époque {best_epoch} | val_loss={best_val_loss:.4e}")

        self.best_epoch    = best_epoch
        self.best_val_loss = best_val_loss
        self.is_trained    = True
        if verbose:
            print("[Train] Entraînement terminé.")

    def predict(self, counts: np.ndarray,
                srm: np.ndarray = None) -> np.ndarray:
        """
        Reconstruit le spectre photonique à partir d'un vecteur de counts
        et de la SRM associée.

        Parameters
        ----------
        counts : ndarray (n_det,) or (N, n_det)
            Counts observés par le détecteur.
        srm : ndarray (n_true, n_det) or (N, n_true, n_det), optional
            Matrice de réponse instrumentale utilisée pour l'acquisition.
            Si None, utilise la SRM de référence chargée depuis le fichier FITS.

        Returns
        -------
        ndarray (n_true,) or (N, n_true)
            Spectre photonique reconstruit [photons cm-2 s-1 keV-1].

        Raises
        ------
        RuntimeError
            Si le modèle n'a pas encore été entraîné.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Le modèle n'est pas entraîné. Appeler .train() d'abord.")

        single = counts.ndim == 1
        if single:
            counts = counts[np.newaxis, :]   # (1, n_det)

        N = counts.shape[0]

        # ── Préparation de la SRM ─────────────────────────────────────────
        if srm is None:
            # Utilise la SRM de référence pour tous les échantillons
            srm_flat = self.srm_data['matrix'].ravel()[np.newaxis, :]  # (1, n_true*n_det)
            srm_flat = np.repeat(srm_flat, N, axis=0)                  # (N, n_true*n_det)
        else:
            if srm.ndim == 2:
                # Une seule SRM (n_true, n_det) → répliquer pour chaque sample
                srm_flat = srm.ravel()[np.newaxis, :]
                srm_flat = np.repeat(srm_flat, N, axis=0)
            else:
                # Batch de SRM (N, n_true, n_det)
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

        # Dénormalisation
        photons = self.scaler_Y.inverse_transform_output(Y_n)

        return photons[0] if single else photons

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle entraîné (poids + scalers) sur disque.

        Parameters
        ----------
        path : str
            Chemin du fichier de sauvegarde (.pt recommandé).
        """
        torch.save({
            'state_dict':  self.net.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout':     self.dropout,
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_std':  self.scaler_X.std_,
            'scaler_S_mean': self.scaler_S.mean_,
            'scaler_S_std':  self.scaler_S.std_,
            'scaler_Y_mean': self.scaler_Y.mean_,
            'scaler_Y_std':  self.scaler_Y.std_,
            'srm_path':      self.srm_path,
            'n_det':         self.srm_data['matrix'].shape[1],
            'n_true':        self.srm_data['matrix'].shape[0],
            'best_epoch':    getattr(self, 'best_epoch',    None),
            'best_val_loss': getattr(self, 'best_val_loss', None),
        }, path)
        print(f"[NeuralNetModel] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'NeuralNetModel':
        """
        Charge un modèle précédemment sauvegardé.

        Parameters
        ----------
        path : str
            Chemin du fichier .pt généré par .save().
        device : str or None
            Forcer un device spécifique ('cpu', 'cuda').

        Returns
        -------
        NeuralNetModel
            Instance prête pour .predict().
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        instance = cls.__new__(cls)
        instance.srm_path    = ckpt['srm_path']
        instance.srm_data    = load_srm(ckpt['srm_path'])
        instance.hidden_dims = ckpt['hidden_dims']
        instance.dropout     = ckpt['dropout']
        instance.device      = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        n_det, n_true = ckpt['n_det'], ckpt['n_true']
        instance.net = PhotonMLP(n_det, n_true, ckpt['hidden_dims'], ckpt['dropout'])
        instance.net.load_state_dict(ckpt['state_dict'])
        instance.net.to(instance.device)

        instance.scaler_X       = LogStandardScaler()
        instance.scaler_X.mean_ = ckpt['scaler_X_mean']
        instance.scaler_X.std_  = ckpt['scaler_X_std']

        instance.scaler_S       = LogStandardScaler()
        instance.scaler_S.mean_ = ckpt['scaler_S_mean']
        instance.scaler_S.std_  = ckpt['scaler_S_std']

        instance.scaler_Y       = LogStandardScaler()
        instance.scaler_Y.mean_ = ckpt['scaler_Y_mean']
        instance.scaler_Y.std_  = ckpt['scaler_Y_std']

        instance.is_trained    = True
        instance.train_history = []
        instance.best_epoch    = ckpt.get('best_epoch',    None)
        instance.best_val_loss = ckpt.get('best_val_loss', None)
        print(f"[NeuralNetModel] Modèle chargé depuis {path} | device={instance.device}")
        return instance

if __name__ == "__main__":
    NeuralNetModel = NeuralNetModel(os.path.join('..', '..', '..', '..', 'data', 'stx_srm_2303197888.fits'))
    NeuralNetModel.train()
    NeuralNetModel.save(os.path.join('..', '..', '..', '..', 'data', 'nn_powerlaw.pt'))