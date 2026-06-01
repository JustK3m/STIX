"""
NeuralNet.py — Reconstruction du spectre photonique par réseau de neurones (PyTorch).

Méthode : apprentissage supervisé sur données simulées.
    - Spectres photoniques générés par loi de puissance : Φ(E) = A · E^(−α)
    - Counts simulés : c = SRM @ Φ
    - Le réseau apprend la relation inverse : counts → Φ

SRM synthétiques : perturbations multiplicatives de la vraie SRM STIX
    (bruit lognormal + lissage gaussien), garantissant la structure physique.

Architecture : MLP 3 couches cachées avec BatchNorm + Dropout + Softplus en sortie
    (Softplus garantit la positivité du spectre photonique reconstruit).

Dimensions STIX :
    - n_det = 30  canaux détecteur (mesurés)
    - n_true = 1028  bins en énergie vraie

Usage rapide :
    srm_path = "data/stx_srm_2303197888.fits"
    nn_model = NeuralNetModel(srm_path)
    nn_model.train(n_samples=10000, n_epochs=150, batch_size=128)
    photons = nn_model.predict(counts_vector)

Intégration dans fit_all.py (idx == N) :
    from process.fitting.methods.NeuralNet import NeuralNetModel
    model = NeuralNetModel(Fitting.rname)
    model.train()
    photon_spectrum = model.predict(counts[fit_mask])
"""
import os
import sys

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
    """
    if rng is None:
        rng = np.random.default_rng()

    n_true, n_det = srm_ref.shape
    counts_all  = np.zeros((n_samples, n_det),  dtype=np.float32)
    photons_all = np.zeros((n_samples, n_true), dtype=np.float32)
    params_all  = np.zeros((n_samples, 2),      dtype=np.float32)

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

        counts_all[i]  = counts_i
        photons_all[i] = phi
        params_all[i]  = [A, alpha]

    return counts_all, photons_all, params_all


# ══════════════════════════════════════════════════════════════════════════════
#  4. Architecture du réseau de neurones
# ══════════════════════════════════════════════════════════════════════════════

class PhotonMLP(nn.Module):
    """
    Réseau de neurones MLP pour la reconstruction du spectre photonique.

    Architecture :
        Input (n_det) → Linear → BN → ReLU → Dropout
                      → Linear → BN → ReLU → Dropout
                      → Linear → BN → ReLU → Dropout
                      → Linear → Softplus → Output (n_true)

    La couche Softplus en sortie garantit des valeurs strictement positives,
    ce qui est physiquement requis pour un spectre de flux photonique.

    Parameters
    ----------
    n_det : int
        Dimension d'entrée (nombre de canaux détecteur mesurés).
    n_true : int
        Dimension de sortie (nombre de bins en énergie vraie).
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

        layers = []
        in_dim = n_det
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
        x : Tensor (batch_size, n_det)
            Counts mesurés en entrée.

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

        self.net = PhotonMLP(n_det, n_true, self.hidden_dims, self.dropout)
        self.net.to(self.device)

        self.scaler_X     = LogStandardScaler()
        self.scaler_Y     = LogStandardScaler()
        self.is_trained   = False
        self.train_history = []

        print(f"[NeuralNetModel] device={self.device} | "
              f"n_det={n_det} | n_true={n_true}")

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
              verbose:         bool  = True) -> None:
        """
        Génère les données simulées et entraîne le réseau.

        Parameters
        ----------
        n_samples : int
            Nombre de spectres simulés pour l'entraînement (défaut : 10 000).
        n_epochs : int
            Nombre d'époques d'entraînement (défaut : 150).
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
        """
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        srm_ref = self.srm_data['matrix']
        e_true  = self.srm_data['e_true']

        # ── Génération des données ─────────────────────────────────────────
        if verbose:
            print(f"[Train] Génération de {n_samples} spectres simulés...")

        counts_all, photons_all, _ = generate_power_law_dataset(
            n_samples       = n_samples,
            e_true          = e_true,
            srm_ref         = srm_ref,
            alpha_range     = alpha_range,
            amp_range       = amp_range,
            noise_level_srm = noise_level_srm,
            rng             = rng,
        )

        # ── Split train / validation ───────────────────────────────────────
        n_val   = int(n_samples * val_frac)
        n_train = n_samples - n_val

        X_train, X_val = counts_all[:n_train],  counts_all[n_train:]
        Y_train, Y_val = photons_all[:n_train], photons_all[n_train:]

        # ── Normalisation ──────────────────────────────────────────────────
        X_train_n = self.scaler_X.fit_transform(X_train)
        X_val_n   = self.scaler_X.transform(X_val)
        Y_train_n = self.scaler_Y.fit_transform(Y_train)
        Y_val_n   = self.scaler_Y.transform(Y_val)

        # ── DataLoaders ────────────────────────────────────────────────────
        def to_loader(X, Y, shuffle):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        loader_train = to_loader(X_train_n, Y_train_n, shuffle=True)
        loader_val   = to_loader(X_val_n,   Y_val_n,   shuffle=False)

        # ── Optimiseur + scheduler ─────────────────────────────────────────
        optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        loss_fn = nn.MSELoss()

        # ── Boucle d'entraînement ──────────────────────────────────────────
        self.train_history = []
        if verbose:
            print(f"[Train] Démarrage — {n_epochs} époques | "
                  f"device={self.device} | lr={lr}")

        for epoch in range(1, n_epochs + 1):
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

            # ── Validation ────────────────────────────────────────────────
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += loss_fn(self.net(xb), yb).item() * len(xb)
            val_loss /= n_val

            self.train_history.append(train_loss)
            scheduler.step()

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  Époque {epoch:4d}/{n_epochs} | "
                      f"train_loss={train_loss:.4e} | val_loss={val_loss:.4e}")

        self.is_trained = True
        if verbose:
            print("[Train] Entraînement terminé.")

    def predict(self, counts: np.ndarray) -> np.ndarray:
        """
        Reconstruit le spectre photonique à partir d'un vecteur de counts.

        Parameters
        ----------
        counts : ndarray (n_det,) or (N, n_det)
            Counts observés par le détecteur.

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
            counts = counts[np.newaxis, :]

        X_n = self.scaler_X.transform(counts)
        X_t = torch.tensor(X_n, dtype=torch.float32).to(self.device)

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
            'scaler_Y_mean': self.scaler_Y.mean_,
            'scaler_Y_std':  self.scaler_Y.std_,
            'srm_path':    self.srm_path,
            'n_det':       self.srm_data['matrix'].shape[1],
            'n_true':      self.srm_data['matrix'].shape[0],
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
        ckpt = torch.load(path, map_location='cpu')

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

        instance.scaler_Y       = LogStandardScaler()
        instance.scaler_Y.mean_ = ckpt['scaler_Y_mean']
        instance.scaler_Y.std_  = ckpt['scaler_Y_std']

        instance.is_trained    = True
        instance.train_history = []
        print(f"[NeuralNetModel] Modèle chargé depuis {path} | device={instance.device}")
        return instance