import numpy as np
from astropy.modeling import FittableModel, Parameter


class ForwardFolded:

    # function to calculate the flux
    def integrate_flux(e1, e2, model_func, n_points=10):
        """
        Intègre model_func sur [e1, e2] par trapèzes et retourne le flux moyen.
        """
        if e1 >= e2:
            raise ValueError(f"e1 ({e1}) doit être strictement < e2 ({e2})")

        energies = np.linspace(e1, e2, n_points)
        fluxes = model_func(energies)
        return np.trapz(fluxes, energies) / (e2 - e1)

    class PowerLaw(FittableModel):
        """
        Loi de puissance forward-foldée sur une matrice de réponse.

        P(E) = amplitude * (E / E_pivot)^(-alpha)

        Parameters
        ----------
        amplitude : float  Normalisation (> 0)
        alpha     : float  Indice spectral (> 0)
        E_pivot   : float  Énergie pivot en keV (fixe, défaut 100 keV)
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
                ForwardFolded.integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])

            folded = np.dot(true_fluxes, self.matrix) / self.exposure

            folded = np.nan_to_num(folded, nan=1e-30, posinf=0.0, neginf=0.0)
            return np.clip(folded, 1e-30, None)

    # === Broken Power Law ===
    class BrokenPowerLaw(FittableModel):
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
                ForwardFolded.integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])
            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded

    # === VTH ===
    class VTH(FittableModel):
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
                ForwardFolded.integrate_flux(e1, e2, thermal_model)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])

            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded

    # === Exponential Power Law ===
    class ExpPowerLaw(FittableModel):
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
                ForwardFolded.integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])
            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded

    # === VTH + Power Law ===
    class VTHPlusPowerLaw(FittableModel):
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
                ForwardFolded.integrate_flux(e1, e2, model_total)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])

            # Forward-folding via SRM
            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded

    # === Power Law Cutoff Fix ===
    class PowerLawCutoffFix(FittableModel):
        """
        Power law avec cutoff fixe (E_c constant mais modifiable par l'utilisateur).
        P(E) = A * E^-alpha si E >= E_c, sinon 0
        """
        n_inputs = 1
        n_outputs = 1

        amplitude = Parameter(default=1e-2)
        alpha = Parameter(default=2.0, min=0.1, max=5.0)

        def __init__(self, e_low_true, e_high_true, matrix, exposure, E_cut=10.0, **kwargs):
            super().__init__(**kwargs)
            self.e_low_true = e_low_true
            self.e_high_true = e_high_true
            self.matrix = matrix
            self.exposure = exposure
            self.E_cut = E_cut

        def evaluate(self, x, amplitude, alpha):
            model_func = lambda E: np.where(E >= self.E_cut, amplitude * (E) ** (-alpha), 0.0)

            true_fluxes = np.array([
                ForwardFolded.integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])

            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded

    # === Power Law Cutoff Free ===
    class PowerLawCutoffFree(FittableModel):
        """
        Power law avec cutoff libre (E_c est un Parameter ajusté).

        P(E) = A * E^(-alpha)  si E >= E_cut
               0               sinon  (coupure dure)

        Parameters
        ----------
        amplitude : float
            Normalisation du flux (> 0).
        alpha     : float
            Indice spectral (> 0).
        E_cut     : float
            Énergie de coupure en keV (> 0).
        """

        n_inputs = 1
        n_outputs = 1

        # FIX 1 : bounds définis via min/max, pas bounds=(...),
        # pour être correctement lus par les fitters scipy.
        # FIX 2 : amplitude et alpha bornés inférieurement pour
        # éviter flux négatifs et indices non physiques.
        amplitude = Parameter(default=1e-2)  # flux > 0 obligatoire
        alpha = Parameter(default=2.0, min=0.1, max=5.0)  # indice physique
        E_cut = Parameter(default=10.0)  # cutoff fitté

        def __init__(self, e_low_true, e_high_true, matrix, exposure, **kwargs):
            super().__init__(**kwargs)
            self.e_low_true = np.asarray(e_low_true, dtype=float)
            self.e_high_true = np.asarray(e_high_true, dtype=float)
            self.matrix = np.asarray(matrix, dtype=float)
            # FIX 3 : exposure ne peut pas être nul (division par zéro)
            if np.isscalar(exposure):
                if exposure <= 0:
                    raise ValueError("exposure doit être > 0")
            self.exposure = float(exposure)

        def evaluate(self, x, amplitude, alpha, E_cut):

            # On force la conversion en scalaire Python pour np.where.
            E_cut_val = float(np.squeeze(E_cut))

            def model_func(E):
                E = np.asarray(E, dtype=float)
                E_safe = np.maximum(E, 1e-12)  # évite E=0 dans le power law
                flux = amplitude * E_safe ** (-alpha)
                return np.where(E >= E_cut_val, flux, 0.0)

            # Intégration sur les bins d'énergie vraie
            true_fluxes = np.array([
                ForwardFolded.integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])

            folded = np.dot(true_fluxes, self.matrix) / self.exposure

            folded = np.nan_to_num(folded, nan=1e-30, posinf=0.0, neginf=0.0)
            return np.clip(folded, 1e-30, None)

    # === VTH + Power Law Cutoff Fix ===
    class VTHPlusPowerLawCutoffFix(FittableModel):
        n_inputs = 1
        n_outputs = 1

        # Paramètres VTH
        EM = Parameter(default=1e48, bounds=(1e44, 1e52))
        T = Parameter(default=1.0, bounds=(0.1, 50.0))

        # Paramètres Power Law
        amplitude = Parameter(default=1e-2)
        alpha = Parameter(default=2.0)

        def __init__(self, e_low_true, e_high_true, matrix, exposure, E_cut=10.0, **kwargs):
            super().__init__(**kwargs)
            self.e_low_true = e_low_true
            self.e_high_true = e_high_true
            self.matrix = matrix
            self.exposure = exposure
            self.E_cut = E_cut

        def evaluate(self, x, EM, T, amplitude, alpha):
            # Constante de Gaunt
            gff = 1.2
            A = 1.07e-42 * gff
            safe_T = max(1e-3, T)

            def model_total(E):
                # Thermal component
                thermal = (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T)
                # Power-law component
                # power = amplitude * (E / self.E_pivot) ** (-alpha)
                power = np.where(E >= self.E_cut, amplitude * (E) ** (-alpha), 0.0)
                return thermal + power

            # Intégration du flux photonique dans chaque bin SRM
            true_fluxes = np.array([
                ForwardFolded.integrate_flux(e1, e2, model_total)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])

            # Forward-folding via SRM
            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded

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
                ForwardFolded.integrate_flux(e1, e2, model_func)
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
                ForwardFolded.integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(self.e_low_true, self.e_high_true)
            ])
            folded = np.dot(true_fluxes, self.matrix) / self.exposure
            return folded