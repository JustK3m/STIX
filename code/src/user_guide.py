from tkinter import *


def open_user_guide():
    """Ouvre la fenêtre User Guide."""
    guide = Toplevel(background="#f7f9fc")
    guide.title("User Guide")
    guide.geometry("700x600")
    guide.resizable(True, True)

    # --- Scrollable frame ---
    frame = Frame(guide)
    frame.pack(fill=BOTH, expand=True, padx=15, pady=10)

    scrollbar = Scrollbar(frame)
    scrollbar.pack(side=RIGHT, fill=Y)

    text = Text(
        frame,
        wrap=WORD,
        yscrollcommand=scrollbar.set,
        padx=10, pady=8,
        font=("Helvetica", 10),
        relief=FLAT,
        state=NORMAL
    )
    text.pack(fill=BOTH, expand=True)
    scrollbar.config(command=text.yview)

    # --- Tags de mise en forme ---
    text.tag_config("title", font=("Helvetica", 13, "bold"), spacing3=6)
    text.tag_config("heading", font=("Helvetica", 11, "bold"), spacing1=10, spacing3=3)
    text.tag_config("subhead", font=("Helvetica", 10, "bold italic"), spacing1=6)
    text.tag_config("body", font=("Helvetica", 10), spacing1=2)
    text.tag_config("note", font=("Helvetica", 9, "italic"), foreground="#555555")

    # --- Contenu ---
    content = [
        ("title", "STIX Spectral Data Analysis Package — User Guide\n"),

        ("heading", "1. General Workflow\n"),
        ("body",
         "The analysis follows three successive steps, accessible from the "
         "File menu of the main window.\n"),

        ("subhead", "Step 1 — Select Input\n"),
        ("body",
         "Load a STIX spectrum FITS file and a response matrix FITS file (SRM). "
         "Data are displayed as a time profile or spectrogram. "
         "The display unit is selectable: Rate (counts/s), Counts (raw counts) "
         "or Flux (photons cm⁻² s⁻¹ keV⁻¹).\n"),

        ("subhead", "Step 2 — Select Background\n"),
        ("body",
         "Select a time interval outside the event (before or after the flare) "
         "as a background reference. The selection can be made manually "
         "(by entering dates) or graphically on the time profile. "
         "The estimation method is chosen per energy band from: "
         "Median, Mean, 1Poly, 2Poly, 3Poly, Exp.\n"),

        ("subhead", "Step 3 — Plot Fit Results\n"),
        ("body",
         "Select a spectral model, configure initial values and bounds via "
         "'Function value(s)', choose the fit statistic (Chi² or C-stat), "
         "define the energy range to fit, then click 'Do Fit'. "
         "The fitted parameters are displayed on the plot. "
         "The 'Photon' option shows the deconvolved photon spectrum "
         "in true energy space.\n"),

        ("heading", "2. Available Spectral Models\n"),

        ("subhead", "Power Law\n"),
        ("body",
         "Φ(E) = A · (E / E_pivot)^(−α)\n"
         "Parameters: amplitude A, spectral index α, pivot energy E_pivot (keV, fixed).\n"
         "Use case: standard non-thermal emission.\n"),

        ("subhead", "Broken Power Law\n"),
        ("body",
         "Φ(E) = A · (E/E_b)^(−α₁) if E < E_b,  A · (E/E_b)^(−α₂) if E ≥ E_b\n"
         "Parameters: A, E_break, α₁, α₂.\n"
         "Use case: transition between two acceleration regimes.\n"),

        ("subhead", "Single Power Law × Exponential\n"),
        ("body",
         "Φ(E) = p0 · (E/p2)^p1 · exp(e3 − E/e4)\n"
         "Parameters: p0 (normalisation), p1 (index), p2 (reference energy), "
         "e3 (exponential offset), e4 (cutoff energy scale).\n"
         "Use case: empirical model with variable spectral curvature.\n"),

        ("subhead", "V_TH (Thermal Bremsstrahlung)\n"),
        ("body",
         "Φ(E) = (A_ff · EM) / (E · √T) · exp(−E/T),  A_ff = 1.07×10⁻⁴² · g_ff\n"
         "Parameters: plasma temperature T (keV), emission measure EM (cm⁻³).\n"
         "Use case: thermal component of the impulsive phase.\n"),

        ("subhead", "V_TH + Power Law\n"),
        ("body",
         "Φ(E) = Φ_VTH(E) + A · (E / E_pivot)^(−α)\n"
         "Parameters: EM, T, amplitude, α, E_pivot.\n"
         "Use case: flares exhibiting simultaneous thermal and non-thermal components.\n"),

        ("subhead", "Power Law Cutoff Fix\n"),
        ("body",
         "Φ(E) = A · (E / E_pivot)^(−α)  if E ≥ E_cut, else 0\n"
         "E_cut is fixed (not fitted) but can be changed via 'Function value(s)'.\n"
         "Fitted parameters: A, α.\n"
         "Use case: non-thermal emission with a known low-energy cutoff.\n"),

        ("subhead", "Power Law Cutoff Free\n"),
        ("body",
         "Φ(E) = A · (E / E_pivot)^(−α)  if E ≥ E_cut, else 0\n"
         "E_cut is determined automatically by minimising χ² over the interval "
         "[Ec_min, Ec_max] defined in 'Function value(s)' (bounded scalar minimisation).\n"
         "Fitted parameters: A, α, E_cut (optimised).\n"
         "Use case: non-thermal emission where the low-energy cutoff is unknown.\n"),

        ("subhead", "V_TH + Power Law Cutoff Fix\n"),
        ("body",
         "Two-component model fitted sequentially over complementary energy ranges:\n"
         "  · Power law with fixed cutoff fitted over [E_cut, E_max].\n"
         "  · V_TH fitted over [E_min, E_cut].\n"
         "Parameters: EM, T (thermal), amplitude, α, E_pivot, E_cut (fixed).\n"
         "Use case: flares with a clearly separated thermal/non-thermal boundary.\n"),

        ("subhead", "V_TH + Power Law Cutoff Free\n"),
        ("body",
         "Same as V_TH + Power Law Cutoff Fix, but E_cut is determined automatically "
         "by minimising χ² over [Ec_min, Ec_max] (bounded scalar minimisation). "
         "The two components are then fitted sequentially on either side of the "
         "optimal E_cut.\n"
         "Parameters: EM, T (thermal), amplitude, α, E_pivot, E_cut (optimised).\n"
         "Use case: flares where the thermal/non-thermal transition energy is unknown.\n"),

        ("heading", "3. Fit Statistics\n"),
        ("subhead", "Chi² (default)\n"),
        ("body",
         "Minimisation of weighted residuals in the least-squares sense. "
         "Suitable when the number of counts per bin is large enough (≳ 10).\n"),
        ("subhead", "C-stat (Cash)\n"),
        ("body",
         "C = 2 · Σ |M_i − D_i · ln(M_i)|\n"
         "Exact maximum-likelihood estimator for Poisson-distributed data. "
         "Recommended at high energies or for weak flares.\n"),

        ("note",
         "Both statistics use the Levenberg-Marquardt optimisation algorithm.\n"),

        ("heading", "4. Display Options\n"),
        ("body",
         "Display parameters: shows the fitted parameters on the plot.\n"
         "Show grid: enables the log-log grid.\n"
         "Data-Background: subtracts the background before fitting "
         "(requires Step 2 to have been completed).\n"
         "Photon: displays the deconvolved photon spectrum in true energy space "
         "(axis scale selectable on request).\n"),
    ]

    for tag, txt in content:
        text.insert(END, txt, tag)

    text.config(state=DISABLED)

    Button(guide, text="close", command=guide.destroy,
           bg="#374151", fg="white", padx=20).pack(pady=8)
