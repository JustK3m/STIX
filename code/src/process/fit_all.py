import copy
import os
import sys
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.optimize import minimize_scalar

from . import background
from .fitting.fitters import LevMarCstatFitter
from .fitting.methods import ForwardFolded
from .fitting.methods.ForwardFolded import *

register_matplotlib_converters()


class Fitting:
    """
    Class to perform a spectrum fitting
    """

    fname_r = 'data/solo_L1_stix-sci-xray' \
              '-spec_20230319T175504-20230320T000014_V02_2303197888-65462.fits'
    rname_r = 'data/stx_srm_2303197888.fits'

    def resource_path(relative_path):
        """Renvoie le chemin absolu même si l'app est congelée avec PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(relative_path)

    fname = resource_path(fname_r)
    rname = resource_path(rname_r)

    # ── Bornes par défaut ──────────────────────────────────────
    default_param_bounds = {
        "PowerLaw1D": {"amplitude": (None, None), "alpha": (None, None)},
        "BrokenPowerLaw1D": {"amplitude": (1e-5, 1e3), "E_break": (1.0, 100.0),
                             "alpha_1": (0.1, 10.0), "alpha_2": (0.1, 10.0)},
        "Single Power Law Times an Exponential": {
            "p0": (1e-3, 1e5), "p1": (-5, 5), "p2": (1e-2, 100),
            "e3": (-10, 10), "e4": (0.1, 100)},
        "V_TH": {"EM": (1e44, 1e52), "T": (0.1, 50.0)},
        "V_TH + PowerLaw": {"EM": (1e44, 1e52), "T": (0.1, 50.0),
                            "amplitude": (1e-2, 1e2), "alpha": (2, 10.0)},
        "PowerLawCutoffFix": {"amplitude": (1e-12, 1e6), "alpha": (0.1, 50.0)},
        "PowerLawCutoffFree": {"amplitude": (1e-12, 1e6), "alpha": (0.1, 50.0)},
        "V_TH + PowerLawCutoffFix": {"EM": (1e44, 1e52), "T": (0.1, 50.0),
                                     "amplitude": (1e-2, 1e2), "alpha": (2, 10.0)},
        "V_TH + PowerLawCutoffFree": {"EM": (1e44, 1e52), "T": (0.1, 50.0),
                                      "amplitude": (1e-2, 1e2), "alpha": (2, 10.0)},
    }

    # ── Valeurs initiales par défaut ───────────────────────────
    default_param_values = {
        "PowerLaw1D": {"amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0},
        "BrokenPowerLaw1D": {"amplitude": 1e-2, "E_break": 10.0,
                             "alpha_1": 2.0, "alpha_2": 3.0},
        "Single Power Law Times an Exponential": {
            "p0": 1.0, "p1": -2.0, "p2": 20.0, "e3": 1.0, "e4": 10.0},
        "V_TH": {"EM": 6e48, "T": 1.0},
        "V_TH + PowerLaw": {"EM": 1e48, "T": 1.0,
                            "amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0},
        "PowerLawCutoffFix": {"amplitude": 1e-2, "alpha": 2.0,
                              "E_cut": 10.0, "E_pivot": 100.0},
        "PowerLawCutoffFree": {"amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0,
                               "Ec_min": 4, "Ec_max":20},
        "V_TH + PowerLawCutoffFix": {"EM": 1e48, "T": 1.0,
                                     "amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0, "E_cut": 10},
        "V_TH + PowerLawCutoffFree": {"EM": 1e48, "T": 1.0,
                                      "amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0},
    }

    # create a new window called 'SPEX Fit Options'
    def __init__(self, root):
        """Creates a new window, providing widgets to perform fitting analysis"""
        self.sender = None

        self.top2 = Toplevel()
        self.top2.title('SPEX Fit Options')  # title of the window
        self.top2.geometry("1000x600")  # size of the new window

        self.root = root
        self.hdul = None  # Opened file
        self.hdul2 = None  # Opened file
        self.name = None  # Name of the .fits file imported
        self.name2 = None  # # Name of the .fits file imported (response matrix)

        self.counts = None  # Matrix contaning the counts per band in function of time time
        self.counts_err = None  # Matrix contaning the error of the counts per band in function of time
        self.times = None  # Index of times for x axis
        self.time_del = None  # Time delay for the data
        self.energies = None  # Energy values for y axis
        self.e_low_det = None
        self.e_high_det = None

        self.area = 6  # Area of the surface of detection of the telescope in cm²; used for the flux

        self.e_low_true = None
        self.e_high_true = None
        self.matrix = None

        self.background_start = None
        self.background_end = None

        self.data = None  # Converts self.counts in chosen unit (rate, counts or flux) and adds time index
        self.bkg = None  # Calculated background noise
        self.data_bkg = None  # Data without background noise

        self.bkg_start_index = []  # Liste des indices de début pour chaque canal
        self.bkg_end_index = []  # Liste des indices de fin pour chaque canal
        self.var_sep_times = IntVar(value=0)

        self.fitter = LevMarLSQFitter()

        self.energy_min_var = tk.DoubleVar(value=0)
        self.energy_min2 = tk.OptionMenu(self.top2, self.energy_min_var, 0)

        self.energy_max_var = tk.DoubleVar(value=0)
        self.energy_max2 = tk.OptionMenu(self.top2, self.energy_max_var, 0)

        self.sepBkVar = IntVar()

        Label(self.top2, text="Choose Fit Function Model:", fg='blue',
              font=("Helvetica", 11, "bold")).place(relx=0.07, rely=0.07)  # set the position on window

        Label(self.top2, text="Information:", fg='blue',
              font=("Helvetica", 11, "bold")).place(relx=0.44, rely=0.07)  # set the position

        Label(self.top2, text="Choose the files and energy range:", fg='blue',
              font=("Helvetica", 11, "bold")).place(relx=0.65, rely=0.07)  # set the position

        # Spectrum: file name
        Label(self.top2, text="Spectrum: ").place(relx=0.65, rely=0.2, anchor=W)
        self.text_filename = Entry(self.top2, width=30)
        self.text_filename.place(relx=0.72, rely=0.2, anchor=W)
        if Fitting.fname:
            self.text_filename.insert(0, Fitting.fname)
            self.open_file(Fitting.fname)
        else:
            self.text_filename.insert(0, "No file chosen")

        Button(self.top2, text='Browse ->', command=self.open_file).place(relx=0.92, rely=0.2, anchor=W)

        # Response matrix: file name
        Label(self.top2, text="Response: ").place(relx=0.65, rely=0.25, anchor=W)
        self.text_filename2 = Entry(self.top2, width=30)
        self.text_filename2.place(relx=0.72, rely=0.25, anchor=W)
        if Fitting.rname:
            self.text_filename2.insert(0, Fitting.rname)
            self.open_srm_file(Fitting.rname)
        else:
            self.text_filename2.insert(0, "No file chosen")

        Button(self.top2, text='Browse ->', command=self.open_srm_file).place(relx=0.92, rely=0.25, anchor=W)

        self.fit_model = str()

        self.user_param_bounds = {}  # bounds set by user in Set_Function
        self.user_param_values = {}  # initial values set by user in Set_Function
        self.user_param_modified = {}  # True if user modified bounds/values from default

        def Set_Function():
            try:
                model_key = self.lbox.get(self.lbox.curselection()[0])
            except Exception:
                messagebox.showwarning("No Model Selected",
                                       "Please select a model first.")
                return

            newwin = Toplevel(self.top2)
            newwin.title(f"{model_key} - Parameter Settings")
            newwin.geometry("680x480")
            newwin.configure(bg="#f7f9fc")

            Label(newwin, text=f"Set parameter values for {model_key}",
                  fg="#1e3a8a", bg="#f7f9fc",
                  font=("Helvetica", 13, "bold")).pack(pady=15)

            base_defaults = Fitting.default_param_values.get(model_key, {})
            base_bounds = Fitting.default_param_bounds.get(model_key, {})
            saved_values = self.user_param_values.get(model_key, {})
            saved_bounds = self.user_param_bounds.get(model_key, {})

            # Paramètres sans min/max (valeur seule)
            VALUE_ONLY_PARAMS = {"E_pivot", "E_cut", "Ec_min", "Ec_max"}
            VALUE_ONLY_MODELS = {
                "PowerLaw1D", "V_TH + PowerLaw", "PowerLawCutoffFix",
                "PowerLawCutoffFree", "V_TH + PowerLawCutoffFix", "V_TH + PowerLawCutoffFree",
            }

            form_frame = Frame(newwin, bg="#f7f9fc")
            form_frame.pack(pady=10, padx=20, fill="x")

            param_entries = {}
            initial_display = {}

            for param, default_val in base_defaults.items():
                disp_default = str(saved_values.get(param, default_val))
                pmin, pmax = saved_bounds.get(param, base_bounds.get(param, (None, None)))
                disp_min = "" if pmin is None else str(pmin)
                disp_max = "" if pmax is None else str(pmax)

                row = Frame(form_frame, bg="#f7f9fc")
                row.pack(pady=6, fill="x")
                Label(row, text=f"{param}:", width=14, anchor="w",
                      bg="#f7f9fc").pack(side="left")

                # Cas value-only
                if model_key in VALUE_ONLY_MODELS and param in VALUE_ONLY_PARAMS:
                    Label(row, text="Value:", bg="#f7f9fc").pack(side="left")
                    e_def = Entry(row, width=10)
                    e_def.insert(0, disp_default)
                    e_def.pack(side="left", padx=6)
                    param_entries[param] = (e_def, None, None)
                    initial_display[param] = (disp_default, "", "")
                    continue

                # Cas normal : Default + Min + Max
                for label_txt, width in [("Default:", 10), ("Min:", 10), ("Max:", 10)]:
                    Label(row, text=label_txt, bg="#f7f9fc").pack(side="left")
                    e = Entry(row, width=width)
                    e.pack(side="left", padx=6)

                # Récupérer les entries créées (les 3 dernières dans row)
                entries_in_row = [w for w in row.winfo_children()
                                  if isinstance(w, Entry)]
                e_def, e_min, e_max = entries_in_row
                e_def.insert(0, disp_default)
                e_min.insert(0, disp_min)
                e_max.insert(0, disp_max)

                param_entries[param] = (e_def, e_min, e_max)
                initial_display[param] = (disp_default, disp_min, disp_max)

            # ── Actions ───────────────────────────────────────────
            def save_params():
                values, bounds = {}, {}
                modified = False
                for param, (e_def, e_min, e_max) in param_entries.items():
                    def_txt = e_def.get().strip()
                    try:
                        values[param] = float(def_txt)
                    except ValueError:
                        messagebox.showerror("Invalid input",
                                             f"{param}: invalid value")
                        return

                    if e_min is None:  # value-only
                        bounds[param] = (None, None)
                        if def_txt != initial_display[param][0]:
                            modified = True
                        continue

                    min_txt = e_min.get().strip()
                    max_txt = e_max.get().strip()
                    lo = float(min_txt) if min_txt else None
                    hi = float(max_txt) if max_txt else None
                    if lo is not None and hi is not None and hi <= lo:
                        messagebox.showerror("Invalid bounds",
                                             f"{param}: max must be > min")
                        return
                    bounds[param] = (lo, hi)
                    if (def_txt, min_txt, max_txt) != initial_display[param]:
                        modified = True

                self.user_param_values[model_key] = values
                self.user_param_bounds[model_key] = bounds
                self.user_param_modified[model_key] = modified
                print(f"[Set_Function] {model_key} saved: values={values}, "
                      f"bounds={bounds}, modified={modified}")
                newwin.destroy()

            def reset_defaults():
                for param, (e_def, e_min, e_max) in param_entries.items():
                    e_def.delete(0, END)
                    e_def.insert(0, str(base_defaults[param]))
                    if e_min is not None:
                        lo, hi = base_bounds.get(param, (None, None))
                        e_min.delete(0, END);
                        e_min.insert(0, "" if lo is None else str(lo))
                        e_max.delete(0, END);
                        e_max.insert(0, "" if hi is None else str(hi))

            # ── Boutons ───────────────────────────────────────────
            btn_frame = Frame(newwin, bg="#f7f9fc")
            btn_frame.pack(pady=20)
            for text, cmd, color in [
                ("Save", save_params, "#16a34a"),
                ("Reset to Defaults", reset_defaults, "#f97316"),
                ("Cancel", newwin.destroy, "#ef4444"),
            ]:
                Button(btn_frame, text=text, command=cmd,
                       bg=color, fg="white", width=14).pack(side="left", padx=10)

        Label(self.top2, text="Set function components: ").place(relx=0.65, rely=0.30)

        Button(self.top2, text="Function value(s)", command=Set_Function).place(relx=0.65, rely=0.35, relheight=0.05,
                                                                                relwidth=0.13)

        self.statname = "Chi2"

        def Set_Statistics(name):
            self.fitter = {"C-stat": LevMarCstatFitter(),
                           "Chi2": LevMarLSQFitter()}[name]
            self.statname = name
            self.menuStat.config(text=name)

        Label(self.top2, text="Set statistics:").place(relx=0.85, rely=0.30)

        self.menuStat = tk.Menubutton(self.top2, text="Chi2", relief="raised")
        self.menuStat.place(relx=0.85, rely=0.35, relheight=0.05, relwidth=0.13)

        self.menuStat.menu = tk.Menu(self.menuStat, tearoff=0)
        self.menuStat["menu"] = self.menuStat.menu
        for stat_name in ["Chi2", "C-stat"]:
            self.menuStat.menu.add_command(
                label=stat_name,
                command=lambda n=stat_name: Set_Statistics(n))

        # Energies range(s) to fit

        if Fitting.fname and Fitting.rname:
            n_channels = len(self.e_low_det)
            self.bkg_start_index = [0] * n_channels
            self.bkg_end_index = [len(self.times)] * n_channels

            usable = np.arange(min(self.matrix.shape[1], len(self.e_low_det)))
            e_low_det = self.e_low_det[usable]
            e_high_det = self.e_high_det[usable]

            e_low_values_int = sorted({int(e) for e in e_low_det if e != 0})
            e_high_values_int = sorted({int(e) for e in e_high_det
                                        if e not in (float('inf'), float('-inf'))})

            self.energy_min_var.set(min(e_low_values_int))
            self.energy_max_var.set(max(e_high_values_int))

            self.text_min_energy = Label(self.top2, text="Min energy")
            self.text_min_energy.place(relx=0.75, rely=0.45, anchor=N)
            self.text_max_energy = Label(self.top2, text="Max energy")
            self.text_max_energy.place(relx=0.85, rely=0.45, anchor=N)

            self.energy_min2 = OptionMenu(self.top2, self.energy_min_var,
                                          *e_low_values_int)
            self.energy_max2 = OptionMenu(self.top2, self.energy_max_var,
                                          *e_high_values_int)
            self.energy_min2.place(relx=0.75, rely=0.50, anchor=N)
            self.energy_max2.place(relx=0.85, rely=0.50, anchor=N)
        else:
            # Placeholder pour update_energy_range
            self.energy_min2 = OptionMenu(self.top2, self.energy_min_var, 0)
            self.energy_max2 = OptionMenu(self.top2, self.energy_max_var, 0)

        # ============== Main window description ==============

        self.lbox = Listbox(self.top2, selectmode=EXTENDED, highlightcolor='red', bd=4, selectbackground='grey')
        """ 
        On the left side of the 'SPEX Fit Options' window: place a list of text alternatives (listbox).
        The user can choose(highlight) one of the options.
        Options(functions):
        1) One Dimensional Power Law;
        2) 1-D Broken Power Law;
        3) Single Power Law Times an Exponetial
        """
        self.lbox.place(relx=0.05, rely=0.15, relheight=0.45, relwidth=0.25)

        self.scroll = Scrollbar(self.top2, command=self.lbox.yview)
        self.scroll.place(relx=0.3, rely=0.15, relheight=0.45, relwidth=0.02)
        self.lbox.config(yscrollcommand=self.scroll.set)

        # New frame at the bottom. Locate there 'Plot Units' and 'Do Fit' widgets
        self.frameFit = LabelFrame(self.top2, relief=RAISED,
                                   borderwidth=10)  # determine the border of the frame and size
        self.frameFit.place(relx=0.05, rely=0.63, relheight=0.25, relwidth=0.85)  # the frame position

        self.PlotUnits5 = Label(self.frameFit, text="Plot Units: ", fg='blue',
                                font=("Helvetica", 11, "bold"))  # lay out new text file
        self.PlotUnits5.place(relx=0.04, rely=0.4)

        # Add button for Units: Rate, Counts, Flux
        # Allows user to make a choice between three parameters
        self.Component_choicesFit = ('Rate', 'Counts', 'Flux')
        self.var = StringVar(self.frameFit)
        self.var.set(self.Component_choicesFit[0])
        self.selection = OptionMenu(self.frameFit, self.var, *self.Component_choicesFit)
        self.selection.place(relx=0.15, rely=0.38, relheight=0.23, relwidth=0.15)

        self.show_params_var = IntVar(value=1)  # Par défaut cochée
        self.show_params_check = Checkbutton(
            self.frameFit,
            text="Display parameters",
            variable=self.show_params_var
        )
        self.show_params_check.place(relx=0.35, rely=0.7)

        self.grid_var = IntVar(value=0)
        self.grid_check = Checkbutton(
            self.frameFit,
            text="Show grid",
            variable=self.grid_var
        )
        self.grid_check.place(relx=0.55, rely=0.7)

        self.show_db_var = IntVar(value=0)
        self.show_db_check = Checkbutton(
            self.frameFit,
            text="Data-Background",
            variable=self.show_db_var,
            command=self.on_background_clicked
        )

        self.show_db_check.place(relx=0.35, rely=0.5)

        self.show_photon_var = IntVar(value=0)

        def on_photon_toggle():
            if self.show_photon_var.get():
                self.ask_photon_axes_scale()

        self.show_photon_check = Checkbutton(
            self.frameFit,
            text="Photon",
            variable=self.show_photon_var,
            command=on_photon_toggle
        )

        self.show_photon_check.place(relx=0.55, rely=0.5)

        Button(self.frameFit, text="Do Fit",
               command=self._selective_fit).place(relx=0.70, rely=0.38, relheight=0.23, relwidth=0.15)  # locate

        Button(self.top2, text="Refresh").place(relx=0.4, rely=0.94)

        """Scrollbar with information related to each function"""
        Button(self.top2, text="Close", command=lambda: self.top2.destroy()).place(relx=0.5, rely=0.94)
        self.models = ['PowerLaw1D', 'BrokenPowerLaw1D', 'Single Power Law Times an Exponential', 'V_TH',
                       'V_TH + PowerLaw', 'PowerLawCutoffFix', 'PowerLawCutoffFree',
                       'V_TH + PowerLawCutoffFix', "V_TH + PowerLawCutoffFree"]  # , 'Neural Network' function names
        for p in self.models:
            """On the right: place an 'entry text' Scrollbar widget (scrollbar) When user highlight the function, 
            displays the text information about function description and input parameters"""
            self.lbox.insert(END, p)
        self.lbox.bind("<<ListboxSelect>>", self.onSelect)
        self.list = {'PowerLaw1D': ['One dimensional power law model',
                                    'amplitude – model amplitude at the reference energy',
                                    'Epivot – energie pivot (kEv)',
                                    'energy_data – reference energy', 'alpha – power law index'
                                    ],
                     # if user choose PowerLaw1D, display
                     'BrokenPowerLaw1D': ['One dimensional power law model with a break',
                                          'amplitude - model amplitude at the break energy',
                                          'alpha 1 – power law index for energy_data<x_break',
                                          'alpha 2 – power law index for energy_data>x_break'],
                     # if user choose BrokenPowerLaw1D, display
                     'Gaussian': ['Single Gaussian function(high quality), width in sigma',
                                  'does not go through DRM',
                                  'This function returns the sum of Gaussian and ', '2nd order Polynomial',
                                  'amplitude - integrated intensity, mean - centroid', 'stddev - sigma'],
                     # if user choose Gaussian, display
                     'Polynomial': ['Polynomial function with offset in energy_data',
                                    'c0 - 0th order coefficient', 'c1 - 1st order coefficient',
                                    'c2 - 2nd order coefficient',
                                    'c3 - 3rd order coefficient', 'c4 - 4th order coefficient',
                                    'c5 - energy_data offset, such that function value at energy_data = c5 is C0 '],
                     # Polynomial
                     'Exponential': ['Exponential function', 't0 - Normalization',
                                     't1 - Pseudo temperature'],  # Exponential
                     'Single Power Law Times an Exponential': ['Multiplication of Single Power Law and Exponential',

                                                               'p0 - normalization at epivot for power-law',
                                                               'p1 - negative power - law index',
                                                               'p2 - epivot (kEv) for power - law',
                                                               'e1 - normalization for exponential',
                                                               'e2 - pseudo temperature for exponential'],
                     # Single Power Law Times an Exponential
                     'Logistic Regression': ['Returns a sigmoid function'],  # Logistic Regression
                     'Lorentz': ['One dimensional Lorentzian model',
                                 'Amplitude correponds to peak value',
                                 'x_0 is the peak position (default value is 0)'],  # Lorentz Model
                     'Moffat': ['able to accurately reconstruct point spread functions',
                                'Moffat distribution'],  # Moffat model
                     'Voigt Profile': ['model computes the sum of Voigt function with a 2nd order polynomial',
                                       'amplitude centered at x_0 with the specified Lorentzian and Gaussian widths'],
                     # Voigt
                     'V_TH': ['Thermal Bremsstrahlung Model',
                              'T - Temperature (keV)',
                              'EM - Emission Measure (cm^-3)'],
                     'V_TH + PowerLaw': ['Addition of V_TH and Single Power Law',
                                         'T - Temperature (keV)',
                                         'EM - Emission Measure (cm^-3)',
                                         'Amplitude - Model amplitude at the reference energy',
                                         'Alpha - Power law index',
                                         'Epivot – energie pivot (kEv)'
                                         ],
                     'Neural Network': ['Neural Network model', ],
                     'PowerLawCutoffFix': ['Power law model with fix cutoff',
                                           'amplitude – model amplitude at the reference energy',
                                           'Ec – Cutoff energy',
                                           'alpha – power law index'
                                           ],
                     'PowerLawCutoffFree': ['Power law model with free cutoff',
                                            'amplitude – model amplitude at the reference energy',
                                            'Ec – Cutoff energy',
                                            'alpha – power law index'
                                            ],
                     'V_TH + PowerLawCutoffFix': ['Mix of V_TH and Power Law with fix cutoff',
                                                  'T - Temperature (keV)',
                                                  'EM - Emission Measure (cm^-3)',
                                                  'Amplitude - Model amplitude at the reference energy',
                                                  'Alpha - Power law index',
                                                  'Ec – Cutoff energy',
                                                  'Epivot – energie pivot (kEv)'],

                     'V_TH + PowerLawCutoffFree': ['Mix of V_TH and Power Law with free cutoff',
                                                   'T - Temperature (keV)',
                                                   'EM - Emission Measure (cm^-3)',
                                                   'Amplitude - Model amplitude at the reference energy',
                                                   'Ec – Cutoff energy',
                                                   'Alpha - Power law index',
                                                   'Epivot – energie pivot (kEv)']

                     }

        self.list_selection = Listbox(self.top2, highlightcolor='red', bd=4)
        self.list_selection.place(relx=0.33, rely=0.15, relheight=0.45, relwidth=0.30)

        if background.BackgroundWindow.DATA_BKG_SELECTED:
            self.show_db_var.set(1)  # Set the checkbox to checked if background data is selected

    def open_file(self, file=None):
        """Reads the input data using Astropy library. It can be any extension. RHESSI .fits files are analysed. \n
        Parameters: \n
            file: if a file has already been opened previously (i.e. in background), automatically re-reads it instead
            of asking user to choose it again."""
        if file:
            self.name = file
        else:
            self.name = askopenfilename(initialdir=".",
                                        filetypes=(("FITS files", "*.fits"), ("All Files", "*.*")),
                                        title="Please Select Spectrum or Image File")
        self.text_filename.delete(0, 'end')
        Fitting.fname = self.name

        if self.name:  # If file has been chosen by user
            with fits.open(self.name) as hdul:
                self.hdul = hdul
                self.text_filename.insert(0, self.name)  # Displays the input file name in Entry box
            # Loading data
            data1 = self.load_data(self.name)
            self.times = data1['time']
            self.counts = data1['counts']
            self.counts_err = data1['counts_err']
            self.e_high_det = data1['e_high']
            self.e_low_det = data1['e_low']
            self.time_del = data1['timedel']
            self.update_energy_range()
        else:
            self.text_filename.insert(0, "No file chosen")

    def open_srm_file(self, file=None):
        """Reads the input data using Astropy library. It can be any extension. RHESSI .fits files are analysed. \n
        Parameters: \n
            file: if a file has already been opened previously (i.e. in background), automatically re-reads it instead
            of asking user to choose it again."""
        if file:
            self.name2 = file
        else:
            self.name2 = askopenfilename(initialdir=".",
                                         filetypes=(("FITS files", "*.fits"), ("All Files", "*.*")),
                                         title="Please Select Spectrum or Image File")
        self.text_filename2.delete(0, 'end')
        Fitting.rname = self.name2

        if self.name2:  # If file has been chosen by user
            with fits.open(self.name2) as hdul:
                self.hdul2 = hdul
                self.text_filename2.insert(0, self.name2)  # Displays the input file name in Entry box
            # Loading data
            data = self.load_srm_data(self.name2)
            self.e_low_true = data['ENERG_LO']
            self.e_high_true = data['ENERG_HI']
            self.matrix = data['MATRIX']
            self.update_energy_range()
        else:
            self.text_filename2.insert(0, "No file chosen")

    def update_energy_range(self):
        if self.e_low_det is None or self.e_high_det is None or self.matrix is None:
            return

        usable_channels = np.arange(min(self.matrix.shape[1], len(self.e_low_det)))

        e_low_det = self.e_low_det[usable_channels]
        e_high_det = self.e_high_det[usable_channels]

        e_low_values = sorted(set(e_low_det))
        e_high_values = sorted(set(e_high_det))
        e_high_values = [e for e in e_high_values if e != float('inf') and e != float('-inf')]
        e_low_values_int = [int(e) for e in e_low_values if e != 0]
        e_high_values_int = [int(e) for e in e_high_values]

        self.energy_min_var.set(min(e_low_values_int))
        self.energy_max_var.set(max(e_high_values_int))

        menu = self.energy_min2["menu"]
        menu.delete(0, "end")
        for val in e_low_values_int:
            menu.add_command(label=val, command=lambda v=val: self.energy_min_var.set(v))

        menu = self.energy_max2["menu"]
        menu.delete(0, "end")
        for val in e_high_values_int:
            menu.add_command(label=val, command=lambda v=val: self.energy_max_var.set(v))

    @staticmethod
    def load_data(file):
        """Reads the Data and Header contents from input file. Loads the input file choosen in 'Select Plotting' section.
        Returns respectively a table containing datas, energies, dates and channels.\n
        Parameters: \n
            file: contains the data in a fits file."""
        hdulist = fits.open(file)  # Reads the data
        hdulist.info()  # Displays the content of the read file

        result = {}

        for hdu in hdulist:
            if not hasattr(hdu, 'columns'):
                continue
            colnames = hdu.columns.names

            # Time & Timedel
            if 'time' in colnames and 'timedel' in colnames:
                result['time'] = hdu.data['time']
                result['timedel'] = hdu.data['timedel']

            # Counts
            if 'counts' in colnames:
                result['counts'] = hdu.data['counts']
            if 'counts_comp_err' in colnames or 'counts_err' in colnames:
                err_col = 'counts_comp_err' if 'counts_comp_err' in colnames else 'counts_err'
                result['counts_err'] = hdu.data[err_col]

            # Triggers (optionnel)
            if 'triggers' in colnames:
                result['triggers'] = hdu.data['triggers']

            # Energy bins
            if 'e_low' in colnames and 'e_high' in colnames:
                result['e_low'] = hdu.data['e_low']
                result['e_high'] = hdu.data['e_high']

            # Version info
            if 'obt_start' in colnames and 'obt_end' in colnames:
                result['obt_start'] = hdu.data['obt_start']
                result['obt_end'] = hdu.data['obt_end']

        # Vérifications de base
        required_keys = ['counts', 'counts_err', 'e_low', 'e_high', 'time', 'timedel']
        for key in required_keys:
            if key not in result:
                print(f"⚠️  Attention : {key} non trouvé dans le FITS.")
        return result

        # return hdulist[2].data, hdulist[3].data, hdulist[0].header, hdulist[3].header

    @staticmethod
    def load_srm_data(file):
        """Reads the Data and Header contents from input file. Loads the input file choosen in 'Select Plotting' section.
        Returns respectively a table containing datas, energies, dates and channels.\n
        Parameters: \n
            file: contains the data in a fits file."""
        hdulist = fits.open(file)  # Reads the data
        hdulist.info()  # Displays the content of the read file

        result = {}

        for hdu in hdulist:
            if not hasattr(hdu, 'columns'):
                continue
            colnames = hdu.columns.names

            # Matrix
            if 'MATRIX' in colnames:
                result['MATRIX'] = hdu.data['MATRIX']

            # Energy bins
            if 'ENERG_LO' in colnames and 'ENERG_HI' in colnames:
                result['ENERG_LO'] = hdu.data['ENERG_LO']
                result['ENERG_HI'] = hdu.data['ENERG_HI']

        # Vérifications de base
        required_keys = ['MATRIX', 'ENERG_LO', 'ENERG_HI']
        for key in required_keys:
            if key not in result:
                print(f"⚠️  Attention : {key} non trouvé dans le FITS.")
        return result

    def onSelect(self, event):
        """Affiche les infos de la fonction sélectionnée dans la Listbox de droite."""
        try:
            # Récupère l’index de la sélection
            selected_index = self.lbox.curselection()[0]
            selected_name = self.lbox.get(selected_index)
            self.fit_model = selected_name

            # Réactive temporairement la Listbox info
            self.list_selection.config(state='normal')
            self.list_selection.delete(0, END)

            # Récupère et insère les infos correspondantes
            if selected_name in self.list:
                for line in self.list[selected_name]:
                    self.list_selection.insert(END, line)
            else:
                self.list_selection.insert(END, "Aucune information disponible.")

            # Désactive la Listbox info (pour la rendre non cliquable)
            self.list_selection.config(state='disabled')

        except Exception as e:
            print("Erreur dans onSelect :", e)

    def ask_custom_yesno(title, message):
        win = Toplevel()
        win.title(title)
        win.resizable(False, False)
        win.grab_set()  # modal

        # Contenu
        Label(win, text=message, padx=20, pady=20, justify='center').pack()

        result = {"value": False}

        def on_yes():
            result["value"] = True
            win.destroy()

        button_frame = Frame(win)
        button_frame.pack(pady=10)

        Button(button_frame, text="Yes", width=10, command=on_yes).pack(side="left", padx=5)
        Button(button_frame, text="No", width=10, command=lambda: win.destroy()).pack(side="left", padx=5)

        # ✅ Center the window on the screen
        win.update_idletasks()
        w = win.winfo_width()
        h = win.winfo_height()
        ws = win.winfo_screenwidth()
        hs = win.winfo_screenheight()
        x = (ws // 2) - (w // 2)
        y = (hs // 2) - (h // 2)
        win.geometry(f'+{x}+{y}')

        win.wait_window()
        return result["value"]

    def ask_photon_axes_scale(self):
        """Ouvre une popup centrée pour choisir les échelles X et Y du graphe photonique."""

        def confirm():
            self.photon_xscale = x_choice.get()
            self.photon_yscale = y_choice.get()
            popup.destroy()

        popup = Toplevel(self.top2)
        popup.title("Photon Plot Axes")

        # Taille désirée
        window_width = 400
        window_height = 200

        # Calculer la position centrée
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        pos_x = int((screen_width / 2) - (window_width / 2))
        pos_y = int((screen_height / 2) - (window_height / 2))

        popup.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        popup.resizable(False, False)

        # Interface utilisateur
        Label(popup, text="Choose axes scale for photon model:", font=("Helvetica", 11, "bold")).pack(pady=10)

        Label(popup, text="X axis scale:").pack()
        x_choice = StringVar(popup)
        x_choice.set("log")
        OptionMenu(popup, x_choice, "linear", "log").pack()

        Label(popup, text="Y axis scale:").pack()
        y_choice = StringVar(popup)
        y_choice.set("log")
        OptionMenu(popup, y_choice, "linear", "log").pack()

        Button(popup, text="Confirm", command=confirm, bg="#4CAF50", fg="white").pack(pady=10)

    def on_background_check(self):
        if self.show_db_var.get():  # Si check activé
            if not background.BackgroundWindow.DATA_BKG_SELECTED:
                answer = Fitting.ask_custom_yesno(
                    "Background Not Selected",
                    "You have not yet generated the Background.\n"
                    "Would you like to open the Background window now?"
                )
                if answer:
                    # ✅ close current Fit Options window and open Background window
                    self.show_db_var.set(0)
                    self.top2.destroy()
                    background.BackgroundWindow()
                else:
                    # ✅ uncheck the checkbox
                    self.show_db_var.set(0)
                return

    def on_background_clicked(self):
        if self.show_db_var.get():
            if background.BackgroundWindow.DATA_BKG_SELECTED:
                answer = Fitting.ask_custom_yesno(
                    "Background already selected",
                    "A background has already been selected.\nWould you like to select a new one?"
                )
                if answer:
                    background.BackgroundWindow.DATA_BKG_SELECTED = False
                    self.top2.destroy()  # Close current Fit Options window
                    background.BackgroundWindow()  # Open new Background selection
                else:
                    self.show_db_var.set(1)  # Keep checkbox checked
            else:
                self.on_background_check()  # Original logic (first-time case)

    def _params_vector_to_model(self, model_template, param_names, vec):
        """
        Retourne une copie du modèle template avec par.value = vec[i] pour les param_names.
        """
        m = copy.deepcopy(model_template)
        for i, name in enumerate(param_names):
            if hasattr(m, name):
                try:
                    getattr(m, name).value = float(vec[i])
                except Exception:
                    pass
        return m

    def fit_unconstrained_then_bounded(self, model_template, x_fit, y_fit, y_err,
                                       param_names, bounds_map=None, initial_values=None):
        """
        1) Fit non-borné (LevMar) en partant des valeurs initiales fournies (initial_values ou celles du modèle).
        2) Si la solution non-bornée satisfait les bornes -> on la renvoie.
        3) Sinon -> on relance un fit borné (LevMar avec .min / .max appliqués).
        Retourne un modèle astropy (copie) correspondant à la solution choisie.
        """
        # Appliquer les valeurs initiales (sans poser de bornes)
        if initial_values:
            for pname, val in initial_values.items():
                if hasattr(model_template, pname):
                    try:
                        getattr(model_template, pname).value = float(val)
                    except Exception:
                        pass

        # 1) Fit non-borné
        try:
            fitted_nc = self.fitter(copy.deepcopy(model_template), x_fit, y_fit,
                                    weights=1.0 / (y_err + 1e-30))
        except Exception as e:
            print("⚠️ Unconstrained LevMar fit failed:", e)
            return copy.deepcopy(model_template)

        # extraire les valeurs non-bornées
        uncon_values = [getattr(fitted_nc, p).value for p in param_names]

        # 2) Pas de bornes fournies -> utiliser solution non-bornée
        if not bounds_map:
            return fitted_nc

        # 4) Sinon, fit borné via LevMarLSQFitter avec min/max
        bounded_model = copy.deepcopy(model_template)

        # appliquer les bornes sur chaque paramètre
        for pname in param_names:
            if hasattr(bounded_model, pname):
                par = getattr(bounded_model, pname)
                lo, hi = bounds_map.get(pname, (None, None))
                if lo is not None:
                    par.min = lo
                if hi is not None:
                    par.max = hi

                # Priorité : valeur utilisateur > valeur par défaut > fit non-borné
                if initial_values and pname in initial_values:
                    par.value = initial_values[pname]
                elif pname in Fitting.default_param_values.get(model_template.__class__.__name__, {}):
                    par.value = Fitting.default_param_values[model_template.__class__.__name__][pname]
                else:
                    par.value = uncon_values[param_names.index(pname)]

        # initialiser aux valeurs du fit non-borné
        for i, pname in enumerate(param_names):
            if hasattr(bounded_model, pname):
                getattr(bounded_model, pname).value = uncon_values[i]

        try:
            fitted_bounded = self.fitter(bounded_model, x_fit, y_fit,
                                         weights=1.0 / (y_err + 1e-30))
            return fitted_bounded
        except Exception as e:
            print("⚠️ Bounded LevMar fit failed:", e)
            return fitted_nc  # fallback

    def fit_with_bounds_check(self, model_template, x_fit, y_fit, y_err,
                              param_names, model_key,
                              internal_bounds_map=None, user_bounds_map=None,
                              initial_values=None):
        """
        Fit en deux étapes avec LevMarLSQFitter uniquement :
        1) Fit avec contraintes "internes" (par ex. default_param_bounds).
        2) Si le résultat respecte les bornes utilisateur → on garde.
        3) Sinon → refit avec bornes utilisateur appliquées.
        """
        # --- Maps par défaut ---
        if internal_bounds_map is None:
            internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
        if user_bounds_map is None:
            user_bounds_map = self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {}))
        if initial_values is None:
            initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

        # --- Étape 0 : appliquer les valeurs initiales ---
        try:
            for pname, val in initial_values.items():
                if hasattr(model_template, pname):
                    getattr(model_template, pname).value = float(val)
        except Exception:
            pass

        # --- Étape 1 : modèle avec bornes internes ---
        model_step1 = copy.deepcopy(model_template)
        for pname in param_names:
            if hasattr(model_step1, pname):
                lo, hi = internal_bounds_map.get(pname, (None, None))
                par = getattr(model_step1, pname)
                try:
                    if lo is not None:
                        par.min = lo
                    if hi is not None:
                        par.max = hi
                except Exception:
                    pass

        try:
            fitted1 = self.fitter(model_step1, x_fit, y_fit, weights=1.0 / (y_err + 1e-30))
            y_model1 = fitted1(x_fit)
            if not np.all(np.isfinite(y_model1)):
                raise ValueError("Non-finite model output at step1")
        except Exception as e_step1:
            print(f"⚠️ Step1 LevMar fit failed: {e_step1}")
            return self._fit_with_user_bounds_only(model_template, x_fit, y_fit, y_err, param_names, user_bounds_map,
                                                   initial_values)

        # --- Vérifier si fitted1 est dans les bornes utilisateur ---
        tol = 1e-12
        in_user_bounds = True
        for pname in param_names:
            if not hasattr(fitted1, pname):
                continue
            attr = getattr(fitted1, pname)
            if isinstance(attr, (int, float, np.floating)):
                # c'est un paramètre fixe → ignorer
                continue
            val = attr.value
            lo, hi = user_bounds_map.get(pname, (None, None))
            if lo is not None and val < lo - tol:
                in_user_bounds = False
                break
            if hi is not None and val > hi + tol:
                in_user_bounds = False
                break

        if in_user_bounds:
            return fitted1

        # --- Étape 2 : refit avec bornes utilisateur ---
        return self._fit_with_user_bounds_only(model_template, x_fit, y_fit, y_err, param_names, user_bounds_map,
                                               initial_values)

    def _fit_with_user_bounds_only(self, model_template, x_fit, y_fit, y_err,
                                   param_names, user_bounds_map, initial_values):
        """Fit avec LevMarLSQFitter en appliquant uniquement les bornes utilisateur."""
        model_bounded = copy.deepcopy(model_template)

        # Appliquer initial values et bornes utilisateur
        for pname in param_names:
            if hasattr(model_bounded, pname):
                val = initial_values.get(pname, getattr(model_bounded, pname).value)
                lo, hi = user_bounds_map.get(pname, (None, None))
                par = getattr(model_bounded, pname)
                try:
                    par.value = float(val)
                    if lo is not None:
                        par.min = lo
                    if hi is not None:
                        par.max = hi
                except Exception:
                    pass

        try:
            fitted2 = self.fitter(model_bounded, x_fit, y_fit, weights=1.0 / (y_err + 1e-30))
            return fitted2
        except Exception as e_step2:
            print(f"⚠️ Step2 LevMar (user bounds) failed: {e_step2}")
            return model_bounded

    def _selective_fit(self):
        selection = self.lbox.curselection()
        if not selection:
            messagebox.showwarning("No Model Selected",
                                   "Please select a fit model before clicking 'Do Fit'.")
            return

        if Fitting.fname is None and Fitting.rname is None:
            messagebox.showwarning("No File Selected", "Please, choose input file.")
            return

        # ── Préparation des données ────────────────────────────────
        if self.show_db_var.get():
            idx_s = background.BackgroundWindow.DATA_BKG_START
            idx_e = background.BackgroundWindow.DATA_BKG_END
            bkg = np.mean(self.counts[idx_s:idx_e + 1, :], axis=0)
            raw = np.where(self.counts - bkg > 0, self.counts - bkg, 1e-5)
            absolute_name = "Data - Background"
        else:
            raw = self.counts
            absolute_name = "Data"

        counts_all = np.mean(raw, axis=0)
        counts_err_all = np.mean(self.counts_err, axis=0)
        exposure = float(np.mean(self.time_del))

        e_low_true = self.e_low_true
        e_high_true = self.e_high_true
        matrix = self.matrix

        usable = np.arange(min(matrix.shape[1], len(self.e_low_det)))
        counts = counts_all[usable]
        counts_err = counts_err_all[usable]
        e_low_det = self.e_low_det[usable]
        e_high_det = self.e_high_det[usable]

        valid = (counts_err > 0) & np.isfinite(counts_err) & np.isfinite(counts)
        counts = counts[valid]
        counts_err = counts_err[valid]
        matrix = matrix[:, valid]
        e_low_det = e_low_det[valid]
        e_high_det = e_high_det[valid]

        edges_det = np.append(e_low_det, e_high_det[-1])
        dE_det = np.diff(edges_det)
        Edges_photon = np.append(e_low_true, e_high_true[-1])

        fit_Emin = self.energy_min_var.get()
        fit_Emax = self.energy_max_var.get()
        fit_mask = (edges_det[:-1] >= fit_Emin) & (edges_det[1:] <= fit_Emax)

        x_fit = np.zeros(fit_mask.sum())
        counts_fit = counts[fit_mask]
        counts_err_fit = counts_err[fit_mask]
        matrix_fit = matrix[:, fit_mask]
        x_fake = np.zeros_like(counts)

        y_fit = counts_fit / exposure
        y_err = counts_err_fit / exposure

        # ── Unités ────────────────────────────────────────────────
        rate = counts / exposure
        rate_err = counts_err / exposure
        flux = rate / (self.area * dE_det)
        flux_err = rate_err / (self.area * dE_det)

        unit = self.var.get()
        unit_map = {
            'Rate': (rate, rate_err, "Rate [counts / (s keV)]"),
            'Counts': (counts, counts_err, "Counts"),
            'Flux': (flux, flux_err, "Flux (Counts/s/cm²/keV)"),
        }
        y_data, y_err_data, y_label = unit_map[unit]

        def to_unit(rate_model):
            if unit == 'Rate':   return rate_model / dE_det
            if unit == 'Counts': return rate_model * exposure
            if unit == 'Flux':   return rate_model / (self.area * dE_det)

        def add_param_text(text):
            plt.text(0.05, 0.4, text,
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.7))

        def finalize_main_plot():
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(fit_Emin, fit_Emax)
            plt.xlabel("Channel Energy (keV)")
            plt.ylabel(y_label)
            plt.title(f"Fitting on [{fit_Emin}, {fit_Emax}] keV using {self.statname}")
            if self.grid_var.get():
                plt.grid(True, which="both", ls="--", alpha=0.5)
            else:
                plt.grid(False)
            plt.legend()
            plt.tight_layout()

        def plot_photon(model_func, param_text):
            flux_photons = np.array([
                integrate_flux(e1, e2, model_func)
                for e1, e2 in zip(e_low_true, e_high_true)
            ])
            plt.figure()
            plt.step(Edges_photon[:-1], flux_photons, where='post',
                     label='Photon model', color='green')
            xscale = getattr(self, "photon_xscale", "log")
            yscale = getattr(self, "photon_yscale", "log")
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Photon flux [photons / (s cm² keV)]")
            plt.title(f"Photon Flux Model using {self.statname}")
            plt.legend()
            if self.show_params_var.get():
                add_param_text(param_text)
            plt.tight_layout()

        # ── Plot données ───────────────────────────────────────────
        plt.figure()
        plt.step(edges_det[:-1], y_data, where='post',
                 label=f'{absolute_name} ({unit})', color='red')

        idx = self.lbox.curselection()[0]

        # ══════════════════════════════════════════════════════════
        #  0 — Power Law
        # ══════════════════════════════════════════════════════════
        if idx == 0:
            model_key = "PowerLaw1D"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            model_template = PowerLaw(
                e_low_true, e_high_true, matrix_fit, exposure, E_pivot=E_pivot_val)
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values)

            amplitude, alpha = fitted.amplitude.value, fitted.alpha.value

            model_display = PowerLaw(
                e_low_true, e_high_true, matrix, exposure, E_pivot=E_pivot_val)
            model_display.amplitude.value = amplitude
            model_display.alpha.value = alpha

            model_y = to_unit(model_display(x_fake))
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Power Law:\n amplitude = {amplitude:.2e}\n"
                          f" alpha = {alpha:.2f}\n E_pivot = {E_pivot_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: amplitude * (E / E_pivot_val) ** (-alpha), param_text)

        # ══════════════════════════════════════════════════════════
        #  1 — Broken Power Law
        # ══════════════════════════════════════════════════════════
        elif idx == 1:
            model_key = "BrokenPowerLaw1D"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))

            model_template = BrokenPowerLaw(
                e_low_true, e_high_true, matrix_fit, exposure)
            fitted = self.fit_with_bounds_check(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "E_break", "alpha_1", "alpha_2"], model_key,
                initial_values=initial_values)

            amplitude = fitted.amplitude.value
            E_break = fitted.E_break.value
            alpha_1 = fitted.alpha_1.value
            alpha_2 = fitted.alpha_2.value

            model_display = BrokenPowerLaw(
                e_low_true, e_high_true, matrix, exposure)
            for p in ["amplitude", "E_break", "alpha_1", "alpha_2"]:
                getattr(model_display, p).value = getattr(fitted, p).value

            model_y = to_unit(model_display(x_fake))
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Broken Power Law:\n amplitude = {amplitude:.2e}\n"
                          f" E_break = {E_break:.2f}\n Alpha_1 = {alpha_1:.2e}\n"
                          f" Alpha_2 = {alpha_2:.2f}")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: amplitude * np.where(
                    E < E_break,
                    (E / E_break) ** (-alpha_1),
                    (E / E_break) ** (-alpha_2)), param_text)

        # ══════════════════════════════════════════════════════════
        #  2 — Exp Power Law
        # ══════════════════════════════════════════════════════════
        elif idx == 2:
            model_key = "Single Power Law Times an Exponential"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))

            model_template = ExpPowerLaw(
                e_low_true, e_high_true, matrix_fit, exposure)
            fitted = self.fit_with_bounds_check(
                model_template, x_fit, y_fit, y_err,
                ["p0", "p1", "p2", "e3", "e4"], model_key,
                initial_values=initial_values)

            p0, p1, p2, e3, e4 = (fitted.p0.value, fitted.p1.value, fitted.p2.value,
                                  fitted.e3.value, fitted.e4.value)

            model_display = ExpPowerLaw(
                e_low_true, e_high_true, matrix, exposure)
            for p in ["p0", "p1", "p2", "e3", "e4"]:
                getattr(model_display, p).value = getattr(fitted, p).value

            model_y = to_unit(model_display(x_fake))
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Exp Power Law:\n p0={p0:.2e} p1={p1:.2f}\n"
                          f" p2={p2:.2f} e3={e3:.2f} e4={e4:.2f}")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: (p0 * (E / p2) ** p1) * np.exp(e3 - E / e4),
                            param_text)

        # ══════════════════════════════════════════════════════════
        #  3 — V_TH
        # ══════════════════════════════════════════════════════════
        elif idx == 3:
            model_key = "V_TH"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))

            model_template = VTH(
                e_low_true, e_high_true, matrix_fit, exposure)
            fitted = self.fit_with_bounds_check(
                model_template, x_fit, y_fit, y_err,
                ["EM", "T"], model_key, initial_values=initial_values)

            T, EM = fitted.T.value, fitted.EM.value

            model_display = VTH(e_low_true, e_high_true, matrix, exposure)
            model_display.T.value = T
            model_display.EM.value = EM

            model_y = to_unit(model_display(x_fake))
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted VTH Model', color='blue')

            param_text = f"V_TH:\n T = {T:.2f} keV\n EM = {EM:.2e} cm⁻³"
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: (1.07e-42 * 1.2 * EM)
                                      / (E * np.sqrt(max(1e-3, T))) * np.exp(-E / T), param_text)

        # ══════════════════════════════════════════════════════════
        #  4 — V_TH + Power Law
        # ══════════════════════════════════════════════════════════
        elif idx == 4:
            model_key = "V_TH + PowerLaw"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            model_template = VTHPlusPowerLaw(
                e_low_true, e_high_true, matrix_fit, exposure, E_pivot=E_pivot_val)
            fitted = self.fit_with_bounds_check(
                model_template, x_fit, y_fit, y_err,
                ["EM", "T", "amplitude", "alpha"], model_key,
                initial_values=initial_values)

            EM, T, amplitude, alpha = (fitted.EM.value, fitted.T.value,
                                       fitted.amplitude.value, fitted.alpha.value)

            model_display = VTHPlusPowerLaw(
                e_low_true, e_high_true, matrix, exposure, E_pivot=E_pivot_val)
            for p in ["EM", "T", "amplitude", "alpha"]:
                getattr(model_display, p).value = getattr(fitted, p).value

            model_y = to_unit(model_display(x_fake))
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"V_TH + Power Law:\n T={T:.2e} keV\n EM={EM:.2e} cm⁻³\n"
                          f" amplitude={amplitude:.2e}\n alpha={alpha:.2f}\n"
                          f" E_pivot={E_pivot_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: (
                        (1.07e-42 * 1.2 * EM) / (E * np.sqrt(max(1e-3, T))) * np.exp(-E / T)
                        + amplitude * (E / 100.0) ** (-alpha)), param_text)

        # ══════════════════════════════════════════════════════════
        #  5 — Power Law Cutoff Fix
        # ══════════════════════════════════════════════════════════
        elif idx == 5:
            model_key = "PowerLawCutoffFix"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_cut_val = initial_values.get("E_cut", 10.0)
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            model_template = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix_fit, exposure, E_cut_val, E_pivot_val)
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values)

            amplitude, alpha = fitted.amplitude.value, fitted.alpha.value

            # E_cut_grid = np.linspace(5, 20, 15)
            # chi2_values = []
            #
            # for ec in E_cut_grid:
            #     model_ec = ForwardFolded.PowerLawCutoffFix(
            #         e_low_true, e_high_true, matrix_fit, exposure, ec, E_pivot_val)
            #     fitted_ec = self.fit_unconstrained_then_bounded(
            #         model_ec, x_fit, y_fit, y_err,
            #         ["amplitude", "alpha"], bounds_map, initial_values)
            #     chi2 = np.sum(((y_fit - fitted_ec(x_fit)) / (y_err + 1e-30)) ** 2)
            #     chi2_values.append(chi2)
            #
            # chi2_values = np.array(chi2_values)
            #
            # plt.figure()
            # plt.plot(E_cut_grid, chi2_values, 'o-', color='steelblue', label='χ²(E_cut)')
            # plt.axvline(E_cut_val, color='red', linestyle='--',
            #             label=f'E_cut utilisé = {E_cut_val:.2f} keV')
            # plt.axhline(chi2_values.min(), color='orange', linestyle='--',
            #             label=f'χ² min = {chi2_values.min():.3f}')
            # plt.xlabel("E_cut (keV)")
            # plt.ylabel("χ²")
            # plt.title("χ²(E_cut) — PowerLaw Cutoff Fix")
            # plt.legend()
            # plt.grid(True, which="both", ls="--", alpha=0.5)
            # plt.tight_layout()

            model_display = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix, exposure, E_cut_val, E_pivot_val)
            model_display.amplitude.value = amplitude
            model_display.alpha.value = alpha

            model_y = to_unit(model_display(x_fake))

            # Votre masque cutoff original
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, 0)

            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Power Law Cutoff Free:\n amplitude={amplitude:.2e}\n"
                          f" alpha={alpha:.2f}\n E_pivot={E_pivot_val:.2f} keV\n"
                          f" E_cut={E_cut_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: np.where(
                    E >= E_cut_val,
                    amplitude * (E / E_pivot_val) ** (-alpha), 0.0), param_text)


        # ══════════════════════════════════════════════════════════
        #  6 — Power Law Cutoff Free
        # ══════════════════════════════════════════════════════════
        elif idx == 6:
            model_key = "PowerLawCutoffFree"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_cut_bound = (initial_values.get("Ec_min", fit_Emin), initial_values.get("Ec_max", fit_Emax))
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            model_template = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix_fit, exposure, E_pivot_val)

            def chi2_for_Ecut(E_cut_val):
                print(E_cut_val)
                model_template.E_cut = E_cut_val
                fitted_model = self.fit_unconstrained_then_bounded(
                    model_template, x_fit, y_fit, y_err,
                    ["amplitude", "alpha"], bounds_map, initial_values
                )
                return np.sum(((y_fit - fitted_model(x_fit)) / (y_err + 1e-30)) ** 2)

            result = minimize_scalar(
                chi2_for_Ecut,
                bounds=E_cut_bound,
                method="bounded",
                options={"xatol":1e-3}
            )

            E_cut_val = result.x

            # Récupérer le modèle final au E_cut optimal
            model_template.E_cut = E_cut_val
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values
            )

            amplitude = fitted.amplitude.value
            alpha = fitted.alpha.value

            model_display = ForwardFolded.PowerLawCutoffFix(e_low_true, e_high_true, matrix, exposure)
            for p in ["amplitude", "alpha"]:
                getattr(model_display, p).value = getattr(fitted, p).value
            model_display.E_cut = E_cut_val

            model_y = to_unit(model_display(x_fake))

            # Votre masque cutoff original
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, 0)

            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Power Law Cutoff Free:\n amplitude={amplitude:.2e}\n"
                          f" alpha={alpha:.2f}\n E_pivot={E_pivot_val:.2f} keV\n"
                          f" E_cut={E_cut_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                plot_photon(lambda E: np.where(
                    E >= E_cut_val,
                    amplitude * (E / E_pivot_val) ** (-alpha), 0.0), param_text)

        # ══════════════════════════════════════════════════════════
        #  7 — V_TH + Cutoff Fix
        # ══════════════════════════════════════════════════════════
        elif idx == 7:
            model_key = "PowerLawCutoffFix"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_cut_val = initial_values.get("E_cut", 10.0)
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            model_template = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix_fit, exposure, E_cut_val, E_pivot_val)
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values)

            amplitude, alpha = fitted.amplitude.value, fitted.alpha.value

            model_display = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix, exposure, E_cut_val, E_pivot_val)
            model_display.amplitude.value = amplitude
            model_display.alpha.value = alpha

            model_y = to_unit(model_display(x_fake))

            # Votre masque cutoff original
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, np.nan)

            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            fit_mask = (edges_det[:-1] >= fit_Emin) & (edges_det[1:] <= E_cut_val)

            x_fit = np.zeros(fit_mask.sum())
            matrix_fit = matrix[:, fit_mask]
            x_fake = np.zeros_like(counts)

            model_key = "V_TH"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))

            model_template = VTH(
                e_low_true, e_high_true, matrix_fit, exposure)
            fitted = self.fit_with_bounds_check(
                model_template, x_fit, y_fit, y_err,
                ["EM", "T"], model_key, initial_values=initial_values)

            T, EM = fitted.T.value, fitted.EM.value

            model_display = VTH(e_low_true, e_high_true, matrix, exposure)
            model_display.T.value = T
            model_display.EM.value = EM

            model_y = to_unit(model_display(x_fake))
            fit_mask_cutoff = (edges_det[:-1] >= fit_Emin) & (edges_det[:-1] < E_cut_val)
            model_y = np.where(fit_mask_cutoff, model_y, np.nan)
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted VTH Model', color='purple')

            param_text = (f"V_TH + Power Law Cutoff Fix:\n"
                          f" T={T:.2f} keV  EM={EM:.2e} cm⁻³\n"
                          f" amplitude={amplitude:.2e}  alpha={alpha:.2f}\n"
                          f" E_pivot={E_pivot_val:.2f} keV  E_cut={E_cut_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)

            if self.show_photon_var.get():
                gff = 1.2
                A = 1.07e-42 * gff
                safe_T = max(1e-3, T)

                def model_total(E):
                    # Thermal component
                    thermal = (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T)
                    # Power-law component
                    power = np.where(E >= E_cut_val, amplitude * (E / E_pivot_val) ** (-alpha), 0.0)
                    return thermal + power

                # Power-law component
                plot_photon(model_total, param_text)

        # ══════════════════════════════════════════════════════════
        #  8 — V_TH + Cutoff Free
        # ══════════════════════════════════════════════════════════
        elif idx == 8:
            model_key = "PowerLawCutoffFree"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            y_fit = y_fit
            y_err = y_err

            model_template = PowerLawCutoffFree(
                e_low_true, e_high_true, matrix_fit, exposure, E_pivot_val)
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values)

            amplitude = fitted.amplitude.value
            alpha = fitted.alpha.value
            E_cut_val = fitted.E_cut.value

            model_display = PowerLawCutoffFree(
                e_low_true, e_high_true, matrix, exposure, E_pivot_val)
            model_display.amplitude.value = amplitude
            model_display.alpha.value = alpha

            model_y = to_unit(model_display(x_fake))

            # Votre masque cutoff original
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, np.nan)

            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            fit_mask = (edges_det[:-1] >= fit_Emin) & (edges_det[1:] <= E_cut_val)

            x_fit = np.zeros(fit_mask.sum())
            matrix_fit = matrix[:, fit_mask]
            x_fake = np.zeros_like(counts)

            model_key = "V_TH"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))

            model_template = VTH(
                e_low_true, e_high_true, matrix_fit, exposure)
            fitted = self.fit_with_bounds_check(
                model_template, x_fit, y_fit, y_err,
                ["EM", "T"], model_key, initial_values=initial_values)

            T, EM = fitted.T.value, fitted.EM.value

            model_display = VTH(e_low_true, e_high_true, matrix, exposure)
            model_display.T.value = T
            model_display.EM.value = EM

            model_y = to_unit(model_display(x_fake))
            fit_mask_cutoff = (edges_det[:-1] >= fit_Emin) & (edges_det[:-1] < E_cut_val)
            model_y = np.where(fit_mask_cutoff, model_y, np.nan)
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted VTH Model', color='purple')

            param_text = (f"V_TH + Power Law Cutoff Fix:\n"
                          f" T={T:.2f} keV  EM={EM:.2e} cm⁻³\n"
                          f" amplitude={amplitude:.2e}  alpha={alpha:.2f}\n"
                          f" E_pivot={E_pivot_val:.2f} keV  E_cut={E_cut_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)

            if self.show_photon_var.get():
                gff = 1.2
                A = 1.07e-42 * gff
                safe_T = max(1e-3, T)

                def model_total(E):
                    # Thermal component
                    thermal = (A * EM) / (E * np.sqrt(safe_T)) * np.exp(-E / safe_T)
                    # Power-law component
                    power = np.where(E >= E_cut_val, amplitude * (E / E_pivot_val) ** (-alpha), 0.0)
                    return thermal + power

                # Power-law component
                plot_photon(model_total, param_text)

        finalize_main_plot()
        plt.show()
