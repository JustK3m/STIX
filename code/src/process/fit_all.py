import copy
import os.path
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.optimize import minimize_scalar
import numpy as np

from . import background
from .fitting.fitters import LevMarCstatFitter
from .fitting.methods import *
from .io import get_data, get_srm_data, loader

register_matplotlib_converters()


class Fitting:
    """
    Main class for the spectral fitting window ('SPEX Fit Options').

    Handles loading the FITS files (spectrum and response matrix),
    model selection, parameter configuration, running the forward-folding
    fit, and displaying the results.

    Class attributes
    ----------------
    fname : str
        Default path to the spectrum FITS file.
    rname : str
        Default path to the SRM FITS file.
    default_param_bounds : dict
        Default bounds {model_name: {param: (min, max)}} for each
        available model.
    default_param_values : dict
        Default initial values {model_name: {param: value}}.
    """

    # ── Default bounds ──────────────────────────────────────
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
        "V_TH x PowerLawCutoffFix": {"EM": (1e44, 1e52), "T": (0.1, 50.0),
                                     "amplitude": (1e-12, 1e6), "alpha": (0.1, 50.0)},
        "V_TH x PowerLawCutoffFree": {"EM": (1e44, 1e52), "T": (0.1, 50.0),
                                      "amplitude": (1e-12, 1e6), "alpha": (0.1, 50.0)},
        "Neural Network":{},
    }

    # ── Default initial values ───────────────────────────
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
                               "Ec_min": 4, "Ec_max": 20},
        "V_TH x PowerLawCutoffFix": {"EM": 1e48, "T": 1.0,
                                     "amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0, "E_cut": 10},
        "V_TH x PowerLawCutoffFree": {"EM": 1e48, "T": 1.0,
                                      "amplitude": 1e-2, "alpha": 2.0, "E_pivot": 100.0,
                                      "Ec_min": 4, "Ec_max": 20},
        "Neural Network": {},
    }

    # create a new window called 'SPEX Fit Options'
    def __init__(self, root):
        """
        Creates the 'SPEX Fit Options' window and initialises all the
        widgets (model listbox, file fields, energy menus, buttons)
        as well as the internal state attributes.

        Parameters
        ----------
        root : tk.Tk or tk.Toplevel
            Parent tkinter window.

        Key instance attributes
        -----------------------
        counts : ndarray, shape (T, C)
            Raw counts matrix (time steps x channels).
        counts_err : ndarray, shape (T, C)
            Associated error matrix.
        times : array-like, shape (T,)
            Time of each step (s since epoch).
        time_del : array-like, shape (T,)
            Duration of each time step (s).
        e_low_det, e_high_det : ndarray, shape (C,)
            Bounds of the channels measured by the detector (keV).
        e_low_true, e_high_true : ndarray, shape (N,)
            Bounds of the SRM's true-energy bins (keV).
        matrix : ndarray, shape (N, M)
            Instrument response matrix (SRM).
        fitter : astropy fitter
            Active fitter instance (LevMarLSQFitter by default,
            LevMarCstatFitter if C-stat is selected).
        user_param_values : dict
            Initial values set by the user via Set Function.
        user_param_bounds : dict
            Bounds set by the user via Set Function.
        """
        self.sender = None

        self.top2 = Toplevel()
        self.top2.title('SPEX Fit Options')  # title of the window
        self.top2.geometry("1000x600")  # size of the new window

        self.fname = loader.activeFile()
        self.rname = loader.activeSRMfile()  # # Name of the .fits file imported (response matrix)

        self.counts = None  # Matrix contaning the counts per band in function of time time
        self.counts_err = None  # Matrix contaning the error of the counts per band in function of time
        self.times = None  # Index of times for x axis
        self.time_del = None  # Time delay for the data
        self.e_low_det = None
        self.e_high_det = None

        self.area = 6  # Area of the surface of detection of the telescope in cm²; used for the flux

        self.e_low_true = None
        self.e_high_true = None
        self.matrix = None

        self.fitter = LevMarLSQFitter()

        self.energy_min_var = tk.DoubleVar(value=0)
        self.energy_min2 = tk.OptionMenu(self.top2, self.energy_min_var, 0)

        self.energy_max_var = tk.DoubleVar(value=0)
        self.energy_max2 = tk.OptionMenu(self.top2, self.energy_max_var, 0)

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

        if self.fname:
            self.text_filename.insert(0, self.fname)
            self.open_file(self.fname)
        else:
            self.text_filename.insert(0, "No file chosen")

        Button(self.top2, text='Browse ->', command=self.open_file).place(relx=0.92, rely=0.2, anchor=W)

        # Response matrix: file name
        Label(self.top2, text="Response: ").place(relx=0.65, rely=0.25, anchor=W)
        self.text_filename2 = Entry(self.top2, width=30)
        self.text_filename2.place(relx=0.72, rely=0.25, anchor=W)
        if self.rname:
            self.text_filename2.insert(0, self.rname)
            self.open_srm_file(self.rname)
        else:
            self.text_filename2.insert(0, "No file chosen")

        Button(self.top2, text='Browse ->', command=self.open_srm_file).place(relx=0.92, rely=0.25, anchor=W)

        self.user_param_bounds = {}  # bounds set by user in Set_Function
        self.user_param_values = {}  # initial values set by user in Set_Function
        self.user_param_modified = {}  # True if user modified bounds/values from default

        Label(self.top2, text="Set function components: ").place(relx=0.65, rely=0.30)

        Button(self.top2, text="Function value(s)", command=self.Set_Function).place(relx=0.65, rely=0.35, relheight=0.05,
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

        Label(self.top2, text="Min energy").place(relx=0.75, rely=0.45, anchor=N)
        Label(self.top2, text="Max energy").place(relx=0.85, rely=0.45, anchor=N)

        self.energy_min2 = OptionMenu(self.top2, self.energy_min_var, [0])
        self.energy_max2 = OptionMenu(self.top2, self.energy_max_var, [0])

        self.energy_min2.place(relx=0.75, rely=0.50, anchor=N)
        self.energy_max2.place(relx=0.85, rely=0.50, anchor=N)

        # ============== Main window description ==============
        """ 
        On the left side of the 'SPEX Fit Options' window: place a list of text alternatives (listbox).
        The user can choose(highlight) one of the options.
        Options(functions):
        1) One Dimensional Power Law;
        2) 1-D Broken Power Law;
        3) Single Power Law Times an Exponetial
        """
        self.lbox = Listbox(self.top2, selectmode=EXTENDED, highlightcolor='red', bd=4, selectbackground='grey')
        self.lbox.place(relx=0.05, rely=0.15, relheight=0.45, relwidth=0.25)

        self.scroll = Scrollbar(self.top2, command=self.lbox.yview)
        self.scroll.place(relx=0.3, rely=0.15, relheight=0.45, relwidth=0.02)
        self.lbox.config(yscrollcommand=self.scroll.set)

        # New frame at the bottom. Locate there 'Plot Units' and 'Do Fit' widgets
        self.frameFit = LabelFrame(self.top2, relief=RAISED,
                                   borderwidth=10)  # determine the border of the frame and size
        self.frameFit.place(relx=0.05, rely=0.63, relheight=0.25, relwidth=0.85)  # the frame position

        Label(self.frameFit, text="Plot Units: ", fg='blue',
              font=("Helvetica", 11, "bold")).place(relx=0.04, rely=0.4)

        # Add button for Units: Rate, Counts, Flux
        # Allows user to make a choice between three parameters
        self.Component_choicesFit = ('Rate', 'Counts', 'Flux')
        self.var = StringVar(self.frameFit)
        self.var.set(self.Component_choicesFit[0])
        OptionMenu(self.frameFit, self.var, *self.Component_choicesFit).place(relx=0.15, rely=0.38, relheight=0.23,
                                                                              relwidth=0.15)

        self.show_params_var = IntVar(value=1)  # Checked by default
        Checkbutton(
            self.frameFit,
            text="Display parameters",
            variable=self.show_params_var
        ).place(relx=0.35, rely=0.7)

        self.grid_var = IntVar(value=0)
        Checkbutton(
            self.frameFit,
            text="Show grid",
            variable=self.grid_var
        ).place(relx=0.55, rely=0.7)

        self.show_db_var = IntVar(value=0)
        Checkbutton(
            self.frameFit,
            text="Data-Background",
            variable=self.show_db_var,
            command=self.on_background_clicked
        ).place(relx=0.35, rely=0.5)

        self.show_photon_var = IntVar(value=0)

        def on_photon_toggle():
            if self.show_photon_var.get():
                self.ask_photon_axes_scale()

        Checkbutton(
            self.frameFit,
            text="Photon",
            variable=self.show_photon_var,
            command=on_photon_toggle
        ).place(relx=0.55, rely=0.5)

        Button(self.frameFit, text="Do Fit",
               command=self._selective_fit).place(relx=0.70, rely=0.20, relheight=0.23, relwidth=0.15)  # locate

        Button(self.frameFit, text="Close Plots", command=lambda: plt.close('all')).place(relx=0.70, rely=0.60,
                                                                                          relheight=0.27, relwidth=0.15)

        Button(self.top2, text="Refresh").place(relx=0.4, rely=0.94)

        """Scrollbar with information related to each function"""
        Button(self.top2, text="Close", command=lambda: self.top2.destroy()).place(relx=0.5, rely=0.94)
        self.models = ['PowerLaw1D', 'BrokenPowerLaw1D', 'Single Power Law Times an Exponential', 'V_TH',
                       'V_TH + PowerLaw', 'PowerLawCutoffFix', 'PowerLawCutoffFree',
                       'V_TH x PowerLawCutoffFix', "V_TH x PowerLawCutoffFree", "Neural Network"]  # , 'Neural Network' function names
        for p in self.models:
            """On the right: place an 'entry text' Scrollbar widget (scrollbar) When user highlight the function, 
            displays the text information about function description and input parameters"""
            self.lbox.insert(END, p)
        self.lbox.bind("<<ListboxSelect>>", self.onSelect)
        self.list = {
            'PowerLaw1D': ['One dimensional power law model',
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
            'V_TH x PowerLawCutoffFix': ['Mix of V_TH and Power Law with fix cutoff',
                                         'T - Temperature (keV)',
                                         'EM - Emission Measure (cm^-3)',
                                         'Amplitude - Model amplitude at the reference energy',
                                         'Alpha - Power law index',
                                         'Ec – Cutoff energy',
                                         'Epivot – energie pivot (kEv)'],

            'V_TH x PowerLawCutoffFree': ['Mix of V_TH and Power Law with free cutoff',
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

    def Set_Function(self):
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

        # Parameters with no min/max (value only)
        VALUE_ONLY_PARAMS = {"E_pivot", "E_cut", "Ec_min", "Ec_max"}
        VALUE_ONLY_MODELS = {
            "PowerLaw1D", "V_TH + PowerLaw", "PowerLawCutoffFix",
            "PowerLawCutoffFree", "V_TH x PowerLawCutoffFix", "V_TH x PowerLawCutoffFree",
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

            # Retrieve the entries just created (the last 3 in row)
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

    def open_file(self, file=None):
        """
        Opens and reads a STIX spectrum FITS file. If a path is
        provided, loads it directly; otherwise opens a file dialog.

        Updates: self.times, self.counts, self.counts_err,
        self.e_low_det, self.e_high_det, self.time_del, self.fname.
        Calls self.update_energy_range() after loading.

        Parameters
        ----------
        file : str or None, optional
            Full path to the FITS file. If None, a file
            dialog is opened.

        Returns
        -------
        None
        """
        if file:
            self.fname = file
        else:
            self.fname = askopenfilename(initialdir=".",
                                        filetypes=(("FITS files", "*.fits"), ("All Files", "*.*")),
                                        title="Please Select Spectrum or Image File")
        self.text_filename.delete(0, 'end')

        if self.fname:
            self.text_filename.insert(0, self.fname)  # Displays the input file name in Entry box
            # Loading data
            data = get_data(self.fname)
            self.times = data['time']
            self.counts = data['counts']
            self.counts_err = data['counts_err']
            self.e_high_det = data['e_high']
            self.e_low_det = data['e_low']
            self.time_del = data['timedel']
            self.update_energy_range()
        else:
            self.text_filename.insert(0, "No file chosen")

    def open_srm_file(self, file=None):
        """
        Updates the energy selection dropdown menus (min/max) based
        on the channels common to the SRM and the detector data.

        Does nothing if e_low_det, e_high_det, or matrix are not yet
        loaded.

        Returns
        -------
        None
        """
        if file:
            self.rname = file
        else:
            self.rname = askopenfilename(initialdir=".",
                                         filetypes=(("FITS files", "*.fits"), ("All Files", "*.*")),
                                         title="Please Select Spectrum or Image File")
        self.text_filename2.delete(0, 'end')

        if self.rname:
            self.text_filename2.insert(0, self.rname)  # Displays the input file name in Entry box
            # Loading data
            data = get_srm_data(self.rname)
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

    def onSelect(self, event):
        """
        Callback triggered when a selection is made in the model listbox.
        Updates self.fit_model and displays the selected model's
        information in the (read-only) info listbox.

        Parameters
        ----------
        event : tk.Event
            <<ListboxSelect>> event generated by tkinter.

        Returns
        -------
        None
        """
        try:
            # Get the index of the selection
            selected_index = self.lbox.curselection()[0]
            selected_name = self.lbox.get(selected_index)

            # Temporarily re-enable the info Listbox
            self.list_selection.config(state='normal')
            self.list_selection.delete(0, END)

            # Retrieve and insert the corresponding info
            if selected_name in self.list:
                for line in self.list[selected_name]:
                    self.list_selection.insert(END, line)
            else:
                self.list_selection.insert(END, "No information available.")

            # Disable the info Listbox (to make it non-clickable)
            self.list_selection.config(state='disabled')

        except Exception as e:
            print("Error in onSelect:", e)

    def ask_custom_yesno(title, message):
        win = Toplevel()
        win.title(title)
        win.resizable(False, False)
        win.grab_set()  # modal

        # Content
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
        """
        Opens a modal popup window letting the user choose the X and Y
        axis scale ('linear' or 'log') for the photon flux plot.

        The choices are saved in self.photon_xscale and
        self.photon_yscale.

        Returns
        -------
        None
        """

        def confirm():
            self.photon_xscale = x_choice.get()
            self.photon_yscale = y_choice.get()
            popup.destroy()

        popup = Toplevel(self.top2)
        popup.title("Photon Plot Axes")

        # Desired size
        window_width = 400
        window_height = 200

        # Compute the centred position
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        pos_x = int((screen_width / 2) - (window_width / 2))
        pos_y = int((screen_height / 2) - (window_height / 2))

        popup.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        popup.resizable(False, False)

        # User interface
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
        """
        Checks whether a background has been computed when the
        'Data-Background' box is checked for the first time. If no
        background is available, offers to open the BackgroundWindow.

        Returns
        -------
        None
        """
        if self.show_db_var.get():  # If checked
            if not background.BackgroundWindow.DATA_BKG_SELECTED:
                answer = Fitting.ask_custom_yesno(
                    "Background Not Selected",
                    "You have not yet generated the Background.\n"
                    "Would you like to open the Background window now?"
                )
                if answer:
                    # ✅ close current Fit Options window and open Background window
                    background.BackgroundWindow()
                else:
                    # ✅ uncheck the checkbox
                    self.show_db_var.set(0)
                return

    def on_background_clicked(self):
        """
        Callback for the 'Data-Background' checkbox. Handles two cases:
        - Background already computed: offers to recompute it.
        - No background: delegates to on_background_check().

        Returns
        -------
        None
        """
        if self.show_db_var.get():
            if background.BackgroundWindow.DATA_BKG_SELECTED:
                answer = Fitting.ask_custom_yesno(
                    "Background already selected",
                    "A background has already been selected.\nWould you like to select a new one?"
                )
                if answer:
                    background.BackgroundWindow.DATA_BKG_SELECTED = False
                    background.BackgroundWindow()  # Open new Background selection

                self.show_db_var.set(1)  # Keep checkbox checked
            else:
                self.on_background_check()  # Original logic (first-time case)

    def fit_unconstrained_then_bounded(self, model_template, x_fit, y_fit, y_err,
                                       param_names, bounds_map=None, initial_values=None):
        """
        Two-step fitting strategy:
        1) Unconstrained fit with the active fitter (LevMar).
        2) If the result falls outside the user bounds, re-runs a bounded
           fit (LevMar with .min / .max applied to the parameters).

        Parameters
        ----------
        model_template : astropy FittableModel
            Initial model (modified in place for the initial values,
            but copied before each fit).
        x_fit : ndarray
            Independent variable over the fit range (zero vector for
            ForwardFolded models).
        y_fit : ndarray
            Observed data over the fit range [counts s-1].
        y_err : ndarray
            Errors on y_fit [counts s-1].
        param_names : list of str
            Names of the model's free parameters.
        bounds_map : dict or None, optional
            {param: (min, max)} — bounds to check and apply in step 2.
            If None, step 2 is skipped.
        initial_values : dict or None, optional
            {param: initial_value} — starting values before step 1.

        Returns
        -------
        astropy FittableModel
            Model with fitted parameters (step 1 or 2).
        """
        # Apply the initial values (without setting bounds)
        if initial_values:
            for pname, val in initial_values.items():
                if hasattr(model_template, pname):
                    try:
                        getattr(model_template, pname).value = float(val)
                    except Exception:
                        pass

        # 1) Unconstrained fit
        try:
            fitted_nc = self.fitter(copy.deepcopy(model_template), x_fit, y_fit,
                                    weights=1.0 / (y_err + 1e-30))
        except Exception as e:
            print("⚠️ Unconstrained LevMar fit failed:", e)
            return copy.deepcopy(model_template)

        # extract the unconstrained values
        uncon_values = [getattr(fitted_nc, p).value for p in param_names]

        # 2) No bounds provided -> use the unconstrained solution
        if not bounds_map:
            return fitted_nc

        # 4) Otherwise, bounded fit via LevMarLSQFitter with min/max
        bounded_model = copy.deepcopy(model_template)

        # apply the bounds to each parameter
        for pname in param_names:
            if hasattr(bounded_model, pname):
                par = getattr(bounded_model, pname)
                lo, hi = bounds_map.get(pname, (None, None))
                if lo is not None:
                    par.min = lo
                if hi is not None:
                    par.max = hi

                # Priority: user value > default value > unconstrained fit
                if initial_values and pname in initial_values:
                    par.value = initial_values[pname]
                elif pname in Fitting.default_param_values.get(model_template.__class__.__name__, {}):
                    par.value = Fitting.default_param_values[model_template.__class__.__name__][pname]
                else:
                    par.value = uncon_values[param_names.index(pname)]

        # initialise to the unconstrained fit values
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
        Two-step fitting strategy with explicit checking of the user
        bounds:
        1) Fit with the internal bounds (default_param_bounds).
        2) If the result violates the user bounds, re-runs with the
           user bounds via _fit_with_user_bounds_only().

        Parameters
        ----------
        model_template : astropy FittableModel
            Initial model.
        x_fit : ndarray
            Independent variable over the fit range.
        y_fit : ndarray
            Observed data [counts s-1].
        y_err : ndarray
            Errors on y_fit [counts s-1].
        param_names : list of str
            Names of the free parameters.
        model_key : str
            Model key in the bounds and values dictionaries.
        internal_bounds_map : dict or None, optional
            Step 1 bounds (default: default_param_bounds[model_key]).
        user_bounds_map : dict or None, optional
            Step 2 bounds (default: user_param_bounds[model_key]).
        initial_values : dict or None, optional
            Initial values (default: user_param_values[model_key]).

        Returns
        -------
        astropy FittableModel
            Model fitted with optimal parameters.
        """
        # --- Default maps ---
        if internal_bounds_map is None:
            internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
        if user_bounds_map is None:
            user_bounds_map = self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {}))
        if initial_values is None:
            initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

        # --- Step 0: apply the initial values ---
        try:
            for pname, val in initial_values.items():
                if hasattr(model_template, pname):
                    getattr(model_template, pname).value = float(val)
        except Exception:
            pass

        # --- Step 1: model with internal bounds ---
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

        # --- Check whether fitted1 is within the user bounds ---
        tol = 1e-12
        in_user_bounds = True
        for pname in param_names:
            if not hasattr(fitted1, pname):
                continue
            attr = getattr(fitted1, pname)
            if isinstance(attr, (int, float, np.floating)):
                # this is a fixed parameter → skip
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

        # --- Step 2: refit with the user bounds ---
        return self._fit_with_user_bounds_only(model_template, x_fit, y_fit, y_err, param_names, user_bounds_map,
                                               initial_values)

    def _fit_with_user_bounds_only(self, model_template, x_fit, y_fit, y_err,
                                   param_names, user_bounds_map, initial_values):
        """
        Fit with the active fitter applying only the user bounds.
        Used as step 2 of fit_with_bounds_check().

        Parameters
        ----------
        model_template : astropy FittableModel
            Source model (copied before modification).
        x_fit : ndarray
            Independent variable over the fit range.
        y_fit : ndarray
            Observed data [counts s-1].
        y_err : ndarray
            Errors on y_fit [counts s-1].
        param_names : list of str
            Names of the free parameters.
        user_bounds_map : dict
            {param: (min, max)} — user bounds to apply.
        initial_values : dict
            {param: value} — initial parameter values.

        Returns
        -------
        astropy FittableModel
            Fitted model, or unfitted bounded model on failure.
        """
        model_bounded = copy.deepcopy(model_template)

        # Apply the initial values and user bounds
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
        """
        Main entry point for the 'Do Fit' button. Orchestrates the whole
        fitting pipeline for the model selected in the listbox.

        Internal steps:
        1) Retrieves and prepares the data (optional background
           subtraction, time averaging, masking of invalid channels).
        2) Computes the count rate, errors, and flux according to the
           selected unit (Rate / Counts / Flux).
        3) Applies the mask over the energy range [fit_Emin, fit_Emax].
        4) Runs the fit via fit_unconstrained_then_bounded() or
           fit_with_bounds_check() depending on the model.
        5) Reconstructs the model over the full domain for display.
        6) Displays the main plot (data + model) and, optionally,
           the deconvolved photon spectrum.

        Returns
        -------
        None
        """
        selection = self.lbox.curselection()
        if not selection:
            messagebox.showwarning("No Model Selected",
                                   "Please select a fit model before clicking 'Do Fit'.")
            return

        if self.fname is None and self.rname is None:
            messagebox.showwarning("No File Selected", "Please, choose input file.")
            return

        # ── Preparing the data ────────────────────────────────
        if self.show_db_var.get():
            idx_s = background.BackgroundWindow.DATA_BKG_START
            idx_e = background.BackgroundWindow.DATA_BKG_END
            bkg = np.nanmean(self.counts[idx_s:idx_e + 1, :], axis=0)
            raw = np.where(self.counts - bkg > 0, self.counts - bkg, 1e-5)
            absolute_name = "Data - Background"
        else:
            raw = self.counts
            absolute_name = "Data"

        counts_all = np.nanmean(raw, axis=0)
        counts_err_all = np.nanmean(self.counts_err, axis=0)
        exposure = float(np.nanmean(self.time_del))

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

        # ── Checking that the arrays are not empty ───────────────
        if len(e_low_det) == 0 or len(e_high_det) == 0:
            messagebox.showerror(
                "Data error",
                "Energy channels array is empty.\n\n"
                "This may be caused by a mismatch between the spectrum file "
                "and the SRM file (incompatible dimensions).\n\n"
                f"Spectrum channels : {len(self.e_low_det)}\n"
                f"SRM channels      : {matrix.shape[1]}"
            )
            return

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

        # ── Units ────────────────────────────────────────────────
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
            plt.xlabel("Channel Energy (keV)")
            plt.ylabel(y_label)
            plt.title(f"Fitting on [{fit_Emin}, {fit_Emax}] keV using {self.statname}")
            if self.grid_var.get():
                plt.grid(True, which="both", ls="--", alpha=0.5)
            else:
                plt.grid(False)
            plt.legend()
            plt.tight_layout()

        def plot_photon(model_func, param_txt):
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
            if self.grid_var.get():
                plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            if self.show_params_var.get():
                add_param_text(param_txt)
            plt.tight_layout()

        def plot_photon_discrete(flux_photons, param_txt, title="Photon Flux Model (Neural Network)"):
            plt.figure()
            plt.step(Edges_photon[:-1], flux_photons, where='post',
                     label='Photon model', color='green')
            xscale = getattr(self, "photon_xscale", "log")
            yscale = getattr(self, "photon_yscale", "log")
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Photon flux [photons / (s cm² keV)]")
            plt.title(title)
            if self.grid_var.get():
                plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            if self.show_params_var.get():
                add_param_text(param_txt)
            plt.tight_layout()

        # ── Data plot ───────────────────────────────────────────

        param_text = None
        model_func_photon = None

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
                model_func_photon = (lambda E: amplitude * (E / E_pivot_val) ** (-alpha))

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
                model_func_photon = (lambda E: amplitude * np.where(
                    E < E_break,
                    (E / E_break) ** (-alpha_1),
                    (E / E_break) ** (-alpha_2)))

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
                model_func_photon = (lambda E: (p0 * (E / p2) ** p1) * np.exp(e3 - E / e4))

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
                model_func_photon = (lambda E: (1.07e-42 * 1.2 * EM) / (E * np.sqrt(max(1e-3, T))) * np.exp(-E / T))

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
                model_func_photon = (lambda E: (
                        (1.07e-42 * 1.2 * EM) / (E * np.sqrt(max(1e-3, T))) * np.exp(-E / T)
                        + amplitude * (E / 100.0) ** (-alpha)))

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

            model_display = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix, exposure, E_cut_val, E_pivot_val)
            model_display.amplitude.value = amplitude
            model_display.alpha.value = alpha

            model_y = to_unit(model_display(x_fake))

            # Cutoff mask
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, 0)

            plt.axvline(E_cut_val, linestyle='dashed', color='y')
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Power Law Cutoff Fix:\n amplitude={amplitude:.2e}\n"
                          f" alpha={alpha:.2f}\n E_pivot={E_pivot_val:.2f} keV\n"
                          f" E_cut={E_cut_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                model_func_photon = (lambda E: np.where(
                    E >= E_cut_val,
                    amplitude * (E / E_pivot_val) ** (-alpha), 0.0))


        # ══════════════════════════════════════════════════════════
        #  6 — Power Law Cutoff Free
        # ══════════════════════════════════════════════════════════
        elif idx == 6:
            model_key = "PowerLawCutoffFree"
            initial_values, bounds_map = (
                self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {})),
                self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {})))
            E_cut_bound = (initial_values.get("Ec_min", fit_Emin), initial_values.get("Ec_max", fit_Emax))
            E_cut_bound = (E_cut_bound[0] if E_cut_bound[0] > fit_Emin else fit_Emin, E_cut_bound[1] if E_cut_bound[1] < fit_Emax else fit_Emax)
            E_pivot_val = initial_values.get("E_pivot", 100.0)

            model_template = PowerLawCutoffFix(
                e_low_true, e_high_true, matrix_fit, exposure, E_pivot_val)

            def chi2_for_Ecut(E_cut_val):
                model_template.E_cut = E_cut_val
                fitted_model = self.fit_unconstrained_then_bounded(
                    model_template, x_fit, y_fit, y_err,
                    ["amplitude", "alpha"], bounds_map, initial_values
                )
                return np.sum(((y_fit - fitted_model(x_fit)) / (y_err + 1e-30)) ** 2)

            # Minimise the error as a function of E_cut within the user-set limits
            result = minimize_scalar(
                chi2_for_Ecut,
                bounds=E_cut_bound,
                method="bounded",
                options={"xatol": 1e-3}
            )

            E_cut_val = result.x

            # Retrieve the final model at the optimal E_cut
            model_template.E_cut = E_cut_val
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values
            )

            amplitude = fitted.amplitude.value
            alpha = fitted.alpha.value

            model_display = PowerLawCutoffFix(e_low_true, e_high_true, matrix, exposure)
            for p in ["amplitude", "alpha"]:
                getattr(model_display, p).value = getattr(fitted, p).value
            model_display.E_cut = E_cut_val

            model_y = to_unit(model_display(x_fake))

            # Cutoff mask
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, 0)
            plt.axvline(E_cut_val, linestyle='dashed', color='y')
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            param_text = (f"Power Law Cutoff Free:\n amplitude={amplitude:.2e}\n"
                          f" alpha={alpha:.2f}\n E_pivot={E_pivot_val:.2f} keV\n"
                          f" E_cut={E_cut_val:.2f} keV")
            if self.show_params_var.get():
                add_param_text(param_text)
            if self.show_photon_var.get():
                model_func_photon = (lambda E: np.where(
                    E >= E_cut_val,
                    amplitude * (E / E_pivot_val) ** (-alpha), 0.0))

        # ══════════════════════════════════════════════════════════
        #  7 — V_TH x Cutoff Fix
        # ══════════════════════════════════════════════════════════
        elif idx == 7:
            model_key = "V_TH x PowerLawCutoffFix"
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

            # Cutoff mask
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, np.nan)

            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            fit_mask = (edges_det[:-1] >= fit_Emin) & (edges_det[1:] <= E_cut_val)

            x_fit = np.zeros(fit_mask.sum())
            counts_fit = counts[fit_mask]
            counts_err_fit = counts_err[fit_mask]
            matrix_fit = matrix[:, fit_mask]
            x_fake = np.zeros_like(counts)

            y_fit = counts_fit / exposure
            y_err = counts_err_fit / exposure

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

            plt.axvline(E_cut_val, linestyle='dashed', color='y')
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

                model_func_photon = model_total

        # ══════════════════════════════════════════════════════════
        #  8 — V_TH x Cutoff Free
        # ══════════════════════════════════════════════════════════
        elif idx == 8:
            model_key = "V_TH x PowerLawCutoffFree"
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
                options={"xatol": 1e-3}
            )

            E_cut_val = result.x

            # Retrieve the final model at the optimal E_cut
            model_template.E_cut = E_cut_val
            fitted = self.fit_unconstrained_then_bounded(
                model_template, x_fit, y_fit, y_err,
                ["amplitude", "alpha"], bounds_map, initial_values
            )

            amplitude = fitted.amplitude.value
            alpha = fitted.alpha.value

            model_display = PowerLawCutoffFix(e_low_true, e_high_true, matrix, exposure)
            for p in ["amplitude", "alpha"]:
                getattr(model_display, p).value = getattr(fitted, p).value
            model_display.E_cut = E_cut_val

            model_y = to_unit(model_display(x_fake))

            # Cutoff mask
            fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[:-1] <= fit_Emax)
            model_y = np.where(fit_mask_cutoff, model_y, np.nan)

            plt.axvline(E_cut_val, linestyle='dashed', color='y')
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            fit_mask = (edges_det[:-1] >= fit_Emin) & (edges_det[1:] <= E_cut_val)

            x_fit = np.zeros(fit_mask.sum())
            counts_fit = counts[fit_mask]
            counts_err_fit = counts_err[fit_mask]
            matrix_fit = matrix[:, fit_mask]
            x_fake = np.zeros_like(counts)

            y_fit = counts_fit / exposure
            y_err = counts_err_fit / exposure

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

            param_text = (f"V_TH + Power Law Cutoff Free:\n"
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

                model_func_photon = model_total

        # ══════════════════════════════════════════════════════════
        #  9 - Neural Network
        # ══════════════════════════════════════════════════════════

        elif idx == 9:
            model_key = "Neural Network"
            nn_model = NeuralNetModel.load(path="data/nn_powerlaw_150k.pt", device="cpu")

            photon_flux = nn_model.predict(counts, srm=matrix)

            folded = (np.asarray(matrix).T @ photon_flux) / exposure
            model_y = to_unit(folded)
            plt.step(edges_det[:-1], model_y, where='post', label='Fitted Model', color='blue')

            e_true = 0.5 * (np.asarray(e_low_true) + np.asarray(e_high_true))

            mask_true = (e_true >= fit_Emin) & (e_true <= fit_Emax)

            valid_pl = (photon_flux > 0) & (e_true > 0) & mask_true
            nn_alpha, nn_amplitude = np.nan, np.nan
            if valid_pl.sum() >= 2:
                slope, intercept = np.polyfit(
                    np.log(e_true[valid_pl]), np.log(photon_flux[valid_pl]), 1)
                nn_alpha = -slope
                nn_amplitude = np.exp(intercept)

            photon_flux_display = np.where(mask_true, photon_flux, np.nan)

            param_text = (f"Neural Network (effective power law):\n"
                          f" amplitude = {nn_amplitude:.2e}\n alpha = {nn_alpha:.2f}\n")
            if self.show_params_var.get():
                add_param_text(param_text)

        finalize_main_plot()
        if self.show_photon_var.get():
            if idx == 9:
                plot_photon_discrete(photon_flux_display, param_text)
            else:
                plot_photon(model_func_photon, param_text)
        plt.show()