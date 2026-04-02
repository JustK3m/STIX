import copy
import os
import sys
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

import numpy as np
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.optimize import least_squares, minimize_scalar

from . import background
from .fitting.fitters import LevMarCstatFitter
from .fitting.methods import ForwardFolded

register_matplotlib_converters()


class Fitting:
    """
    Class to perform a spectrum fitting
    """

    # fname = 'solo_L1A_stix-sci-spectrogram' \
    #         '-2102140001_20210214T014006-20210214T015515_008648_V01.fits'
    # rname = 'stx_srm_2021feb14_0140_0155.fits'

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

    # === default parameter bounds for each model ===
    default_param_bounds = {
        "PowerLaw1D": {
            "amplitude": (None, None),
            "alpha": (None, None),
        },
        "BrokenPowerLaw1D": {
            "amplitude": (1e-5, 1e3),
            "E_break": (1.0, 100.0),
            "alpha_1": (0.1, 10.0),
            "alpha_2": (0.1, 10.0)
        },
        "Single Power Law Times an Exponential": {
            "p0": (1e-3, 1e5),
            "p1": (-5, 5),
            "p2": (1e-2, 100),
            "e3": (-10, 10),
            "e4": (0.1, 100)
        },
        "V_TH": {
            "EM": (1e44, 1e52),
            "T": (0.1, 50.0)
        },
        "V_TH + PowerLaw": {
            "EM": (1e44, 1e52),
            "T": (0.1, 50.0),
            "amplitude": (1e-2, 1e2),
            "alpha": (2, 10.0)
        },
        "PowerLawCutoffFix": {
            "amplitude": (1e-12, 1e6),
            "alpha": (0.1, 50.0),
        },
        "PowerLawCutoffFree": {
            "amplitude": (None, None),
            "alpha": (None, None),
        },
        "V_TH + PowerLawCutoffFix": {
            "EM": (1e44, 1e52),
            "T": (0.1, 50.0),
            "amplitude": (1e-2, 1e2),
            "alpha": (2, 10.0)
        },
    }

    default_param_values = {
        "PowerLaw1D": {
            "amplitude": 1e-2,
            "alpha": 2.0,
            "E_pivot": 100.0
        },
        "BrokenPowerLaw1D": {
            "amplitude": 1e-2,
            "E_break": 10.0,
            "alpha_1": 2.0,
            "alpha_2": 3.0
        },
        "Single Power Law Times an Exponential": {
            "p0": 1.0,
            "p1": -2.0,
            "p2": 20.0,
            "e3": 1.0,
            "e4": 10.0
        },
        "V_TH": {
            "EM": 6e48,  # 5.71e+48
            "T": 1.0,  # 1.23
        },
        "V_TH + PowerLaw": {
            "EM": 1e48,
            "T": 1.0,
            "amplitude": 1e-2,
            "alpha": 2.0,
            "E_pivot": 100.0
        },
        "PowerLawCutoffFix": {
            "amplitude": 1e-2,
            "alpha": 2.0,
            "E_cut": 10.0,  # fixe mais modifiable
            "E_pivot": 100.0
        },
        "PowerLawCutoffFree": {
            "amplitude": 1e-2,
            "alpha": 2.0,
            "E_pivot": 100.0  # fitté
        },
        "V_TH + PowerLawCutoffFix": {
            "EM": 1e48,
            "T": 1.0,
            "amplitude": 1e-2,
            "alpha": 2.0,
            "E_cut": 10
        },
    }

    # create a new window called 'SPEX Fit Options'
    def __init__(self, root):
        """Creates a new window, providing widgets to perform fitting analysis"""
        self.sender = None

        self.top2 = Toplevel()
        self.top2.title('SPEX Fit Options')  # title of the window
        self.top2.geometry("1000x600")  # size of the new window
        # Label(self.top2,
        #    text="Fit Options",  # place the text at the top of the window
        #    fg="red",  # in red
        #    font="Helvetica 12 bold italic").pack()  # with specific text font

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

        self.energy_min_var = tk.DoubleVar()
        self.energy_max_var = tk.DoubleVar()

        self.energy_min_var = tk.DoubleVar(value=0)
        self.energy_min2 = tk.OptionMenu(self.top2, self.energy_min_var, 0)
        # self.energy_min2.pack()

        self.energy_max_var = tk.DoubleVar(value=0)
        self.energy_max2 = tk.OptionMenu(self.top2, self.energy_max_var, 0)
        # self.energy_max2.pack()

        self.sepBkVar = IntVar()

        self.lbl1 = Label(self.top2, text="Choose Fit Function Model:", fg='blue',
                          font=("Helvetica", 11, "bold"))  # name the listbox
        self.lbl1.place(relx=0.07, rely=0.07)  # set the position on window

        self.lbl2 = Label(self.top2, text="Information:", fg='blue',
                          font=("Helvetica", 11, "bold"))  # name the scrollbar
        self.lbl2.place(relx=0.44, rely=0.07)  # set the position

        self.lbl3 = Label(self.top2, text="Choose the files and energy range:", fg='blue',
                          font=("Helvetica", 11, "bold"))  # name the scrollbar
        self.lbl3.place(relx=0.65, rely=0.07)  # set the position

        # self.lblFunc = Label(self.top2, text="Set function components: ")  # name the scrollbar
        # self.lblFunc.place(relx=0.73, rely=0.20)  # set the position

        # Spectrum: file name
        self.label_filename = Label(self.top2, text="Spectrum: ")
        self.label_filename.place(relx=0.65, rely=0.2, anchor=W)
        self.text_filename = Entry(self.top2, width=30)
        self.text_filename.place(relx=0.72, rely=0.2, anchor=W)
        if Fitting.fname:
            self.text_filename.insert(0, Fitting.fname)
            self.open_file(Fitting.fname)
        else:
            self.text_filename.insert(0, "No file chosen")

        self.btn_browse = Button(self.top2, text='Browse ->', command=self.open_file)
        self.btn_browse.place(relx=0.92, rely=0.2, anchor=W)

        # Response matrix: file name
        self.label_filename2 = Label(self.top2, text="Response: ")
        self.label_filename2.place(relx=0.65, rely=0.25, anchor=W)
        self.text_filename2 = Entry(self.top2, width=30)
        self.text_filename2.place(relx=0.72, rely=0.25, anchor=W)
        if Fitting.rname:
            self.text_filename2.insert(0, Fitting.rname)
            self.open_srm_file(Fitting.rname)
        else:
            self.text_filename2.insert(0, "No file chosen")

        self.btn_browse2 = Button(self.top2, text='Browse ->', command=self.open_srm_file)
        self.btn_browse2.place(relx=0.92, rely=0.25, anchor=W)

        self.fit_model = str()

        self.user_param_bounds = {}  # bounds set by user in Set_Function
        self.user_param_values = {}  # initial values set by user in Set_Function
        self.user_param_modified = {}  # True if user modified bounds/values from default

        # def Set_Function():
        #     """Fenêtre popup pour définir les bornes des paramètres du modèle sélectionné"""
        #     if not self.fit_model:
        #         messagebox.showwarning("No Model Selected", "Please select a model first.")
        #         return

        #     newwin = Toplevel(root)
        #     newwin.title(f'{self.fit_model} - Parameter Ranges')
        #     newwin.geometry("500x400")

        #     Label(newwin, text=f"Set parameter ranges for {self.fit_model}:",
        #         fg='blue', font=("Helvetica", 11, "bold")).pack(pady=10)

        #     # Dictionnaire des paramètres par modèle
        #     model_params = {
        #         "PowerLaw1D": ["amplitude", "alpha"],
        #         "BrokenPowerLaw1D": ["amplitude", "E_break", "alpha_1", "alpha_2"],
        #         "Single Power Law Times an Exponential": ["p0", "p1", "p2", "e3", "e4"],
        #         "V_TH": ["EM", "T"],
        #         "V_TH + PowerLaw": ["EM", "T", "amplitude", "alpha"]
        #     }

        #     self.param_entries = {}  # stocker les entrées

        #     if self.fit_model in model_params:
        #         for param in model_params[self.fit_model]:
        #             frame = Frame(newwin)
        #             frame.pack(pady=5, fill="x")

        #             Label(frame, text=f"{param} min:", width=12, anchor="w").pack(side=LEFT)
        #             min_entry = Entry(frame, width=10)
        #             min_entry.pack(side=LEFT, padx=5)

        #             Label(frame, text=f"{param} max:", width=12, anchor="w").pack(side=LEFT)
        #             max_entry = Entry(frame, width=10)
        #             max_entry.pack(side=LEFT, padx=5)

        #             self.param_entries[param] = (min_entry, max_entry)

        #     def save_params():
        #         self.param_bounds = {}
        #         for param, (min_e, max_e) in self.param_entries.items():
        #             try:
        #                 min_val = float(min_e.get())
        #                 max_val = float(max_e.get())
        #                 self.param_bounds[param] = (min_val, max_val)
        #             except ValueError:
        #                 messagebox.showerror("Invalid input", f"Invalid bounds for {param}")
        #                 return
        #         newwin.destroy()

        #     Button(newwin, text="Save", command=save_params, bg="green", fg="white").pack(pady=15)

        #     """Fenêtre pour définir les bornes des paramètres du modèle sélectionné."""
        #     try:
        #         idx = self.lbox.curselection()[0]
        #         model_key = self.lbox.get(idx)
        #     except Exception:
        #         messagebox.showwarning("No Model Selected", "Please select a model first.")
        #         return

        #     newwin = Toplevel(self.top2)
        #     newwin.title(f"{model_key} - Parameter Ranges")
        #     newwin.geometry("520x420")

        #     Label(newwin, text=f"Set parameter ranges for {model_key}:",
        #         fg='blue', font=("Helvetica", 11, "bold")).pack(pady=10)

        #     # Récupère les bornes sauvegardées par l’utilisateur, sinon par défaut
        #     if model_key in self.user_param_bounds:
        #         base_bounds = self.user_param_bounds[model_key]
        #     else:
        #         base_bounds = Fitting.default_param_bounds.get(model_key, {})

        #     self.param_entries = {}
        #     initial_display = {}

        #     for param in base_bounds.keys():
        #         default_lo, default_hi = Fitting.default_param_bounds.get(model_key, {}).get(param, (None, None))
        #         saved_lo, saved_hi = self.user_param_bounds.get(model_key, {}).get(param, (None, None))

        #         # if model_key == "PowerLaw1D":
        #         #     # ⚡ Par défaut : aucune borne affichée → champs vides
        #         #     disp_min, disp_max = "", ""
        #         # else:
        #         #     disp_min = str(saved_lo if saved_lo is not None else default_lo or "")
        #         #     disp_max = str(saved_hi if saved_hi is not None else default_hi or "")

        #         # --- Toujours afficher un min (par défaut ou sauvegardé)
        #         disp_min = str(saved_lo if saved_lo is not None else default_lo or "")

        #         # --- Pour PowerLaw1D : max vide par défaut
        #         if model_key == "PowerLaw1D":
        #             disp_max = "" if saved_hi is None else str(saved_hi)
        #         else:
        #             disp_max = str(saved_hi if saved_hi is not None else default_hi or "")

        #         row = Frame(newwin)
        #         row.pack(pady=6, fill="x")

        #         Label(row, text=f"{param} min:", width=14, anchor="w").pack(side=LEFT)
        #         e_min = Entry(row, width=14)
        #         e_min.insert(0, disp_min)
        #         e_min.pack(side=LEFT, padx=6)

        #         Label(row, text=f"{param} max:", width=14, anchor="w").pack(side=LEFT)
        #         e_max = Entry(row, width=14)
        #         e_max.insert(0, disp_max)
        #         e_max.pack(side=LEFT, padx=6)

        #         self.param_entries[param] = (e_min, e_max)
        #         initial_display[param] = (disp_min, disp_max)

        #     def save_params():
        #         bounds = {}
        #         modified = False

        #         for param, (e_min, e_max) in self.param_entries.items():
        #             min_txt = e_min.get().strip()
        #             max_txt = e_max.get().strip()

        #             lo = float(min_txt) if min_txt != "" else None
        #             hi = float(max_txt) if max_txt != "" else None

        #             # Vérifier cohérence si min et max donnés
        #             if lo is not None and hi is not None and hi <= lo:
        #                 messagebox.showerror("Invalid bounds", f"{param}: max must be > min")
        #                 return

        #             bounds[param] = (lo, hi)

        #             # Détection de modification
        #             init_min, init_max = initial_display[param]
        #             if min_txt != init_min or max_txt != init_max:
        #                 modified = True

        #         self.user_param_bounds[model_key] = bounds
        #         self.user_param_modified[model_key] = modified

        #         print(f"[Set_Function] {model_key} bounds saved: {bounds}, modified={modified}")
        #         newwin.destroy()

        #     Button(newwin, text="Save", command=save_params, bg='green', fg='white').pack(pady=12)

        def Set_Function():
            """Fenêtre pour définir valeur initiale (default), min et max de chaque paramètre du modèle sélectionné."""
            try:
                idx = self.lbox.curselection()[0]
                model_key = self.lbox.get(idx)
            except Exception:
                messagebox.showwarning("No Model Selected", "Please select a model first.")
                return

            newwin = Toplevel(self.top2)
            newwin.title(f"{model_key} - Parameter Settings")
            newwin.geometry("680x480")
            newwin.configure(bg="#f7f9fc")

            Label(
                newwin,
                text=f"Set parameter values for {model_key}",
                fg="#1e3a8a",
                bg="#f7f9fc",
                font=("Helvetica", 13, "bold")
            ).pack(pady=15)

            self.param_entries = {}
            initial_display = {}

            # Récupération des valeurs par défaut et bornes
            base_defaults = Fitting.default_param_values.get(model_key, {})
            base_bounds = Fitting.default_param_bounds.get(model_key, {})
            saved_values = self.user_param_values.get(model_key, {})
            saved_bounds = self.user_param_bounds.get(model_key, {})

            form_frame = Frame(newwin, bg="#f7f9fc")
            form_frame.pack(pady=10, padx=20, fill="x")

            for param in base_defaults.keys():
                disp_default = str(saved_values.get(param, base_defaults[param]))
                pmin, pmax = saved_bounds.get(param, base_bounds.get(param, (None, None)))
                disp_min = "" if pmin is None else str(pmin)
                disp_max = "" if pmax is None else str(pmax)

                row = Frame(form_frame, bg="#f7f9fc")
                row.pack(pady=6, fill="x")

                Label(row, text=f"{param}:", width=14, anchor="w", bg="#f7f9fc").pack(side="left")

                # --- Cas spécial : E_pivot n'a pas de Min/Max ---
                if (model_key == "PowerLaw1D" or model_key == "V_TH + PowerLaw" or model_key == "PowerLawCutoffFix"
                    or model_key == "PowerLawCutoffFree" or model_key == "V_TH + PowerLawCutoffFix") and (
                        param == "E_pivot" or param == "E_cut"):
                    Label(row, text="Value:", bg="#f7f9fc").pack(side="left")
                    e_def = Entry(row, width=10)
                    e_def.insert(0, disp_default)
                    e_def.pack(side="left", padx=6)

                    self.param_entries[param] = (e_def, None, None)
                    initial_display[param] = (disp_default, "", "")
                    continue

                # --- Cas normal : Default + Min + Max ---
                Label(row, text="Default:", bg="#f7f9fc").pack(side="left")
                e_def = Entry(row, width=10)
                e_def.insert(0, disp_default)
                e_def.pack(side="left", padx=6)

                Label(row, text="Min:", bg="#f7f9fc").pack(side="left")
                e_min = Entry(row, width=10)
                e_min.insert(0, disp_min)
                e_min.pack(side="left", padx=6)

                Label(row, text="Max:", bg="#f7f9fc").pack(side="left")
                e_max = Entry(row, width=10)
                e_max.insert(0, disp_max)
                e_max.pack(side="left", padx=6)

                self.param_entries[param] = (e_def, e_min, e_max)
                initial_display[param] = (disp_default, disp_min, disp_max)

            # --- Actions ---
            def save_params():
                values, bounds = {}, {}
                modified = False

                for param, (e_def, e_min, e_max) in self.param_entries.items():
                    # Cas spécial : pas de min/max (E_pivot)
                    if e_min is None and e_max is None:
                        def_txt = e_def.get().strip()
                        try:
                            def_val = float(def_txt)
                        except ValueError:
                            messagebox.showerror("Invalid input", f"{param}: invalid value")
                            return
                        values[param] = def_val
                        bounds[param] = (None, None)
                        if def_txt != initial_display[param][0]:
                            modified = True
                        continue

                    # Cas normal
                    def_txt, min_txt, max_txt = e_def.get().strip(), e_min.get().strip(), e_max.get().strip()

                    try:
                        def_val = float(def_txt)
                    except ValueError:
                        messagebox.showerror("Invalid input", f"{param}: invalid default value")
                        return

                    lo = float(min_txt) if min_txt != "" else None
                    hi = float(max_txt) if max_txt != "" else None

                    if lo is not None and hi is not None and hi <= lo:
                        messagebox.showerror("Invalid bounds", f"{param}: max must be > min")
                        return

                    values[param] = def_val
                    bounds[param] = (lo, hi)

                    if (def_txt, min_txt, max_txt) != initial_display[param]:
                        modified = True

                self.user_param_values[model_key] = values
                self.user_param_bounds[model_key] = bounds
                self.user_param_modified[model_key] = modified

                print(f"[Set_Function] {model_key} values saved: {values}, bounds={bounds}, modified={modified}")
                newwin.destroy()

            def reset_defaults():
                """Réinitialise tous les champs aux valeurs par défaut."""
                for param, (e_def, e_min, e_max) in self.param_entries.items():
                    def_val = Fitting.default_param_values[model_key][param]
                    lo, hi = Fitting.default_param_bounds.get(model_key, {}).get(param, (None, None))

                    e_def.delete(0, END)
                    e_def.insert(0, str(def_val))

                    if e_min is not None and e_max is not None:
                        e_min.delete(0, END)
                        e_min.insert(0, "" if lo is None else str(lo))
                        e_max.delete(0, END)
                        e_max.insert(0, "" if hi is None else str(hi))
                print(f"[Set_Function] {model_key} reset to defaults")

            def cancel_window():
                newwin.destroy()

            # --- Boutons ---
            btn_frame = Frame(newwin, bg="#f7f9fc")
            btn_frame.pack(pady=20)

            Button(btn_frame, text="Save", command=save_params,
                   bg="#16a34a", fg="white", width=12).pack(side="left", padx=10)

            Button(btn_frame, text="Reset to Defaults", command=reset_defaults,
                   bg="#f97316", fg="white", width=16).pack(side="left", padx=10)

            Button(btn_frame, text="Cancel", command=cancel_window,
                   bg="#ef4444", fg="white", width=12).pack(side="left", padx=10)

        self.lblFunc = Label(self.top2, text="Set function components: ")  # name the scrollbar
        self.lblFunc.place(relx=0.65, rely=0.30)

        self.Value_Button = Button(self.top2, text="Function value(s)",
                                   command=Set_Function)  # place a "Function value" button
        self.Value_Button.place(relx=0.65, rely=0.35, relheight=0.05, relwidth=0.13)

        self.lblStat = Label(self.top2, text="Set statistics: ")
        self.lblStat.place(relx=0.85, rely=0.30)

        self.statname = "Chi2"

        def Set_Statistics(name):
            lkupStatistic = {"C-stat": LevMarCstatFitter(), "Chi2": LevMarLSQFitter()}
            self.fitter = lkupStatistic[name]
            self.menuStat.config(text=name)
            self.statname = name

            # if name == "C-stat":
            #     statWindow = Toplevel(self.top2)
            #     statWindow.title(f"Cstat - Parameter Settings")
            #     statWindow.geometry("380x350")
            #     statWindow.configure(bg="#f7f9fc")
            #
            #     cstat_params = {"Cost tolerance": self.fitter.f_tol, "Parameters Tolerance": self.fitter.x_tol,
            #                     "Gradient Tolerance": self.fitter.g_tol, "Max Iterations": self.fitter.max_iter, }
            #     Label(
            #         statWindow,
            #         text=f"Set parameter values for C-stat",
            #         fg="#1e3a8a",
            #         bg="#f7f9fc",
            #         font=("Helvetica", 13, "bold")
            #     ).pack(pady=15)
            #
            #     for name in cstat_params.keys():
            #         row = Frame(statWindow, bg="#f7f9fc")
            #         row.pack(pady=20, fill="x")
            #         Label(row, text=name, width=14, anchor="w", bg="#f7f9fc").pack(padx=10, side="left")
            #         Entry(row, bg="#f7f9fc", width=10, textvariable=cstat_params[name]).pack(padx=10, side="left")
            #
            #     Button(statWindow, text="Close", command=lambda: statWindow.destroy(), bg="#ef4444", fg="white",
            #            width=12).pack(pady=10, side="top")

        self.menuStat = tk.Menubutton(self.top2, text="Chi2", relief="raised")
        self.menuStat.place(relx=0.85, rely=0.35, relheight=0.05, relwidth=0.13)

        self.menuStat.menu = tk.Menu(self.menuStat, tearoff=0)
        self.menuStat["menu"] = self.menuStat.menu
        self.menuStat.menu.add_command(label="Chi2", command=lambda: Set_Statistics("Chi2"))
        self.menuStat.menu.add_command(label="C-stat", command=lambda: Set_Statistics("C-stat"))

        # Energies range(s) to fit

        fname = Fitting.fname
        rname = Fitting.rname
        if fname is None or rname is None:  # if file not choosen, print
            print('Please, choose input file')

        else:
            # counts_file = fits.open(fname)
            # srm_file = fits.open(rname)

            # Initialisation des listes pour le fond
            n_channels = len(self.e_low_det)

            self.bkg_start_index = [0] * n_channels
            self.bkg_end_index = [len(self.times)] * n_channels

            usable_channels = np.arange(min(self.matrix.shape[1], len(self.e_low_det)))

            e_low_det = self.e_low_det[usable_channels]
            e_high_det = self.e_high_det[usable_channels]

            self.text_min_energy = Label(self.top2, text="Min energy")
            self.text_min_energy.place(relx=0.75, rely=0.45, anchor=N)
            self.text_max_energy = Label(self.top2, text="Max energy")
            self.text_max_energy.place(relx=0.85, rely=0.45, anchor=N)

            e_low_values = sorted(set(e_low_det))
            e_high_values = sorted(set(e_high_det))

            e_high_values = [e for e in e_high_values if e != float('inf') and e != float('-inf')]

            e_low_values_int = [int(e) for e in e_low_values if e != 0]
            e_high_values_int = [int(e) for e in e_high_values]

            self.energy_min_var = IntVar()
            self.energy_max_var = IntVar()

            self.energy_min_var.set(min(e_low_values_int))
            self.energy_max_var.set(max(e_high_values_int))

            # Créer les OptionMenu pour l'énergie minimale et maximale
            self.energy_min2 = OptionMenu(self.top2, self.energy_min_var, *e_low_values_int)
            self.energy_max2 = OptionMenu(self.top2, self.energy_max_var, *e_high_values_int)

            self.energy_min2.place(relx=0.75, rely=0.50, anchor=N)
            self.energy_max2.place(relx=0.85, rely=0.50, anchor=N)

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

        self.DoFit5_Button = Button(self.frameFit, text="Do Fit",
                                    command=self._selective_fit)  # place a "Do Fit" button
        self.DoFit5_Button.place(relx=0.70, rely=0.38, relheight=0.23, relwidth=0.15)  # locate

        self.refreshButton5 = Button(self.top2, text="Refresh")  # add Refresh button at the buttom
        # resets original view
        self.refreshButton5.place(relx=0.4, rely=0.94)

        """Scrollbar with information related to each function"""
        self.closeButton5 = Button(self.top2, text="Close", command=self.destroy5)  # add Close button
        # Close "Fit Options" window
        self.closeButton5.place(relx=0.5, rely=0.94)
        self.models = ['PowerLaw1D', 'BrokenPowerLaw1D', 'Single Power Law Times an Exponential', 'V_TH',
                       'V_TH + PowerLaw', 'PowerLawCutoffFix', 'PowerLawCutoffFree',
                       'V_TH + PowerLawCutoffFix']  # , 'Neural Network' function names
        for p in self.models:
            """On the right: place an 'entry text' Scrollbar widget (scrollbar) When user highlight the function, 
            displays the text information about function description and input parameters"""
            self.lbox.insert(END, p)
        self.lbox.bind("<<ListboxSelect>>", self.onSelect)
        self.list = {'PowerLaw1D': {'One dimensional power law model', '\n\n',
                                    'amplitude – model amplitude at the reference energy', '\n',
                                    'Epivot – energie pivot (kEv)', '\n',
                                    'energy_data – reference energy', '\n', 'alpha – power law index'
                                    },
                     # if user choose PowerLaw1D, display
                     'BrokenPowerLaw1D': {'One dimensional power law model with a break', '\n\n',
                                          'amplitude - model amplitude at the break energy', '\n',
                                          'alpha 1 – power law index for energy_data<x_break', '\n',
                                          'alpha 2 – power law index for energy_data>x_break'},
                     # if user choose BrokenPowerLaw1D, display
                     'Gaussian': {'Single Gaussian function(high quality), width in sigma', '\n',
                                  'does not go through DRM', '\n',
                                  'This function returns the sum of Gaussian and ', '\n', '2nd order Polynomial',
                                  'amplitude - integrated intensity, mean - centroid', '\n', 'stddev - sigma'},
                     # if user choose Gaussian, display
                     'Polynomial': {'Polynomial function with offset in energy_data', '\n',
                                    'c0 - 0th order coefficient', '\n', 'c1 - 1st order coefficient', '\n',
                                    'c2 - 2nd order coefficient', '\n',
                                    'c3 - 3rd order coefficient', '\n', 'c4 - 4th order coefficient', '\n',
                                    'c5 - energy_data offset, such that function value at energy_data = c5 is C0 '},
                     # Polynomial
                     'Exponential': {'Exponential function', '\n', 't0 - Normalization', '\n',
                                     't1 - Pseudo temperature'},  # Exponential
                     'Single Power Law Times an Exponential': {'Multiplication of Single Power Law and Exponential',
                                                               '\n',
                                                               'p0 - normalization at epivot for power-law', '\n',
                                                               'p1 - negative power - law index', '\n',
                                                               'p2 - epivot (kEv) for power - law', '\n',
                                                               'e1 - normalization for exponential', '\n',
                                                               'e2 - pseudo temperature for exponential'},
                     # Single Power Law Times an Exponential
                     'Logistic Regression': {'Returns a sigmoid function', '\n'},  # Logistic Regression
                     'Lorentz': {'One dimensional Lorentzian model', '\n\n',
                                 'Amplitude correponds to peak value', '\n',
                                 'x_0 is the peak position (default value is 0)'},  # Lorentz Model
                     'Moffat': {'able to accurately reconstruct point spread functions', '\n',
                                'Moffat distribution'},  # Moffat model
                     'Voigt Profile': {'model computes the sum of Voigt function with a 2nd order polynomial', '\n',
                                       'amplitude centered at x_0 with the specified Lorentzian and Gaussian widths'},
                     # Voigt
                     'V_TH': {'Thermal Bremsstrahlung Model', '\n',
                              'T - Temperature (keV)', '\n',
                              'EM - Emission Measure (cm^-3)'},
                     'V_TH + PowerLaw': {'Addition of V_TH and Single Power Law', '\n',
                                         'T - Temperature (keV)', '\n',
                                         'EM - Emission Measure (cm^-3)', '\n',
                                         'Amplitude - Model amplitude at the reference energy', '\n',
                                         'Alpha - Power law index',
                                         '\n', 'Epivot – energie pivot (kEv)'
                                         },
                     'Neural Network': {'Neural Network model', '\n', },
                     'PowerLawCutoffFix': {'Power law model with fix cutoff', '\n',
                                           'amplitude – model amplitude at the reference energy', '\n',
                                           'Ec – Cutoff energy', '\n',
                                           'alpha – power law index'
                                           },
                     'PowerLawCutoffFree': {'Power law model with free cutoff', '\n',
                                            'amplitude – model amplitude at the reference energy', '\n',
                                            'Ec – Cutoff energy', '\n',
                                            'alpha – power law index'
                                            },

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
        # return hdulist[1].data

    # @staticmethod
    # def editEnergy(p1):
    #    """Call new class to edit spec_data axis"""
    #    new_window.Set_Energy(p1)

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
                for line in sorted(list(self.list[selected_name])):
                    self.list_selection.insert(END, line)
            else:
                self.list_selection.insert(END, "Aucune information disponible.")

            # Désactive la Listbox info (pour la rendre non cliquable)
            self.list_selection.config(state='disabled')

        except Exception as e:
            print("Erreur dans onSelect :", e)

    def update_file_list(self, file_list):
        """Updating the frame (in information:) and adding new function description, related to the user choice"""
        self.list_selection.delete(0, END)
        for i in file_list:
            self.list_selection.insert(END, i)

    def findfiles(self, val):
        """Finding the information related to the function name"""
        self.sender = val.widget

    def destroy5(self):
        """Closing 'SPEX Fit Options' window"""
        self.top2.destroy()

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

        def on_no():
            win.destroy()

        button_frame = Frame(win)
        button_frame.pack(pady=10)

        Button(button_frame, text="Yes", width=10, command=on_yes).pack(side="left", padx=5)
        Button(button_frame, text="No", width=10, command=on_no).pack(side="left", padx=5)

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

    def _apply_param_bounds(self, model, model_key: str):
        """
        Applique les valeurs initiales + bornes (min, max).
        """
        values_map = self.user_param_values.get(
            model_key, Fitting.default_param_values.get(model_key, {})
        )
        bounds_map = self.user_param_bounds.get(
            model_key, Fitting.default_param_bounds.get(model_key, {})
        )

        for pname in values_map.keys():
            if hasattr(model, pname):
                par = getattr(model, pname)
                init_val = values_map[pname]
                lo, hi = bounds_map.get(pname, (None, None))
                try:
                    if lo is not None:
                        par.min = lo
                    else:
                        par.min = -np.inf

                    if hi is not None:
                        par.max = hi
                    else:
                        par.max = np.inf
                    # try:
                    #     par.bounds = (lo, hi)
                    # except Exception:
                    #     pass
                    par.value = init_val
                except Exception as exc:
                    print(f"⚠️ Failed to set {pname}: {exc}")

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

        # def fit_unconstrained_then_bounded(self, model_template, x_fit, y_fit, y_err, param_names, bounds_map=None, initial_values=None):
        """
        1) Fit non-borné (LevMar) en partant des valeurs initiales fournies (initial_values ou celles du modèle).
        2) Si la solution non-bornée satisfait les bornes -> on la renvoie.
        3) Sinon -> on lance un fit borné (scipy.least_squares) démarrant depuis la solution non-bornée.
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

        # 1) Fit non-borné (LevMar)
        fitter = LevMarLSQFitter()
        try:
            fitted_nc = fitter(copy.deepcopy(model_template), x_fit, y_fit, weights=1.0 / (y_err + 1e-30))
        except Exception as e:
            print("⚠️ Unconstrained LevMar fit failed:", e)
            # fallback : retourner le template (ou on peut tenter directement least_squares)
            return copy.deepcopy(model_template)

        # extraire les valeurs non-bornées
        uncon_values = [getattr(fitted_nc, p).value for p in param_names]

        # 2) Pas de bornes fournies -> utiliser solution non-bornée
        if not bounds_map:
            return fitted_nc

        # 3) Vérifier si la solution non-bornée respecte les bornes
        in_bounds = True
        tol = 1e-12
        for i, pname in enumerate(param_names):
            lo, hi = bounds_map.get(pname, (None, None))
            val = uncon_values[i]
            if (lo is not None and val < (lo - tol)) or (hi is not None and val > (hi + tol)):
                in_bounds = False
                break

        if in_bounds:
            return fitted_nc

        # 4) Sinon, fit borné via least_squares en démarrant de uncon_values
        x0 = np.array(uncon_values, dtype=float)
        lb = []
        ub = []
        for pname in param_names:
            lo, hi = bounds_map.get(pname, (None, None))
            lb.append(-np.inf if lo is None else lo)
            ub.append(np.inf if hi is None else hi)
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)

        def residuals(vec):
            m = self._params_vector_to_model(model_template, param_names, vec)
            y_model = m(x_fit)
            return (y_model - y_fit) / (y_err + 1e-30)

        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-8, ftol=1e-8, max_nfev=2000)
        except Exception as e:
            print("⚠️ least_squares failed:", e)
            return fitted_nc

        best_vec = res.x if res is not None else uncon_values
        fitted_bounded = self._params_vector_to_model(model_template, param_names, best_vec)
        return fitted_bounded

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

        # 3) Vérifier si la solution non-bornée respecte les bornes
        # in_bounds = True
        # tol = 1e-12
        # for i, pname in enumerate(param_names):
        #     lo, hi = bounds_map.get(pname, (None, None))
        #     val = uncon_values[i]
        #     if (lo is not None and val < (lo - tol)) or (hi is not None and val > (hi + tol)):
        #         in_bounds = False
        #         break

        # if in_bounds:
        #     return fitted_nc

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
        """Selection depending on Plot Units and Function Model
          Predefine Plotting Data in energy_data and spec_data
          We equate three components to rate_data, counts_data, flux_data. The value of energy_data is the same for all cases
          energy_data - independent variable, nominally energy in keV
          spec_data - Plot Unit"""

        selection = self.lbox.curselection()
        if not selection:
            messagebox.showwarning("No Model Selected", "Please select a fit model before clicking 'Do Fit'.")
            return

        # load chosen file in Select Plotting section
        fname = Fitting.fname
        rname = Fitting.rname
        if fname is None and rname is None:  # if file not choosen, print
            messagebox.showwarning("No File Selected", "Please, choose input file.")

        else:

            if self.show_db_var.get():

                index_start = background.BackgroundWindow.DATA_BKG_START
                index_end = background.BackgroundWindow.DATA_BKG_END

                background_slice = self.counts[index_start:index_end + 1, :]
                bkg_vector = np.mean(background_slice, axis=0)
                counts_bkg_removed = self.counts - bkg_vector
                self.data_background = np.where(counts_bkg_removed > 0, counts_bkg_removed, 1e-5)

                used_data = self.data_background
                absolute_name = "Data - Background"

            else:

                used_data = self.counts
                absolute_name = "Data"
                # print("test data:")

            counts_all = np.mean(used_data, axis=0)
            counts_err_all = np.mean(self.counts_err, axis=0)
            exposure = np.mean(self.time_del)
            e_low_det_all = self.e_low_det
            e_high_det_all = self.e_high_det

            # Read SRM data
            e_low_true = self.e_low_true
            e_high_true = self.e_high_true
            matrix = self.matrix

            # --- use mask channel to avoid shape problem ---
            usable_channels = np.arange(min(matrix.shape[1], len(e_low_det_all)))

            counts = counts_all[usable_channels]
            counts_err = counts_err_all[usable_channels]
            e_low_det = e_low_det_all[usable_channels]
            e_high_det = e_high_det_all[usable_channels]

            # remove NaN values and negative values from counts and counts_err
            valid = (counts_err > 0) & np.isfinite(counts_err) & np.isfinite(counts)

            counts = counts[valid]
            counts_err = counts_err[valid]
            x_fake = np.zeros_like(counts)  # X fake for plotting must be same shape as counts

            matrix = matrix[:, valid]  # matrix is 2D array, so we need to remove the same channels from it
            e_low_det = e_low_det[valid]
            e_high_det = e_high_det[valid]

            # matrix[:] = 1.0  # temporary for testing fitting procedure without matrix
            # print('after setting matrix to 1.0:')
            # print('min', np.min(matrix), 'max', np.max(matrix), 'shape', np.shape(matrix))

            edges_det = np.append(e_low_det, e_high_det[-1])
            dE_det = np.diff(edges_det)

            Edges_photon = np.append(e_low_true, e_high_true[-1])

            # --- Fitting avec LevMarLSQFitter ---

            # === fitting range ===
            fit_Emin = self.energy_min_var.get()  # keV
            fit_Emax = self.energy_max_var.get()  # keV
            # fit_Emin = 10.0  # keV
            # fit_Emax = 20.0  # keV

            # mask for fitting range
            fit_mask = (edges_det[:-1] >= fit_Emin) & (edges_det[1:] <= fit_Emax)

            x_fit = x_fake[fit_mask]
            counts_fit = counts[fit_mask]
            counts_err_fit = counts_err[fit_mask]
            matrix_fit = matrix[:, fit_mask]

            # Counts
            mean_counts = counts
            mean_counts_err = counts_err

            # Rate
            rate = mean_counts / exposure
            rate_err = mean_counts_err / exposure

            # Flux (photons / s / cm² / keV)
            flux = rate / (self.area * dE_det)
            flux_err = rate_err / (self.area * dE_det)

            # Unit selection
            unit = self.var.get()

            if unit == 'Rate':
                y_data = rate
                y_err = rate_err
                y_label = "Rate [counts / (s keV)]"
            elif unit == 'Counts':
                y_data = mean_counts
                y_err = mean_counts_err
                y_label = "Counts (Counts)'"
            elif unit == 'Flux':
                y_data = flux
                y_err = flux_err
                y_label = "Flux (Counts/s/cm²/keV)"
            else:
                raise ValueError("Choose unit = 'rate', 'counts' ou 'flux'")

            plt.figure()
            plt.step(edges_det[:-1], y_data, where='mid', label=f'{absolute_name} ({unit})', color='red')
            # plt.axvspan(self.e_low_det[self.background_channel_start], self.e_high_det[self.background_channel_end - 1], 
            #             color='gray', alpha=0.3, label="Background Interval")

            if self.lbox.curselection()[0] == 0:

                self.fit_model = 'Power Law'
                model_key = "PowerLaw1D"
                # model_fit = ForwardFolded.PowerLaw(e_low_true, e_high_true, matrix_fit, exposure)

                # # # Apply user-defined initial values and bounds
                # self._apply_param_bounds(model_fit, model_key)

                # # # Fitting
                # fitter = LevMarLSQFitter()
                # fitted_model = fitter(model_fit, x_fit, counts_fit / exposure,
                #                     weights=1.0 / (counts_err_fit / exposure))

                # residuals = (counts_fit / exposure - fitted_model(x_fit)) / (counts_err_fit / exposure + 1e-30)
                # chi2 = np.sum(residuals**2)
                # print("Erreur finale (chi²) :", chi2)

                # 1) préparer un template (sans bornes forcées) et appliquer les valeurs initiales souhaitées
                E_pivot_val = self.user_param_values.get(model_key, {}).get("E_pivot", 100.0)
                model_template = ForwardFolded.PowerLaw(e_low_true, e_high_true, matrix_fit, exposure,
                                                        E_pivot=E_pivot_val)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))
                bounds_map = self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {}))

                # 2) fit robuste : unconstrained puis bounded si nécessaire
                fitted_model = self.fit_unconstrained_then_bounded(
                    model_template,
                    x_fit,
                    counts_fit / exposure,  # y_fit
                    counts_err_fit / exposure,  # y_err
                    ["amplitude", "alpha"],
                    bounds_map,
                    initial_values
                )

                # Récupérer les paramètres
                amplitude = fitted_model.amplitude.value
                alpha = fitted_model.alpha.value

                print(
                    f"Fitted Power Law: amplitude = {amplitude:.2e}, alpha = {alpha:.2f}, E_pivot = {E_pivot_val:.2f} keV")

                # Construire modèle complet pour affichage (optionnel — comme tu fais ailleurs)
                model_display = ForwardFolded.PowerLaw(e_low_true, e_high_true, matrix, exposure, E_pivot=E_pivot_val)
                try:
                    model_display.amplitude = fitted_model.amplitude
                    model_display.alpha = fitted_model.alpha
                except Exception:
                    # safer: assign values
                    model_display.amplitude.value = amplitude
                    model_display.alpha.value = alpha

                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = (rate_modeled_full / dE_det)
                elif unit == 'Counts':
                    model_y = (rate_modeled_full * exposure)
                elif unit == 'Flux':
                    model_y = (rate_modeled_full / (self.area * dE_det))
                else:
                    raise ValueError("Unit most be choose")

                plt.step(edges_det[:-1], model_y, where='mid',
                         label='Fitted Model', color='blue')

                if self.show_params_var.get():
                    # show model parameters on the plot
                    plt.text(0.05, 0.4,
                             f"Power Law:\n amplitude = {amplitude:.2e}\n alpha = {alpha:.2f} \n E_pivot = {E_pivot_val:.2f} keV \n",
                             transform=plt.gca().transAxes,
                             fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.7))

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

                if self.show_photon_var.get():
                    # --- Photon ---
                    model_func = lambda E: amplitude * (E / E_pivot_val) ** (-alpha)
                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(0.05, 0.4,
                                 f"Power Law:\n amplitude = {amplitude:.2e}\n alpha = {alpha:.2f} \n E_pivot = {E_pivot_val:.2f} keV \n",
                                 transform=plt.gca().transAxes,
                                 fontsize=10,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.7))
                    plt.tight_layout()

            elif self.lbox.curselection()[0] == 1:

                # Create model
                self.fit_model = 'Broken Power Law'
                model_key = "BrokenPowerLaw1D"

                # model_fit = ForwardFolded.BrokenPowerLaw(e_low_true, e_high_true, matrix_fit, exposure)

                # # Apply user-defined initial values and bounds
                # self._apply_param_bounds(model_fit, model_key)

                # fitter = LevMarLSQFitter()
                # fitted_model = fitter(model_fit, x_fit, counts_fit / exposure,
                #                     weights=1.0 / (counts_err_fit / exposure))

                model_template = ForwardFolded.BrokenPowerLaw(e_low_true, e_high_true, matrix_fit, exposure)
                param_names = list(Fitting.default_param_values.get(model_key, {}).keys())
                internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
                user_bounds_map = self.user_param_bounds.get(model_key, internal_bounds_map)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

                # --- Lancer la logique "fit en deux étapes" ---
                fitted_model = self.fit_with_bounds_check(
                    model_template, x_fit, counts_fit / exposure, counts_err_fit / exposure,
                    param_names, model_key,
                    internal_bounds_map=internal_bounds_map,
                    user_bounds_map=user_bounds_map,
                    initial_values=initial_values
                )

                # Parameters
                amplitude = fitted_model.amplitude.value
                E_break = fitted_model.E_break.value
                alpha_1 = fitted_model.alpha_1.value
                alpha_2 = fitted_model.alpha_2.value

                # Modèle complet pour affichage sur tout le domaine
                model_display = ForwardFolded.BrokenPowerLaw(e_low_true, e_high_true, matrix, exposure)
                model_display.amplitude = fitted_model.amplitude
                model_display.alpha_1 = fitted_model.alpha_1
                model_display.alpha_2 = fitted_model.alpha_2
                model_display.E_break = fitted_model.E_break

                # Calcul du modèle simulé complet
                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = (rate_modeled_full / dE_det)
                elif unit == 'Counts':
                    model_y = (rate_modeled_full * exposure)
                elif unit == 'Flux':
                    model_y = (rate_modeled_full / (self.area * dE_det))
                else:
                    raise ValueError("Unit most be choose")

                plt.step(edges_det[:-1], model_y, where='mid',
                         label='Fitted Model', color='blue')

                if self.show_params_var.get():
                    # show model parameters on the plot
                    plt.text(0.05, 0.4,
                             f"Broken Power Law:\n amplitude = {amplitude:.2e}\n E_break = {E_break:.2f} \n Alpha_1 = {alpha_1:.2e}\n Alpha_2 = {alpha_2:.2f} \n",
                             transform=plt.gca().transAxes,
                             fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.7))

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

                if self.show_photon_var.get():
                    # --- Photon ---
                    model_func = lambda E: amplitude * np.where(E < E_break, (E / E_break) ** (-alpha_1),
                                                                (E / E_break) ** (-alpha_2))
                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(0.05, 0.4,
                                 f"Broken Power Law:\n amplitude = {amplitude:.2e}\n E_break = {E_break:.2f} \n Alpha_1 = {alpha_1:.2e}\n Alpha_2 = {alpha_2:.2f} \n",
                                 transform=plt.gca().transAxes,
                                 fontsize=10,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.7))
                    plt.tight_layout()

            elif self.lbox.curselection()[0] == 2:

                # Create model
                self.fit_model = 'Exponential Power Law'
                model_key = "Single Power Law Times an Exponential"

                # model_fit = ForwardFolded.ExpPowerLaw(e_low_true, e_high_true, matrix_fit, exposure)

                # # Apply user-defined initial values and bounds
                # self._apply_param_bounds(model_fit, model_key)

                # # Fitting
                # fitter = LevMarLSQFitter()
                # fitted_model = fitter(model_fit, x_fit, counts_fit / exposure,
                #                     weights=1.0 / (counts_err_fit / exposure))

                model_template = ForwardFolded.ExpPowerLaw(e_low_true, e_high_true, matrix_fit, exposure)
                param_names = list(Fitting.default_param_values.get(model_key, {}).keys())
                internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
                user_bounds_map = self.user_param_bounds.get(model_key, internal_bounds_map)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

                # --- Lancer la logique "fit en deux étapes" ---
                fitted_model = self.fit_with_bounds_check(
                    model_template, x_fit, counts_fit / exposure, counts_err_fit / exposure,
                    param_names, model_key,
                    internal_bounds_map=internal_bounds_map,
                    user_bounds_map=user_bounds_map,
                    initial_values=initial_values
                )

                # Parameters
                p0 = fitted_model.p0.value
                p1 = fitted_model.p1.value
                p2 = fitted_model.p2.value
                e3 = fitted_model.e3.value
                e4 = fitted_model.e4.value

                # Modèle complet pour affichage sur tout le domaine
                model_display = ForwardFolded.ExpPowerLaw(e_low_true, e_high_true, matrix, exposure)
                model_display.p0 = fitted_model.p0
                model_display.p1 = fitted_model.p1
                model_display.p2 = fitted_model.p2
                model_display.e3 = fitted_model.e3
                model_display.e4 = fitted_model.e4

                # Calcul du modèle simulé complet
                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = (rate_modeled_full / dE_det)
                elif unit == 'Counts':
                    model_y = (rate_modeled_full * exposure)
                elif unit == 'Flux':
                    model_y = (rate_modeled_full / (self.area * dE_det))
                else:
                    raise ValueError("Unit most be choose")

                plt.step(edges_det[:-1], model_y, where='mid',
                         label='Fitted Model', color='blue')

                if self.show_params_var.get():
                    # show model parameters on the plot
                    plt.text(0.05, 0.4,
                             f"Exponential Power Law:\n p0 = {p0:.2e}\n p1 = {p1:.2f} \n p2 = {p2:.2f} \n e3 = {e3:.2f} \n e4 = {e4:.2f} \n",
                             transform=plt.gca().transAxes,
                             fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.7))

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

                if self.show_photon_var.get():
                    # --- Photon ---
                    model_func = lambda E: (p0 * (E / p2) ** p1) * np.exp(e3 - E / e4)
                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(0.05, 0.4,
                                 f"Exponential Power Law:\n p0 = {p0:.2e}\n p1 = {p1:.2f} \n p2 = {p2:.2f} \n e3 = {e3:.2f} \n e4 = {e4:.2f} \n",
                                 transform=plt.gca().transAxes,
                                 fontsize=10,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.7))
                    plt.tight_layout()

            elif self.lbox.curselection()[0] == 3:

                self.fit_model = 'VTH'
                model_key = "V_TH"
                # model_fit = ForwardFolded.VTH(e_low_true, e_high_true, matrix_fit, exposure)

                # # Apply user-defined initial values and bounds
                # self._apply_param_bounds(model_fit, model_key)

                # # Fitting
                # fitter = LevMarLSQFitter()
                # fitted_model = fitter(model_fit, x_fit, counts_fit / exposure,
                #                     weights=1.0 / (counts_err_fit / exposure))

                model_template = ForwardFolded.VTH(e_low_true, e_high_true, matrix_fit, exposure)
                param_names = list(Fitting.default_param_values.get(model_key, {}).keys())
                internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
                user_bounds_map = self.user_param_bounds.get(model_key, internal_bounds_map)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

                # --- Lancer la logique  ---
                fitted_model = self.fit_with_bounds_check(
                    model_template, x_fit, counts_fit / exposure, counts_err_fit / exposure,
                    param_names, model_key,
                    internal_bounds_map=internal_bounds_map,
                    user_bounds_map=user_bounds_map,
                    initial_values=initial_values
                )

                T = fitted_model.T.value
                EM = fitted_model.EM.value

                model_display = ForwardFolded.VTH(e_low_true, e_high_true, matrix, exposure)
                model_display.T = fitted_model.T
                model_display.EM = fitted_model.EM

                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = rate_modeled_full / dE_det
                elif unit == 'Counts':
                    model_y = rate_modeled_full * exposure
                elif unit == 'Flux':
                    model_y = rate_modeled_full / (self.area * dE_det)

                plt.step(edges_det[:-1], model_y, where='mid', label='Fitted VTH Model', color='blue')

                if self.show_params_var.get():
                    plt.text(0.05, 0.4,
                             f"V_TH Model:\n T = {T:.2f} keV\n EM = {EM:.2e} cm⁻³",
                             transform=plt.gca().transAxes,
                             fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.7))

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

                # === AFFICHAGE PHOTONIQUE ===
                if self.show_photon_var.get():
                    # Constantes physiques
                    gff = 1.2
                    A = 1.07e-42 * gff
                    k_B_keV = 8.617333262e-8  # erg/K in keV

                    T_keV = (T * k_B_keV) / 1.60218e-9  # conversion K -> keV

                    model_func = lambda E: (A * EM) / (E * np.sqrt(T)) * np.exp(-E / T_keV)
                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(0.05, 0.4,
                                 f"V_TH Model:\n T = {T:.2e} K\n EM = {EM:.2e} cm⁻³",
                                 transform=plt.gca().transAxes,
                                 fontsize=10,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.7))
                    plt.tight_layout()

            elif self.lbox.curselection()[0] == 4:

                self.fit_model = 'V_TH + Power Law'
                model_key = "V_TH + PowerLaw"
                # model_fit = ForwardFolded.VTHPlusPowerLaw(e_low_true, e_high_true, matrix_fit, exposure)

                # # Apply user-defined initial values and bounds
                # self._apply_param_bounds(model_fit, model_key)

                # fitter = LevMarLSQFitter()
                # fitted_model = fitter(model_fit, x_fit, counts_fit / exposure,
                #                     weights=1.0 / (counts_err_fit / exposure))

                E_pivot_val = self.user_param_values.get(model_key, {}).get("E_pivot", 100.0)
                model_template = ForwardFolded.VTHPlusPowerLaw(e_low_true, e_high_true, matrix_fit, exposure,
                                                               E_pivot=E_pivot_val)
                param_names = [
                    p for p in Fitting.default_param_values[model_key].keys()
                    if p != "E_pivot"
                ]
                internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
                user_bounds_map = self.user_param_bounds.get(model_key, internal_bounds_map)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

                # --- Lancer la logique  ---
                fitted_model = self.fit_with_bounds_check(
                    model_template, x_fit, counts_fit / exposure, counts_err_fit / exposure,
                    param_names, model_key,
                    internal_bounds_map=internal_bounds_map,
                    user_bounds_map=user_bounds_map,
                    initial_values=initial_values
                )

                # Paramètres du modèle
                EM = fitted_model.EM.value
                T = fitted_model.T.value
                amplitude = fitted_model.amplitude.value
                alpha = fitted_model.alpha.value

                # Création du modèle à afficher sur tout le domaine
                model_display = ForwardFolded.VTHPlusPowerLaw(e_low_true, e_high_true, matrix, exposure,
                                                              E_pivot=E_pivot_val)
                model_display.EM = fitted_model.EM
                model_display.T = fitted_model.T
                model_display.amplitude = fitted_model.amplitude
                model_display.alpha = fitted_model.alpha

                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = rate_modeled_full / dE_det
                elif unit == 'Counts':
                    model_y = rate_modeled_full * exposure
                elif unit == 'Flux':
                    model_y = rate_modeled_full / (self.area * dE_det)

                plt.step(edges_det[:-1], model_y, where='mid', label='Fitted VTH Model', color='blue')

                if self.show_params_var.get():
                    if self.show_params_var.get():
                        plt.text(
                            0.06, 0.5,
                            f"V_TH + Power Law:\n"
                            f"T  = {T:.2e} keV\n"
                            f"EM = {EM:.2e} cm⁻³\n"
                            f"amplitude = {amplitude:.2e}\n"
                            f"alpha     = {alpha:.2f}\n"
                            f"E_pivot = {E_pivot_val:.2f} keV",
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7)
                        )

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

                # === AFFICHAGE PHOTONIQUE ===
                if self.show_photon_var.get():
                    model_func = lambda E: (
                            (1.07e-42 * 1.2 * EM) / (E * np.sqrt(max(1e-3, T))) * np.exp(-E / T) +
                            amplitude * (E / 100.0) ** (-alpha)
                    )

                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(
                            0.05, 0.4,
                            f"V_TH + Power Law:\n"
                            f"T  = {T:.2e} keV\n"
                            f"EM = {EM:.2e} cm⁻³\n"
                            f"amplitude = {amplitude:.2e}\n"
                            f"alpha     = {alpha:.2f}\n"
                            f"E_pivot = {E_pivot_val:.2f} keV",
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7)
                        )

                    plt.tight_layout()

            elif self.lbox.curselection()[0] == 5:

                self.fit_model = 'PowerLawCutoffFix'
                model_key = "PowerLawCutoffFix"

                E_cut_val = self.user_param_values.get(model_key, {}).get("E_cut", 10.0)
                E_pivot_val = self.user_param_values.get(model_key, {}).get("E_pivot", 100.0)
                model_template = ForwardFolded.PowerLawCutoffFix(e_low_true, e_high_true, matrix_fit, exposure, E_cut_val, E_pivot_val)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))
                bounds_map = self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {}))

                fitted_model = self.fit_unconstrained_then_bounded(
                    model_template,
                    x_fit,
                    counts_fit / exposure,  # y_fit
                    counts_err_fit / exposure,  # y_err
                    ["amplitude", "alpha"],
                    bounds_map,
                    initial_values
                )

                amplitude = fitted_model.amplitude.value
                alpha = fitted_model.alpha.value

                model_display = ForwardFolded.PowerLawCutoffFix(e_low_true, e_high_true, matrix, exposure, E_cut_val, E_pivot_val)
                try:
                    model_display.amplitude = fitted_model.amplitude
                    model_display.alpha = fitted_model.alpha
                except Exception:
                    model_display.amplitude.value = amplitude
                    model_display.alpha.value = alpha

                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = (rate_modeled_full / dE_det)
                elif unit == 'Counts':
                    model_y = (rate_modeled_full * exposure)
                elif unit == 'Flux':
                    model_y = (rate_modeled_full / (self.area * dE_det))
                else:
                    raise ValueError("Unit most be choose")

                # Apply cutoff in plotting range E >= E_cut_val (cutoff value) and E <= fit_Emax (fitting max value)
                fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[1:] <= fit_Emax)
                model_y = np.where(fit_mask_cutoff, model_y, 0)

                plt.step(edges_det[:-1], model_y, label='Fitted Model', color='blue')

                if self.show_params_var.get():
                    # show model parameters on the plot
                    plt.text(0.05, 0.4,
                             f"Power Law:\n amplitude = {amplitude:.2e}\n alpha = {alpha:.2f} \n E_pivot = {E_pivot_val:.2f} keV \n E_cut = {E_cut_val:.2f}"
                             ,
                             transform=plt.gca().transAxes,
                             fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.7))

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

                if self.show_photon_var.get():

                    # --- Photon ---
                    # Should be verify because it's not true
                    model_func = lambda E: np.where(E >= E_cut_val, amplitude * (E / E_pivot_val) ** (-alpha), 0.0)
                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(0.05, 0.4,
                                 f"Power Law:\n amplitude = {amplitude:.2e}\n alpha = {alpha:.2f} \n E_pivot = {E_pivot_val:.2f} keV \n E_cut = {E_cut_val:.2f}",
                                 transform=plt.gca().transAxes,
                                 fontsize=10,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.7))
                    plt.tight_layout()

            ####################
            # PARTIE NON VALIDEE (6 & 7)
            ####################

            elif self.lbox.curselection()[0] == 6:

                self.fit_model = 'PowerLawCutoffFree'
                model_key = "PowerLawCutoffFree"

                y_fit = counts_fit / exposure
                y_err = counts_err_fit / exposure

                E_pivot_val = self.user_param_values.get(model_key, {}).get("E_pivot", 100.0)
                model_template = ForwardFolded.PowerLawCutoffFree(e_low_true, e_high_true, matrix_fit, exposure, E_pivot_val)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))
                bounds_map = self.user_param_bounds.get(model_key, Fitting.default_param_bounds.get(model_key, {}))

                fitted_model = self.fit_unconstrained_then_bounded(
                    model_template,
                    x_fit,
                    y_fit,
                    y_err,
                    ["amplitude", "alpha"],
                    bounds_map,
                    initial_values
                )


                amplitude = fitted_model.amplitude.value
                alpha = fitted_model.alpha.value
                E_cut_val = fitted_model.E_cut.value

                model_display = ForwardFolded.PowerLawCutoffFree(e_low_true, e_high_true, matrix, exposure, E_pivot_val)
                try:
                    model_display.amplitude = fitted_model.amplitude
                    model_display.alpha = fitted_model.alpha
                except Exception:
                    model_display.amplitude.value = amplitude
                    model_display.alpha.value = alpha

                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = (rate_modeled_full / dE_det)
                elif unit == 'Counts':
                    model_y = (rate_modeled_full * exposure)
                elif unit == 'Flux':
                    model_y = (rate_modeled_full / (self.area * dE_det))
                else:
                    raise ValueError("Unit most be choose")

                # Apply cutoff in plotting range E >= E_cut_val (cutoff value) and E <= fit_Emax (fitting max value)
                fit_mask_cutoff = (edges_det[1:] > E_cut_val) & (edges_det[:-1] < fit_Emax)
                model_y = np.where(fit_mask_cutoff, model_y, 0)

                plt.step(edges_det[:-1], model_y, label='Fitted Model', color='blue')

                if self.show_params_var.get():
                    # show model parameters on the plot
                    plt.text(0.05, 0.4,
                             f"Power Law:\n amplitude = {amplitude:.2e}\n alpha = {alpha:.2f} \n E_pivot = {E_pivot_val:.2f} keV \n E_cut = {E_cut_val:.2f}",
                             transform=plt.gca().transAxes,
                             fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.7))

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

                if self.show_photon_var.get():

                    # --- Photon ---
                    # Should be verify because it's not true
                    model_func = lambda E: np.where(E >= E_cut_val, amplitude * (E / E_pivot_val) ** (-alpha), 0.0)
                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(0.05, 0.4,
                                 f"Power Law:\n amplitude = {amplitude:.2e}\n alpha = {alpha:.2f} \n E_pivot = {E_pivot_val:.2f} keV \n E_cut = {E_cut_val:.2f}",
                                 transform=plt.gca().transAxes,
                                 fontsize=10,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.7))
                    plt.tight_layout()


            elif self.lbox.curselection()[0] == 7:

                messagebox.showwarning("Not yet available", "Please be patient.")
                return

                self.fit_model = 'V_TH + PowerLawCutoffFix'
                model_key = "V_TH + PowerLawCutoffFix"

                E_cut_val = self.user_param_values.get(model_key, {}).get("E_cut", 10.0)
                model_template = ForwardFolded.VTHPlusPowerLawCutoffFix(e_low_true, e_high_true, matrix_fit, exposure,
                                                                        E_cut_val)
                param_names = [
                    p for p in Fitting.default_param_values[model_key].keys()
                    if p != "E_cut"
                ]
                internal_bounds_map = Fitting.default_param_bounds.get(model_key, {})
                user_bounds_map = self.user_param_bounds.get(model_key, internal_bounds_map)
                initial_values = self.user_param_values.get(model_key, Fitting.default_param_values.get(model_key, {}))

                fitted_model = self.fit_with_bounds_check(
                    model_template, x_fit, counts_fit / exposure, counts_err_fit / exposure,
                    param_names, model_key,
                    internal_bounds_map=internal_bounds_map,
                    user_bounds_map=user_bounds_map,
                    initial_values=initial_values
                )

                # Paramètres du modèle
                EM = fitted_model.EM.value
                T = fitted_model.T.value
                amplitude = fitted_model.amplitude.value
                alpha = fitted_model.alpha.value

                # Création du modèle à afficher sur tout le domaine
                model_display = ForwardFolded.VTHPlusPowerLawCutoffFix(e_low_true, e_high_true, matrix, exposure,
                                                                       E_cut_val)
                model_display.EM = fitted_model.EM
                model_display.T = fitted_model.T
                model_display.amplitude = fitted_model.amplitude
                model_display.alpha = fitted_model.alpha

                rate_modeled_full = model_display(x_fake)

                if unit == 'Rate':
                    model_y = rate_modeled_full / dE_det
                elif unit == 'Counts':
                    model_y = rate_modeled_full * exposure
                elif unit == 'Flux':
                    model_y = rate_modeled_full / (self.area * dE_det)

                # Apply cutoff in plotting range E >= E_cut_val (cutoff value) and E <= fit_Emax (fitting max value)
                # fit_mask_cutoff = (edges_det[:-1] >= E_cut_val) & (edges_det[1:] <= fit_Emax)
                # model_y = np.where(fit_mask_cutoff, model_y, 0)

                plt.step(edges_det[:-1], model_y, where='mid', label='Fitted VTH Model', color='blue')

                if self.show_params_var.get():
                    if self.show_params_var.get():
                        plt.text(
                            0.06, 0.5,
                            f"V_TH + Power Law:\n"
                            f"T  = {T:.2e} keV\n"
                            f"EM = {EM:.2e} cm⁻³\n"
                            f"amplitude = {amplitude:.2e}\n"
                            f"alpha     = {alpha:.2f}\n"
                            f"E_cut = {E_cut_val:.2f} keV",
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7)
                        )

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

                # === AFFICHAGE PHOTONIQUE ===
                # Should be verify because it's not true
                if self.show_photon_var.get():
                    model_func = lambda E: (
                            (1.07e-42 * 1.2 * EM) / (E * np.sqrt(max(1e-3, T))) * np.exp(-E / T) +
                            amplitude * (E / 100.0) ** (-alpha)
                    )

                    flux_photons = np.array([
                        ForwardFolded.integrate_flux(e1, e2, model_func)
                        for e1, e2 in zip(e_low_true, e_high_true)
                    ])

                    plt.figure()
                    plt.step(Edges_photon[:-1], flux_photons, where='mid',
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
                        plt.text(
                            0.05, 0.4,
                            f"V_TH + Power Law:\n"
                            f"T  = {T:.2e} keV\n"
                            f"EM = {EM:.2e} cm⁻³\n"
                            f"amplitude = {amplitude:.2e}\n"
                            f"alpha     = {alpha:.2f}\n"
                            f"E_cut = {E_cut_val:.2f} keV",
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7)
                        )

                    plt.tight_layout()

            plt.show()
