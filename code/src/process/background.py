import datetime
import os
import re
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename

import numpy as np
from astropy.io import fits

from . import inputWindow
from .graphics import IntervalSelector
from .io import loader

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DT_FMT_MS  = "%Y-%m-%dT%H:%M:%S.%f"
_DT_FMT     = "%Y-%m-%dT%H:%M:%S"
_BAND_COLORS = ['blue', 'red', 'green', 'black', 'orange']
_UNIT_CHOICES   = ('Rate', 'Counts', 'Flux')
_METHOD_CHOICES = ('Median', 'Mean', '1Poly', '2Poly', '3Poly', 'Exp')
_BAND_LIST      = ('1', '2', '3', '4', '5')

# Mapping: total_seconds threshold → tick interval in seconds
_TICK_INTERVALS = [
    (1_800,   120),
    (3_600,   480),
    (10_800,  1_200),
    (28_800,  1_800),
    (43_200,  3_600),
    (86_400,  7_200),
    (172_800, 10_800),
    (432_000, 21_600),
    (864_000, 43_200),
]
_TICK_DEFAULT = 86_400


class BackgroundWindow:

    DATA_BKG_SELECTED = False   # If background data has been selected by user
    DATA_BKG_RESULT   = None    # Result of the background calculation
    DATA_BKG_START    = None    # Starting index of the background data
    DATA_BKG_END      = None    # Ending index of the background data

    def __init__(self, root=None, show=True):
        """Initialise the background calculator window.

        Parameters
        ----------
        root : optional
            Parent Tk root.
        show : bool
            When True, build and display the Toplevel window.
        """
        self.energies      = None
        self.root          = root
        self.hdul          = None
        self.hdulist       = None
        self.name          = loader.activeFile()

        self.counts      = None
        self.counts_err  = None
        self.times       = None
        self.del_times   = None
        self.data        = None
        self.data_err    = None
        self.del_data    = None
        self.bkg         = None
        self.data_bkg    = None
        self.area        = 6   # cm²

        self.nb_bands      = StringVar()
        self.lower_bands   = []
        self.upper_bands   = []
        self.energies_low  = []
        self.energies_high = []
        self.rounded       = int()

        self.start_date = []
        self.end_date   = []
        self.date_day   = None
        self.date_time  = None
        self.start_time = int()
        self.end_time   = int()

        self.choice_bands = None
        self.list_bands   = _BAND_LIST

        self.type       = str()
        self.unit_var   = StringVar()
        self.unit_choices = _UNIT_CHOICES

        self.method_list    = []
        self.method_var     = []
        self.method_choices = _METHOD_CHOICES

        if show:
            self._init_show_state()
            self.build_bkg_window()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_show_state(self):
        """Initialise all attributes that are only needed when the window is shown."""
        self.bkg_window = None
        self.frame1 = self.frame2 = self.frame3 = self.frame4 = self.frame5 = None

        self.state_bkg  = DISABLED
        self.state_time = DISABLED
        self.backup_bkg = 0

        # Labels / text widgets
        self.label1 = self.label2 = self.label4 = self.label5 = None
        self.label_filename = None
        self.text_min_energy = self.text_max_energy = None
        self.text_start_time = self.text_end_time   = None
        self.text_noise = None

        # Menus & buttons
        self.menu_units              = None
        self.btn_time_profile_plotting = None
        self.close                   = None
        self.text_filename           = None
        self.btn_browse              = None
        self.btn_og_data = self.btn_bkg = self.btn_data_bkg = None
        self.btn_error   = self.btn_sep_times               = None

        self.var_og_data  = IntVar(value=1)
        self.var_bkg      = IntVar()
        self.var_data_bkg = IntVar()
        self.var_error    = IntVar()
        self.var_sep_times = IntVar(value=1)

        # Per-band widget lists
        self.energy_min      = []
        self.energy_max      = []
        self.time_min        = []
        self.time_max        = []
        self.btn_graphical   = []
        self.btn_spectrogram = []
        self.method_selection = []
        self.energy_min_var  = []
        self.energy_max_var  = []

        self.energy_min_list = []
        self.energy_max_list = []
        self.time_min_list   = []
        self.time_max_list   = []

        self.bkg_start_time  = []
        self.bkg_end_time    = []
        self.bkg_start_index = []
        self.bkg_end_index   = []

        self.title       = str()
        self.xlabel      = str()
        self.ylabel      = str()
        self.time_scale  = []
        self.plot_label  = []
        self.data_label  = []
        self.color       = list(_BAND_COLORS)

    # =================== Building window ===================

    def build_bkg_window(self):
        """Build the main window skeleton and the first frame."""
        self.bkg_window = Toplevel()
        self.bkg_window.title('STIX Background Options')
        self.bkg_window.geometry("1000x600")
        Label(self.bkg_window, text="Select Background",
              fg="black", font="Helvetica 12 bold italic").pack()

        self.close = Button(self.bkg_window, text="Close", command=self.destroy)
        self.close.place(relx=0.5, rely=0.95, anchor='center')

        # --- Frame 1: Data & Unit Plotting ---
        self.frame1 = LabelFrame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame1.place(relx=0.025, rely=0.05, relheight=0.15, relwidth=0.95)

        Label(self.frame1, text="Spectrum or Image File: ").place(relx=0.02, rely=0.25, anchor=W)
        self.text_filename = Entry(self.frame1, width=20)
        self.text_filename.place(relx=0.19, rely=0.25, relheight=0.3, relwidth=0.57, anchor=W)

        if self.name is not None:
            self.text_filename.insert(0, self.name)
            self.open_file(self.name)
        else:
            self.text_filename.insert(0, "No file chosen")


        self.btn_browse = Button(self.frame1, text='Browse ->', command=self.open_file)
        self.btn_browse.place(relx=0.78, rely=0.25, anchor=W)

        Label(self.frame1, text="Plot time profile for: ").place(relx=0.02, rely=0.65, anchor=W)

        Checkbutton(self.frame1, text="Data",
                    variable=self.var_og_data).place(relx=0.19, rely=0.65, anchor=W)
        self.btn_bkg = Checkbutton(self.frame1, text="Background",
                                   variable=self.var_bkg, command=self.disable_bkg)
        self.btn_bkg.place(relx=0.26, rely=0.65, anchor=W)
        self.btn_data_bkg = Checkbutton(self.frame1, text="Data-Background",
                                        variable=self.var_data_bkg, command=self.disable_bkg)
        self.btn_data_bkg.place(relx=0.37, rely=0.65, anchor=W)
        Checkbutton(self.frame1, text="Error",
                    variable=self.var_error).place(relx=0.51, rely=0.65, anchor=W)

    def build_second_frame(self):
        """Build the energy-band selection frame."""
        self.frame2 = Canvas(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame2.place(relx=0.025, rely=0.2, relheight=0.3, relwidth=0.15)

        Label(self.frame2, text="Energy bands selection: ").place(relx=0.5, rely=0.05, anchor=N)

        self.nb_bands = StringVar(self.frame2)
        self.nb_bands.set('-')
        self.choice_bands = OptionMenu(self.frame2, self.nb_bands, *self.list_bands)
        self.choice_bands.place(relx=0.5, rely=0.20, anchor=N)
        self.nb_bands.trace("w", self.build_other_frames)

    def build_other_frames(self, *args):
        """Build frames 3, 4, and 5 once the number of bands is chosen."""
        # --- Frame 3: Energy Interval Selection ---
        self.frame3 = Frame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame3.place(relx=0.175, rely=0.2, relheight=0.3, relwidth=0.80)
        Label(self.frame3, text="Min energy").place(relx=0.375, rely=0.18, anchor="center")
        Label(self.frame3, text="Max energy").place(relx=0.625, rely=0.18, anchor="center")

        # --- Frame 4: Time Interval Selection ---
        self.frame4 = Frame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame4.place(relx=0.025, rely=0.5, relheight=0.3, relwidth=0.95)
        Label(self.frame4, text="Time interval selection for background noise: ").place(relx=0.01, rely=0.05)

        self.btn_sep_times = Checkbutton(
            self.frame4, text="Same time interval for all bands",
            variable=self.var_sep_times, command=self.disable_times, state=self.state_bkg)
        self.btn_sep_times.place(relx=0.95, rely=0.01, anchor=NE)

        Label(self.frame4, text="Start time").place(relx=0.2,  rely=0.16, anchor=N)
        Label(self.frame4, text="End time").place(relx=0.4,    rely=0.16, anchor=N)
        Label(self.frame4, text="Method for noise calculation: ").place(relx=0.85, rely=0.16, anchor=N)

        # --- Frame 5: Plot Units ---
        self.frame5 = LabelFrame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame5.place(relx=0.025, rely=0.8, relheight=0.1, relwidth=0.95)
        Label(self.frame5, text="Plot units: ").place(relx=0.01, rely=0.5, anchor=W)

        self.unit_var = StringVar(self.frame5)
        self.unit_var.set(self.unit_choices[0])
        self.type = self.unit_var.get()
        self.menu_units = OptionMenu(self.frame5, self.unit_var, *self.unit_choices)
        self.menu_units.place(relx=0.1, rely=0.5, anchor=W)

        self.btn_time_profile_plotting = Button(
            self.frame5, text="Plot Time Profile", command=self.time_profile_plotting)
        self.btn_time_profile_plotting.place(relx=0.3, rely=0.5, anchor="center")

        self.grid_var = IntVar(value=0)
        Checkbutton(self.frame5, text="Show grid",
                    variable=self.grid_var).place(relx=0.45, rely=0.5, anchor="center")

        self.entries_list()

    def destroy(self):
        """Close the 'Select Background' window."""
        self.bkg_window.destroy()

    # =================== Enable / disable helpers ===================

    def disable_bkg(self):
        """Enable or disable frame-4 controls depending on background checkbox state."""
        n_checked = self.var_bkg.get() + self.var_data_bkg.get()
        if n_checked < 2 and self.backup_bkg < 2:
            if self.state_bkg == NORMAL:
                if self.frame4:
                    nb = int(self.nb_bands.get())
                    for i in range(nb):
                        self.time_min[i].delete(0, 'end')
                        self.time_max[i].delete(0, 'end')
                    self.btn_sep_times.select()
                self.state_bkg = self.state_time = DISABLED
            else:
                self.state_bkg = NORMAL
                if self.frame4:
                    for i in range(int(self.nb_bands.get())):
                        self.time_min[i].insert(0, self.start_date)
                        self.time_max[i].insert(0, self.end_date)

        if self.frame4:
            nb = int(self.nb_bands.get())
            self.btn_sep_times.config(state=self.state_bkg)
            for widget in (self.time_min[0], self.time_max[0],
                           self.btn_graphical[0], self.btn_spectrogram[0],
                           self.method_selection[0]):
                widget.config(state=self.state_bkg)
            for i in range(1, nb):
                for widget in (self.time_min[i], self.time_max[i],
                               self.btn_graphical[i], self.btn_spectrogram[i],
                               self.method_selection[i]):
                    widget.config(state=self.state_time)

            if not self.time_min[0].get():
                self.time_min[0].insert(0, self.start_date)
            if not self.time_max[0].get():
                self.time_max[0].insert(0, self.end_date)

        self.backup_bkg = n_checked

    def disable_times(self):
        """Toggle per-band independent time intervals."""
        if self.state_time == NORMAL:
            self.state_time = DISABLED
            for i in range(1, int(self.nb_bands.get())):
                self.time_min[i].delete(0, 'end')
                self.time_max[i].delete(0, 'end')
        else:
            self.state_time = NORMAL
            for i in range(1, int(self.nb_bands.get())):
                self.time_min[i].insert(0, self.start_date)
                self.time_max[i].insert(0, self.end_date)

        for i in range(1, int(self.nb_bands.get())):
            for widget in (self.time_min[i], self.time_max[i],
                           self.btn_graphical[i], self.btn_spectrogram[i],
                           self.method_selection[i]):
                widget.config(state=self.state_time)
            if not self.time_min[i].get():
                self.time_min[i].insert(0, self.start_date)
            if not self.time_max[i].get():
                self.time_max[i].insert(0, self.end_date)

    # =================== Reading file ===================

    def open_file(self, file=None):
        """Open and parse a FITS file.

        Parameters
        ----------
        file : str, optional
            Pre-selected path; skips the file dialog when provided.
        """
        self.name = file or askopenfilename(
            initialdir=".",
            filetypes=(("FITS files", "*.fits"), ("All Files", "*.*")),
            title="Please Select Spectrum or Image File",
        )

        if not self.name:
            self.text_filename.insert(0, "No file chosen")
            return

        data    = loader.get_data(self.name)
        headers = loader.get_header(self.name)

        self.start_date = headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown'))
        self.end_date   = headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))
        self.text_filename.insert(0, self.name)

        self.lower_bands = np.array(data['e_low'])
        self.upper_bands = np.array(data['e_high'])
        self.times       = data['time']
        self.counts      = data['counts']
        self.counts_err  = data['counts_err']
        self.del_times   = self._delay_times(data['timedel'])
        self.energies    = data

        start_dt = datetime.datetime.strptime(self.start_date, _DT_FMT_MS)
        end_dt   = datetime.datetime.strptime(self.end_date,   _DT_FMT_MS)
        delta_days = (end_dt.replace(hour=0, minute=0, second=0) -
                      start_dt.replace(hour=0, minute=0, second=0)).days

        sh, sm, ss = self._parse_hms(self.start_date)
        eh, em, es = self._parse_hms(self.end_date)
        self.start_time = sh * 3600 + sm * 60 + ss
        self.end_time   = eh * 3600 + em * 60 + es + delta_days * 86_400

        self.build_second_frame()

    # ------------------------------------------------------------------
    # Static / pure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_datetime_string(date_str):
        """Parse an ISO datetime string to a datetime object."""
        for fmt in (_DT_FMT_MS, _DT_FMT):
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {date_str!r}")

    @staticmethod
    def _delay_times(data):
        """Shift the timedel array by one position (first value repeated)."""
        del_data = np.empty_like(data)
        del_data[0]  = data[0]
        del_data[1:] = data[:-1]
        return del_data

    def _parse_hms(self, date_str):
        """Return (hour, minute, second) integers from an ISO datetime string."""
        time_part = date_str.split('T')[1]
        h, m, s_frac = time_part.split(':')
        s = int(s_frac.split('.')[0])
        return int(h), int(m), s

    def find_time(self, date):
        """Return (hour, minute, second) from an ISO datetime string."""
        return self._parse_hms(date)

    def round_value(self, liste, value):
        """Return the index of the element in *liste* closest to *value*.

        NaN and infinite entries in *liste* are ignored so they can never be
        selected as the nearest value.
        """
        arr = np.asarray(liste, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            raise ValueError(f"round_value: no finite values in array (value={value!r})")

        diffs = np.where(finite_mask, np.abs(arr - float(value)), np.inf)
        idx = int(np.argmin(diffs))
        self.rounded = int(arr[idx])
        return idx

    # =================== Getting user values ===================

    def entries_list(self):
        """Create per-band widgets from the current band count."""
        if not self.nb_bands.get().isdigit():
            return

        nb = int(self.nb_bands.get())
        self.energy_min  = [''] * nb
        self.energy_max  = [''] * nb
        self.energy_min_list = [''] * nb
        self.energy_max_list = [''] * nb
        self.time_min    = [''] * nb
        self.time_max    = [''] * nb
        self.time_min_list   = [''] * nb
        self.time_max_list   = [''] * nb
        self.bkg_start_time  = [''] * nb
        self.bkg_end_time    = [''] * nb
        self.bkg_start_index = [''] * nb
        self.bkg_end_index   = [''] * nb
        self.method_selection = [''] * nb
        self.method_list = [''] * nb
        self.btn_graphical   = [''] * nb
        self.btn_spectrogram = [''] * nb
        self.energy_min_var  = []
        self.energy_max_var  = []

        for i in range(nb):
            self.method_var.append(StringVar(self.frame3))
            self.method_var[i].set(self.method_choices[1])
            self._create_band_widgets(i)
            self.energy_min_list[i] = self.energy_min_var[i]
            self.energy_max_list[i] = self.energy_max_var[i]
            self.time_min_list[i]   = self.time_min[i]
            self.time_max_list[i]   = self.time_max[i]
            self.method_list[i]     = self.method_var[i]

    def _create_band_widgets(self, i):
        """Create energy / time entry widgets for band *i*."""
        e_values_l = sorted(set(self.energies['e_low']))
        e_values_h = sorted(set(self.energies['e_high']))
        e_values_h = [e for e in e_values_h if np.isfinite(e)]

        e_low_values  = [int(e) for e in e_values_l if e != 0]
        e_high_values = [int(e) for e in e_values_h]

        var_min = IntVar(value=min(e_low_values))
        var_max = IntVar(value=max(e_high_values))
        self.energy_min_var.append(var_min)
        self.energy_max_var.append(var_max)

        self.energy_min[i] = OptionMenu(self.frame3, var_min, *e_low_values)
        self.energy_max[i] = OptionMenu(self.frame3, var_max, *e_high_values)
        self.energy_min[i].place(relx=0.375, rely=0.4 + 0.14 * i, anchor="center", relwidth=0.3)
        self.energy_max[i].place(relx=0.625, rely=0.4 + 0.14 * i, anchor="center", relwidth=0.3)

        self.time_min[i] = Entry(self.frame4, width=20, state=self.state_time)
        self.time_max[i] = Entry(self.frame4, width=20, state=self.state_time)
        self.time_min[i].place(relx=0.2, rely=0.3 + 0.14 * i, anchor=N, width=150)
        self.time_max[i].place(relx=0.4, rely=0.3 + 0.14 * i, anchor=N, width=150)
        self.time_min[i].insert(0, self.start_date)
        self.time_max[i].insert(0, self.end_date)

        self.bkg_start_index[i] = 0
        self.bkg_end_index[i]   = len(self.times)

        self.btn_graphical[i] = Button(
            self.frame4, text="Time Profile", state=self.state_time,
            command=lambda idx=i: self.graphical_interval(idx))
        self.btn_graphical[i].place(relx=0.59, rely=0.35 + 0.14 * i, anchor=W)

        self.btn_spectrogram[i] = Button(
            self.frame4, text="Spectrogram", state=self.state_time,
            command=lambda idx=i: self.spectrogram_interval(idx))
        self.btn_spectrogram[i].place(relx=0.68, rely=0.35 + 0.14 * i, anchor=W)

        self.method_selection[i] = OptionMenu(self.frame4, self.method_var[i], *self.method_choices)
        self.method_selection[i].place(relx=0.85, rely=0.35 + 0.14 * i, anchor="center")
        self.method_selection[i].config(state=self.state_time)

        # First band is controlled by state_bkg, not state_time
        if i == 0:
            for w in (self.time_min[0], self.time_max[0],
                      self.btn_graphical[0], self.btn_spectrogram[0],
                      self.method_selection[0]):
                w.config(state=self.state_bkg)

    # =================== Time / index conversions ===================

    def date_to_times_index(self, date_value):
        """Map an ISO date string to the nearest index in self.times."""
        t_min = np.min(self.times)
        t_max = np.max(self.times)
        start_dt = datetime.datetime.strptime(self.start_date, _DT_FMT_MS)
        end_dt   = datetime.datetime.strptime(self.end_date,   _DT_FMT_MS)
        date_dt  = datetime.datetime.strptime(date_value,      _DT_FMT_MS)

        total_s = (end_dt - start_dt).total_seconds()
        delta_s = (date_dt - start_dt).total_seconds()
        t_estimated = t_min + (delta_s / total_s) * (t_max - t_min)
        index = int(np.argmin(np.abs(self.times - t_estimated)))
        return index

    def convert_time_to_date(self, t):
        """Convert a value in the self.times axis to an ISO datetime string."""
        try:
            start_dt = self.parse_datetime_string(self.start_date)
            end_dt   = self.parse_datetime_string(self.end_date)
        except Exception as e:
            print(f"[convert_time_to_date] Error parsing dates: {e}")
            return "Invalid Date"

        t_min = min(self.times)
        t_max = max(self.times)
        t = np.clip(t, t_min, t_max)

        fraction = (t - t_min) / (t_max - t_min)
        new_dt = start_dt + datetime.timedelta(
            seconds=fraction * (end_dt - start_dt).total_seconds())
        return new_dt.strftime(_DT_FMT_MS)[:-3]

    def _apply_selection(self, i, dt_start, dt_end):
        """Shared logic to store a background time selection for band *i*.

        Called by both graphical_interval and spectrogram_interval after the
        user has picked a region in either plot.
        """
        start_dt = self.parse_datetime_string(self.start_date)
        end_dt   = self.parse_datetime_string(self.end_date)
        t_min = min(self.times)
        t_max = max(self.times)
        total_s = (end_dt - start_dt).total_seconds()

        frac_start = (dt_start - start_dt).total_seconds() / total_s
        frac_end   = (dt_end   - start_dt).total_seconds() / total_s

        rel_start = np.clip(t_min + frac_start * (t_max - t_min), t_min, t_max)
        rel_end   = np.clip(t_min + frac_end   * (t_max - t_min), t_min, t_max)
        if rel_start > rel_end:
            rel_start, rel_end = rel_end, rel_start

        self.bkg_start_time[i] = rel_start
        self.bkg_end_time[i]   = rel_end

        date_start_str = self.convert_time_to_date(rel_start)
        date_end_str   = self.convert_time_to_date(rel_end)

        self.time_min[i].delete(0, 'end')
        self.time_max[i].delete(0, 'end')
        self.time_min[i].insert(0, date_start_str)
        self.time_max[i].insert(0, date_end_str)

        BackgroundWindow.DATA_BKG_SELECTED = True
        BackgroundWindow.DATA_BKG_START    = self.date_to_times_index(date_start_str)
        BackgroundWindow.DATA_BKG_END      = self.date_to_times_index(date_end_str)

        self.bkg_start_index[i] = self.round_value(self.times, rel_start)
        self.bkg_end_index[i]   = self.round_value(self.times, rel_end)

    # =================== Band energy helpers ===================

    def add_bands(self):
        """Populate self.energies_low / self.energies_high from UI selections."""
        self.energies_low  = []
        self.energies_high = []
        for i in range(int(self.nb_bands.get())):
            if self.energy_min_var[i].get() != '':
                self.round_value(self.lower_bands, int(self.energy_min_list[i].get()))
                self.energies_low.append(int(self.energy_min_list[i].get()))
            if self.energy_max_var[i].get() != '':
                self.round_value(self.upper_bands, int(self.energy_max_list[i].get()))
                self.energies_high.append(int(self.energy_max_list[i].get()))

    # =================== Data computation ===================

    def get_data(self):
        """Build self.data and self.data_err arrays in the chosen unit."""
        nb = len(self.energies_low)
        nt = len(self.times)
        self.data     = np.zeros((nt, nb + 1))
        self.data_err = np.zeros((nt, nb + 1))

        # Last column = times
        self.data[:, nb]     = self.times
        self.data_err[:, nb] = self.times

        for j, (e_low, e_high) in enumerate(zip(self.energies_low, self.energies_high)):
            a = self._band_index(self.lower_bands, e_low)
            b = self._band_index(self.upper_bands, e_high)

            counts_slice     = self.counts[:,     a:b + 1].sum(axis=1)
            counts_err_slice = self.counts_err[:, a:b + 1].sum(axis=1)

            if self.type == 'Rate':
                self.data[:, j]     = counts_slice     / self.del_times
                self.data_err[:, j] = counts_err_slice / self.del_times
            elif self.type == 'Counts':
                self.data[:, j]     = counts_slice
                self.data_err[:, j] = counts_err_slice
            elif self.type == 'Flux':
                e_diff = self.area * abs(e_high - e_low)
                self.data[:, j]     = counts_slice     / self.del_times / e_diff
                self.data_err[:, j] = counts_err_slice / self.del_times / e_diff
            else:
                print("No matching unit plotting type found")

    def _band_index(self, band_array, value):
        """Return the index in *band_array* closest to *value*.

        Only finite entries are considered for both exact and nearest matches.
        """
        arr = np.asarray(band_array, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        matches = np.where(finite_mask & (arr == float(value)))[0]
        if matches.size:
            return int(matches[0])
        return self.round_value(band_array, value)

    def get_bkg(self):
        """Compute background for each band using the chosen method."""
        self.bkg = np.zeros_like(self.data)
        nb = len(self.energies_low)
        self.bkg[:, nb] = self.times

        for band in range(nb):
            if self.var_sep_times.get() == 1 and band != 0:
                self.method_var[band].set(self.method_var[0].get())
                self.method_list[band]     = self.method_var[band]
                self.bkg_start_index[band] = self.bkg_start_index[0]
                self.bkg_end_index[band]   = self.bkg_end_index[0]

            s, e    = self.bkg_start_index[band], self.bkg_end_index[band]
            method  = self.method_list[band].get()
            y_slice = self.data[s:e, band]
            t_slice = self.times[s:e]

            if method == "Median":
                self.bkg[:, band] = np.median(y_slice)
            elif method == "Mean":
                self.bkg[:, band] = np.mean(y_slice)
            elif method in ("1Poly", "2Poly", "3Poly"):
                deg  = int(method[0])
                poly = np.poly1d(np.polyfit(t_slice, y_slice, deg))
                self.bkg[:, band] = poly(np.arange(len(self.times)))
            elif method == "Exp":
                coeffs = np.polyfit(t_slice, np.log(y_slice), 1)
                a_coef = np.exp(coeffs[1])
                b_coef = coeffs[0]
                self.bkg[:, band] = a_coef * np.exp(b_coef * self.times)
            else:
                self.bkg[:, band] = 0
                print("Background detection method not found")

            # Clamp negatives
            self.bkg[:, band] = np.maximum(self.bkg[:, band], 0)

    def get_data_bkg(self):
        """Compute data minus background; clamp to zero; store as class attribute."""
        self.data_bkg = np.maximum(self.data - self.bkg, 0)
        nb = len(self.energies_low)
        self.data_bkg[:, nb] = self.times
        BackgroundWindow.DATA_BKG_RESULT = self.data_bkg

    # =================== Plotting helpers ===================

    def plot_options(self):
        """Compute axis ticks, labels, and datetime axis for the current data."""
        from datetime import timedelta

        start_dt = self.parse_datetime_string(self.start_date)
        end_dt   = self.parse_datetime_string(self.end_date)
        total_s  = (end_dt - start_dt).total_seconds()

        interval_sec = _TICK_DEFAULT
        for threshold, interval in _TICK_INTERVALS:
            if total_s <= threshold:
                interval_sec = interval
                break

        tick_dts = []
        current = start_dt
        while current <= end_dt:
            tick_dts.append(current)
            current += datetime.timedelta(seconds=interval_sec)

        t_min = min(self.times)
        t_max = max(self.times)

        self.time_scale = [
            t_min + (dt - start_dt).total_seconds() * (t_max - t_min) / total_s
            for dt in tick_dts
        ]
        self.times_datetime = [
            start_dt + datetime.timedelta(
                seconds=(t - t_min) / (t_max - t_min) * total_s)
            for t in self.times
        ]

        fmt = '%H:%M' if total_s <= 86_400 else '%m-%d\n%H:%M'
        self.plot_label = [dt.strftime(fmt) for dt in tick_dts]
        self.xlabel = f'Start time: {self.start_date} -- End time: {self.end_date}'

        self.type = self.unit_var.get()
        _labels = {
            'Rate':   ('Rate (Counts/s) by Bands',  'Time Profile Plotting Rate (Counts/s)'),
            'Counts': ('Counts by Bands',            'Time Profile Plotting Counts'),
            'Flux':   ('Flux by Bands',              'Time Profile Plotting Flux'),
        }
        self.ylabel, self.title = _labels.get(self.type, ('', ''))
        if not self.ylabel:
            print("No matching unit plotting type found")

        self.data_label = [
            [f'{dtype} for {lo}-{hi} keV'
             for lo, hi in zip(self.energies_low, self.energies_high)] + ['Times']
            for dtype in ('Original data', 'Background', 'Corrected data')
        ]

    def graphical_interval(self, i):
        """Open an interactive time-profile plot for graphical background selection."""
        self.type = self.unit_var.get()
        self.add_bands()
        self.get_data()
        self.plot_options()

        dt_start, dt_end = IntervalSelector(
            self.times_datetime,
            self.data,
            col_label=self.data_label,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            color=self.color,
            samefig=self.var_sep_times.get(),
            band=i,
        ).graphical_selection()

        if dt_start is None or dt_end is None:
            print("[graphical_interval] No selection made, aborting.")
            return

        self._apply_selection(i, dt_start, dt_end)

    def spectrogram_interval(self, i):
        """Open a spectrogram with a span selector for background definition."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import SpanSelector
        from matplotlib import dates as mdates

        data    = loader.get_data(self.name)
        headers = loader.get_header(self.name)

        from .graphics import plotting
        plot_instance = plotting.Plotting(data=data, headers=headers)

        def _no_popup_specgm_lim():
            lower_clean = plot_instance.lower_bands[np.isfinite(plot_instance.lower_bands)]
            upper_clean = plot_instance.upper_bands[np.isfinite(plot_instance.upper_bands)]
            if lower_clean.size == 0 or upper_clean.size == 0:
                plot_instance.energy_min = 0.0
                plot_instance.energy_max = 20.0
            else:
                plot_instance.energy_min = float(np.min(lower_clean))
                plot_instance.energy_max = float(np.max(upper_clean))

        plot_instance.specgm_lim = _no_popup_specgm_lim
        plot_instance._Plotting__plot_spectrogram('rate')

        def onselect(xmin, xmax):
            dt_start = mdates.num2date(xmin).replace(tzinfo=None)
            dt_end   = mdates.num2date(xmax).replace(tzinfo=None)
            self._apply_selection(i, dt_start, dt_end)
            plt.close()

        fig = plt.gcf()
        ax  = plt.gca()
        _span_selector = SpanSelector(
            ax, onselect, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor='red'),
        )
        fig.canvas.manager.window.lift()
        fig.canvas.manager.window.focus_force()
        plt.show()

    def time_profile_plotting(self):
        """Plot the time profile with dynamic x-axis date formatting."""
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.dates import AutoDateLocator, DateFormatter

        plt.close()
        self.add_bands()
        self.type = self.unit_var.get()
        self.get_data()
        self.plot_options()

        yerr = self.data_err if self.var_error.get() else np.zeros_like(self.data_err)

        plt.figure()

        if self.var_og_data.get():
            df = pd.DataFrame(self.data, index=self.times_datetime, columns=self.data_label[0])
            for band in range(len(self.energies_low)):
                df.plot(y=self.data_label[0][band], yerr=yerr[:, band],
                        color=self.color[band], ax=plt.gca(), linewidth=0.25)

        if self.var_bkg.get() or self.var_data_bkg.get():
            self.get_bkg()

            if self.var_bkg.get():
                df = pd.DataFrame(self.bkg, index=self.times_datetime, columns=self.data_label[1])
                for band in range(len(self.energies_low)):
                    df.plot(y=self.data_label[1][band], color=self.color[band],
                            ax=plt.gca(), linewidth=0.1)

            if self.var_data_bkg.get():
                self.get_data_bkg()
                df = pd.DataFrame(self.data_bkg, index=self.times_datetime, columns=self.data_label[2])
                for band in range(len(self.energies_low)):
                    df.plot(y=self.data_label[2][band], yerr=yerr[:, band],
                            color=self.color[band], ax=plt.gca(), linewidth=1)

        ax = plt.gca()
        ax.set_xlim(self.times_datetime[0], self.times_datetime[-1])

        if self.grid_var.get():
            plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.8)
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        else:
            plt.grid(False)
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', linewidth=0.1, alpha=0.1)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        locator   = AutoDateLocator()
        formatter = DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        def _update_xticks(event):
            event.canvas.figure.axes[0].xaxis.set_major_locator(locator)
            event.canvas.figure.axes[0].xaxis.set_major_formatter(formatter)
            event.canvas.draw_idle()

        plt.gcf().canvas.mpl_connect('draw_event', _update_xticks)
        plt.gcf().autofmt_xdate()
        plt.show()