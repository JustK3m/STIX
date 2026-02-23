import numpy as np
import pandas as pd
import datetime
import re
from tkinter import *
from astropy.io import fits
from matplotlib import pyplot as plt
from entry_int import EntryInt
from tkinter.filedialog import askopenfilename
from interval_selector import IntervalSelector
import second
import os
import sys


class BackgroundWindow:
    """Class to create a background Window"""
    # FIXME: When tests are finished, remember to set fname as None, this is just to gain time for each test
    # fname = 'solo_L1A_stix-sci-spectrogram' \
    #         '-2102140001_20210214T014006-20210214T015515_008648_V01.fits'
    
    fname_r = 'data/solo_L1_stix-sci-xray-spec_20230319T175504-20230320T000014_V02_2303197888-65462.fits'

    def resource_path(relative_path):
        """Renvoie le chemin absolu même si l'app est congelée avec PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
    
    fname = resource_path(fname_r)  # Default file to open

    DATA_BKG_SELECTED = False  # If background data has been selected by user
    DATA_BKG_RESULT = None     # Result of the background calculation, to be used in other windows
    DATA_BKG_START = None  # Starting time of the background data, to be used in other windows
    DATA_BKG_END = None    # Ending time of the background data, to be used in other windows

    def __init__(self, root=None, show=True):
        """The main interest of this class is to calculate background caused by instruments and plot the data after
        removal of the background. As for plotting.py, this class opens a new window (if show=True) to let the user
        decide plotting options such as unit type, data type, energy bands, time intervals, and calculation method for
        background. Parameters: \n
            show (bool): if True, a new window called 'Select Background' will be opened, containing options to plot
                         data, background, and error."""

        self.root = root            # Root of the file
        self.hdul = None            # Opened file
        self.hdulist = None         # Data reading
        self.name = None            # Name of the .fits file imported

        self.counts = None          # Matrix contaning the counts per band in function of time time
        self.counts_err = None      # Matrix contaning the error of the counts per band in function of time
        self.times = None           # Index of times for x axis
        self.del_times = None       # List containing the difference between two successive times
        self.data = None            # Converts self.counts in chosen unit (rate, counts or flux) and adds time index
        self.data_err = None        # Converts self.counts_err in chosen unit (rate, counts or flux) and adds time index
        self.del_data = None        # Delayed data to match times
        self.bkg = None             # Calculated background noise
        self.data_bkg = None        # Data without background noise
        self.area = 6               # Area of the surface of detection of the telescope in cm²; used for the flux

        self.nb_bands = StringVar()     # Number of energy bands to plot
        self.lower_bands = list()       # Lower bounds for energy bands from .fits file
        self.upper_bands = list()       # Upper bounds for energy bands from .fits file
        self.energies_low = []          # Lower bounds for energy bands chosen by user
        self.energies_high = []         # Upper bounds for energy bands chosen by user
        self.rounded = int()            # Nearest rounded value to fit in energy list; used in function self.round

        self.start_date = []            # Starting date at format YYYY-MM-DD-HH-MM-SS
        self.end_date = []              # Ending date at format YYYY-MM-DD-HH-MM-SS
        self.date_day = None            # Date at format YYYY-MM-DD
        self.date_time = None           # Time at format HH-MM-SS
        self.start_time = int()         # Starting time in seconds
        self.end_time = int()           # Ending time in seconds

        self.choice_bands = None                        # OptionMenu to choose number of bands
        self.list_bands = ('1', '2', '3', '4', '5')     # Choices for number of bands dropdown list

        self.type = str()                               # Data type (rate, counts, flux)
        self.unit_var = StringVar()                     # StringVar for data type
        self.unit_choices = ('Rate', 'Counts', 'Flux')  # Choices for data type

        self.method_list = list()                       # Background methods calculation
        self.method_var = list()                        # StringVars for background methods calculation
        self.method_choices = ('Median', 'Mean', '1Poly', '2Poly', '3Poly', 'Exp')  # Choices for background methods

        if show:
            self.bkg_window = None          # Background main window
            self.frame1 = None              # Frame 1: Data & Unit Plotting
            self.frame2 = None              # Frame 2: Number of Bands Selection
            self.frame3 = None              # Frame 3: Energy Interval Selection
            self.frame4 = None              # Frame 4: Time Interval Selection
            self.frame5 = None              # Frame 5: Plot Units

            self.state_bkg = DISABLED       # State for buttons in frame 3
            self.state_time = DISABLED      # State for buttons in frame 4
            self.backup_bkg = 0             # Keeps trace of number of boxes with bkg have been ticked in frame 1

            self.label1 = None              # Title label for frame 1
            self.label_filename = None      # Text for choosing file
            self.label2 = None              # Title label for frame 2
            self.text_min_energy = None     # Text for min energy
            self.text_max_energy = None     # Text for max energy
            self.label4 = None              # Title label for frame 3
            self.text_start_time = None     # Text for starting time
            self.text_end_time = None       # Text for ending time
            self.text_noise = None          # Text for noise calculation method
            self.label5 = None              # Title label for frame 5

            self.menu_units = None                  # Dropdown list to choose data unit type
            self.btn_time_profile_plotting = None   # Plot time profile button
            self.close = None                       # Close button

            self.text_filename = None               # Entrybox displaying the path of the chosen file
            self.btn_browse = None                  # Browse button to import a file
            self.btn_og_data = None                 # Checkbutton to plot original data
            self.btn_bkg = None                     # Checkbutton to plot background noise
            self.btn_data_bkg = None                # Checkbutton to plot data after removing background noise
            self.btn_error = None                   # Checkbutton to plot error on data
            self.btn_sep_times = None               # Checkbutton to decide different times for each band for background

            self.var_og_data = IntVar(value=1)      # Variable for original data checkbox; ticked by default
            self.var_bkg = IntVar()                 # Variable for background checkbox
            self.var_data_bkg = IntVar()            # Variable for data - background checkbox
            self.var_error = IntVar()               # Variable for error checkbox
            self.var_sep_times = IntVar(value=1)    # Variable for separate times checkbox

            self.energy_min = list()                # Entry boxes for lower bounds of energy bands
            self.energy_max = list()                # Entry boxes for upper bounds of energy bands
            self.time_min = list()                  # Entry boxes for lower bounds of time intervals
            self.time_max = list()                  # Entry boxes for upper bounds of time intervals
            self.btn_graphical = list()             # Buttons to select time intervals on the time profile plot
            self.btn_spectrogram = list()            # Buttons to select energy bands on the spectrogram plot
            self.method_selection = list()  
            self.energy_min_var = []
            self.energy_max_var = []        # Dropdown lists to select method for background calculation

            self.energy_min_list = list()           # Storage for lower bounds of energy bands
            self.energy_max_list = list()           # Storage for upper bounds of energy bands
            self.time_min_list = list()             # Storage for lower bounds of time intervals
            self.time_max_list = list()             # Storage for upper bounds of time intervals

            self.bkg_start_time = list()            # Storage for starting times selected on "Graphical"
            self.bkg_end_time = list()              # Storage for ending times selected on "Graphical"
            self.bkg_start_index = list()           # Storage for starting indexes selected on "Graphical"
            self.bkg_end_index = list()             # Storage for ending indexes selected on "Graphical"

            self.title = str()                      # Label for the title of the plot
            self.xlabel = str()                     # Label for x axis
            self.ylabel = str()                     # Label for y axis
            self.time_scale = []                    # Time range used for the time plot
            self.plot_label = []                    # Labels of energy bands
            self.data_label = list()                # List of labels for each band for data plotting
            self.color = ['blue', 'red', 'green', 'black', 'orange']  # Colors used to plot energy bands

            self.build_bkg_window()  # End of __init__, starting to build window

        # Half-smoothing
        # self.HalfSmooth = Label(self.frame2, text="Profile Half Smoothing width(#pts):")
        # self.HalfSmooth.place(relx=0.41, rely=0.05)
        # self.HalfSmoothList = ("0", "1", "4", "8", "16", "32", "64", "128", "256")
        # self.SpinboxHalfSmooth = Spinbox(self.frame2, values=self.HalfSmoothList)
        # self.SpinboxHalfSmooth.place(relx=0.63, rely=0.05, width=50)

    # =================== Building window ===================

    def build_bkg_window(self):
        """Builds the main components of the background window and the first frame; waits for the user to choose a file
        to build the second frame."""

        # =================== Window definition ===================

        self.bkg_window = Toplevel()
        self.bkg_window.title('STIX Background Options')
        self.bkg_window.geometry("1000x600")
        Label(self.bkg_window,
              text="Select Background",
              fg="black",
              font="Helvetica 12 bold italic").pack()

        self.close = Button(self.bkg_window, text="Close", command=self.destroy)
        self.close.place(relx=0.5, rely=0.95, anchor='center')

        # =================== 1st frame : Data & Unit Plotting ===================

        self.frame1 = LabelFrame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame1.place(relx=0.025, rely=0.05, relheight=0.15, relwidth=0.95)

        self.label_filename = Label(self.frame1, text="Spectrum or Image File: ")
        self.label_filename.place(relx=0.02, rely=0.25, anchor=W)
        self.text_filename = Entry(self.frame1, width=20)
        self.text_filename.place(relx=0.19, rely=0.25, relheight=0.3, relwidth=0.57, anchor=W)
        if BackgroundWindow.fname:
            self.text_filename.insert(0, BackgroundWindow.fname)
            self.open_file(BackgroundWindow.fname)
        else:
            self.text_filename.insert(0, "No file chosen")

        self.btn_browse = Button(self.frame1, text='Browse ->', command=self.open_file)
        self.btn_browse.place(relx=0.78, rely=0.25, anchor=W)

        self.label1 = Label(self.frame1, text="Plot time profile for: ")
        self.label1.place(relx=0.02, rely=0.65, anchor=W)

        self.btn_og_data = Checkbutton(self.frame1, text="Data", variable=self.var_og_data)
        self.btn_og_data.place(relx=0.19, rely=0.65, anchor=W)

        self.btn_bkg = Checkbutton(self.frame1, text="Background", variable=self.var_bkg, command=self.disable_bkg)
        self.btn_bkg.place(relx=0.26, rely=0.65, anchor=W)

        self.btn_data_bkg = Checkbutton(self.frame1, text="Data-Background", variable=self.var_data_bkg,
                                        command=self.disable_bkg)
        self.btn_data_bkg.place(relx=0.37, rely=0.65, anchor=W)

        self.btn_error = Checkbutton(self.frame1, text="Error", variable=self.var_error)
        self.btn_error.place(relx=0.51, rely=0.65, anchor=W)

    def build_second_frame(self):
        """After selecting the file, builds the second frame, containing a list with the number of energy bands;
        waits for the user to select the number of bands to build the other frames."""

        # =================== 2nd frame : Number of Bands Selection ===================

        self.frame2 = Canvas(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame2.place(relx=0.025, rely=0.2, relheight=0.3, relwidth=0.15)

        self.label2 = Label(self.frame2, text="Energy bands selection: ")
        self.label2.place(relx=0.5, rely=0.05, anchor=N)

        self.nb_bands = StringVar(self.frame2)
        self.nb_bands.set('-')
        self.choice_bands = OptionMenu(self.frame2, self.nb_bands, *self.list_bands)
        self.choice_bands.place(relx=0.5, rely=0.20, anchor=N)
        self.nb_bands.trace("w", self.build_other_frames)

    def build_other_frames(self, *args):
        """Builds third, fourth and fifth frames. Buttons call for their respective functions."""

        # =================== 3rd frame : Energy Interval Selection ===================

        self.frame3 = Frame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame3.place(relx=0.175, rely=0.2, relheight=0.3, relwidth=0.80)

        # Titre des champs
        self.text_min_energy = Label(self.frame3, text="Min energy")
        self.text_min_energy.place(relx=0.375, rely=0.18, anchor="center")
        self.text_max_energy = Label(self.frame3, text="Max energy")
        self.text_max_energy.place(relx=0.625, rely=0.18, anchor="center")


        # =================== 4th frame : Time Interval Selection ===================

        self.frame4 = Frame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame4.place(relx=0.025, rely=0.5, relheight=0.3, relwidth=0.95)

        self.label4 = Label(self.frame4, text="Time interval selection for background noise: ")
        self.label4.place(relx=0.01, rely=0.05)

        self.btn_sep_times = Checkbutton(self.frame4, text="Same time interval for all bands",
                                         variable=self.var_sep_times, command=self.disable_times,
                                         state=self.state_bkg)
        self.btn_sep_times.place(relx=0.95, rely=0.01, anchor=NE)

        self.text_start_time = Label(self.frame4, text="Start time")
        self.text_start_time.place(relx=0.2, rely=0.16, anchor=N)
        self.text_end_time = Label(self.frame4, text="End time")
        self.text_end_time.place(relx=0.4, rely=0.16, anchor=N)

        self.text_noise = Label(self.frame4, text="Method for noise calculation: ")
        self.text_noise.place(relx=0.85, rely=0.16, anchor=N)

        # =================== 5th frame : Plot Units ===================

        self.frame5 = LabelFrame(self.bkg_window, relief=RAISED, borderwidth=2)
        self.frame5.place(relx=0.025, rely=0.8, relheight=0.1, relwidth=0.95)

        self.label5 = Label(self.frame5, text="Plot units: ")
        self.label5.place(relx=0.01, rely=0.5, anchor=W)

        self.unit_var = StringVar(self.frame5)
        self.unit_var.set(self.unit_choices[0])
        self.type = self.unit_var.get()
        self.menu_units = OptionMenu(self.frame5, self.unit_var, *self.unit_choices)
        self.menu_units.place(relx=0.1, rely=0.5, anchor=W)

        self.btn_time_profile_plotting = Button(self.frame5, text="Plot Time Profile",
                                                command=self.time_profile_plotting)
        self.btn_time_profile_plotting.place(relx=0.3, rely=0.5, anchor="center")

        self.grid_var = IntVar(value=0) 
        self.grid_check = Checkbutton(
            self.frame5,
            text="Show grid",
            variable=self.grid_var
        )
        self.grid_check.place(relx=0.45, rely=0.5, anchor="center")

        # ============== Entries for frames 3 & 4 ==============

        self.entries_list()

    def destroy(self):
        """Closing 'Select Input' window when clicking 'Close' button."""
        self.bkg_window.destroy()

    def disable_bkg(self):
        """If none of the checkboxes "Background" and "Data-Background" are ticked, disables all buttons in frame 4;
        else, enables them."""

        # If the user (un)selects background or data-background for the plot, enales or disables all background options
        if self.var_bkg.get() + self.var_data_bkg.get() < 2 and self.backup_bkg < 2:
            if self.state_bkg == NORMAL:
                if self.frame4:
                    for i in range(int(self.nb_bands.get())):
                        self.time_min[i].delete(0, 'end')
                        self.time_max[i].delete(0, 'end')
                    self.btn_sep_times.select()
                self.state_bkg = DISABLED
                self.state_time = DISABLED
            else:
                self.state_bkg = NORMAL
                if self.frame4:
                    for i in range(int(self.nb_bands.get())):
                        self.time_min[i].insert(0, self.start_date)
                        self.time_max[i].insert(0, self.end_date)

        # If background frames have been built:
        if self.frame4:
            self.btn_sep_times.config(state=self.state_bkg)
            self.time_min[0].config(state=self.state_bkg)
            self.time_max[0].config(state=self.state_bkg)
            self.btn_graphical[0].config(state=self.state_bkg)
            self.btn_spectrogram[0].config(state=self.state_bkg)
            self.method_selection[0].config(state=self.state_bkg)
            for i in range(1, int(self.nb_bands.get())):
                self.time_min[i].config(state=self.state_time)
                self.time_max[i].config(state=self.state_time)
                self.btn_graphical[i].config(state=self.state_time)
                self.btn_spectrogram[i].config(state=self.state_time)
                self.method_selection[i].config(state=self.state_time)

            if self.time_min[0].get() == '':
                self.time_min[0].insert(0, self.start_date)
            if self.time_max[0].get() == '':
                self.time_max[0].insert(0, self.end_date)

        # Backup variable is used to compare old state of background checkboxes;
        # So if user ticks both background & data-background, background frames will remain active.
        self.backup_bkg = self.var_bkg.get() + self.var_data_bkg.get()


    def disable_times(self):
        """Allows user to set different time intervals for each band for background calculation. If the checkbutton
        "Same time interval for all bands" is ticked, disables all buttons in frame 4, except for the first band;
        else, enables all other buttons in frame 4."""

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
            self.time_min[i].config(state=self.state_time)
            self.time_max[i].config(state=self.state_time)
            self.btn_graphical[i].config(state=self.state_time)
            self.btn_spectrogram[i].config(state=self.state_time)
            self.method_selection[i].config(state=self.state_time)

            if self.time_min[i].get() == '':
                self.time_min[i].insert(0, self.start_date)
            if self.time_max[i].get() == '':
                self.time_max[i].insert(0, self.end_date)

    # =================== Reading file ===================

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
        BackgroundWindow.fname = self.name

        if self.name:  # If file has been chosen by user
            # self.nb_bands.trace_remove("w", self.build_other_frames)  # Remove trace to avoid infinite loop
            with fits.open(self.name) as hdul:
                self.hdul = hdul

                #loading stix data
                data = second.SecondWindow.extract_stix_data(self.hdul)
                headers = second.SecondWindow.extract_stix_header(self.hdul)
                self.time_summarize = [data['time'][-1], headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown')),
                                        headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))]  # time data

                # Loads headers informations
                # time_summarize = [self.hdul[2].data.time[-1], self.hdul[0].header[19],
                #                   self.hdul[0].header[21]]  # time data
                self.start_date, self.end_date = self.time_summarize[1], self.time_summarize[2]
                self.text_filename.insert(0, self.name)  # Displays the input file name in Entry box
                # self.lower_bands = np.array(self.hdul[3].data.e_low)
                # self.upper_bands = np.array(self.hdul[3].data.e_high)
                self.lower_bands = np.array(data['e_low'])
                self.upper_bands = np.array(data['e_high'])

                

            # Loading data
            hdu = fits.open(self.name)
            data1 = second.SecondWindow.extract_stix_data(hdu)
            
            # data1, data2, header0, header1 = self.load_data(self.name)
            # self.times = data1.time
            # self.counts = data1.counts
            # self.counts_err = data1.counts_err

            self.times = data1['time']
            self.counts = data1['counts']
            self.counts_err = data1['counts_err']

            self.del_times = self.delay_times(data1['timedel'])


            start_tuple = self.find_time(self.start_date)
            end_tuple = self.find_time(self.end_date)

            # print('start_date:', self.start_date)
            # print('end_date:', self.end_date)   

            # print('start_tuple:', start_tuple)
            # print('end_tuple:', end_tuple)  

            # Si self.start_date est une chaîne ISO
            start_dt = datetime.datetime.strptime(self.start_date, "%Y-%m-%dT%H:%M:%S.%f")
            end_dt = datetime.datetime.strptime(self.end_date, "%Y-%m-%dT%H:%M:%S.%f")
            
            start_midnight = start_dt.replace(hour=0, minute=0, second=0)
            end_midnight = end_dt.replace(hour=0, minute=0, second=0)

            delta_days = (end_midnight - start_midnight).days
            # print("Écart en jours entiers :", delta_days)


            self.start_time = start_tuple[0] * 3600 + start_tuple[1] * 60 + start_tuple[2]
            self.end_time = end_tuple[0] * 3600 + end_tuple[1] * 60 + end_tuple[2] + delta_days * 86400

            # print('start_time:', self.start_time)
            # print('end_time:', self.end_time)

            # self.start_datetime = BackgroundWindow.parse_datetime_string(self.start_date)
            # self.end_datetime  = BackgroundWindow.parse_datetime_string(self.end_date)

            # self.start_time = 0
            # self.end_time = int((self.end_datetime - self.start_datetime).total_seconds())


            self.energies = data1

            # Importing plotting.py with self.name now chosen
            self.build_second_frame()
        else:
            self.text_filename.insert(0, "No file chosen")

    @staticmethod
    def load_data(file):
        """Reads the Data and Header contents from input file. Loads the input file choosen in 'Select Input' section.
        Returns respectively a table containing datas, energies, dates and channels.\n
        Parameters: \n
            file: contains the data in a fits file."""
        hdulist = fits.open(file)   # Reads the data
        hdulist.info()              # Displays the content of the read file
        return hdulist[2].data, hdulist[3].data, hdulist[0].header, hdulist[3].header

    def find_time(self, date):
        """Converts the starting time in the data. \n
        Parameters: \n
            date: starting date as a list."""
        date_split = re.split('T', date)
        self.date_day = re.split('-', date_split[0])
        self.date_time = re.split(':', date_split[1])
        int_time_sec = re.split('\\.', self.date_time[2])
        found_time = (int(self.date_time[0]), int(self.date_time[1]), int(int_time_sec[0]))
        return found_time

    @staticmethod
    def parse_datetime_string(date_str):
        """Convertit une chaîne ISO en objet datetime"""
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")



    @staticmethod
    def delay_times(data):
        """Creates a TEMPORARY -1 interval, because of the step between deltimes data. Returns the shifted data. \n
        Parameters: \n
            data: original data."""
        del_data = np.zeros_like(data)
        for i in range(1, len(data)):
            del_data[i] = data[i - 1]
        del_data[0] = data[0]
        return del_data

    # =================== Getting user values ===================

    def entries_list(self):
        """Allows to create all buttons and Entry boxes to choose energy values."""

        if self.nb_bands.get().isdigit():
            self.energy_min = [''] * int(self.nb_bands.get())
            self.energy_max = [''] * int(self.nb_bands.get())
            self.energy_min_list = [''] * int(self.nb_bands.get())
            self.energy_max_list = [''] * int(self.nb_bands.get())

            self.time_min = [''] * int(self.nb_bands.get())
            self.time_max = [''] * int(self.nb_bands.get())
            self.time_min_list = [''] * int(self.nb_bands.get())
            self.time_max_list = [''] * int(self.nb_bands.get())

            self.bkg_start_time = [''] * int(self.nb_bands.get())
            self.bkg_end_time = [''] * int(self.nb_bands.get())
            self.bkg_start_index = [''] * int(self.nb_bands.get())
            self.bkg_end_index = [''] * int(self.nb_bands.get())

            self.method_selection = [''] * int(self.nb_bands.get())
            self.method_list = [''] * int(self.nb_bands.get())

            self.btn_graphical = [''] * int(self.nb_bands.get())
            self.btn_spectrogram = [''] * int(self.nb_bands.get())

            self.energy_min_var = []
            self.energy_max_var = []

            for i in range(int(self.nb_bands.get())):
                self.method_var.append(StringVar(self.frame3))
                self.method_var[i].set(self.method_choices[1])

                # Getting values for each band
                self.open_value(i)

                self.energy_min_list[i] = self.energy_min_var[i]
                self.energy_max_list[i] = self.energy_max_var[i]

                self.time_min_list[i] = self.time_min[i]
                self.time_max_list[i] = self.time_max[i]

                self.method_list[i] = self.method_var[i]

    def open_value(self, i):
        """Creates entry boxes to let the user choose the limits of the plot.
        Dependencies: entry_int.py"""

        # Liste des énergies possibles, combinées, triées et filtrées
        e_values_l = sorted(set(self.energies['e_low'])) 
        e_values_h = sorted(set(self.energies['e_high'])) 

        # Filtrer les valeurs infinies et NaN dans e_high_values
        e_values_h = [e for e in e_values_h 
                if not np.isinf(e) and not np.isnan(e)] 

        e_low_values = [int(e) for e in e_values_l if e != 0]
        e_high_values = [int(e) for e in e_values_h]

        # Créer et stocker les variables IntVar spécifiques à cette bande
        var_min = IntVar()
        var_max = IntVar()
        var_min.set(min(e_low_values))
        var_max.set(max(e_high_values))

        self.energy_min_var.append(var_min)
        self.energy_max_var.append(var_max)

        self.energy_min[i] = OptionMenu(self.frame3, var_min, *e_low_values)
        self.energy_max[i] = OptionMenu(self.frame3, var_max, *e_high_values)

        self.energy_min[i].place(relx=0.375, rely=0.4 + 0.14 * i, anchor="center", relwidth=0.3)
        self.energy_max[i].place(relx=0.625, rely=0.4 + 0.14 * i, anchor="center", relwidth=0.3)

        # self.energy_min[i].place(relx=0.375, rely=0.27 + 0.14 * i, anchor=N)
        # self.energy_max[i].place(relx=0.625, rely=0.27 + 0.14 * i, anchor=N)

        self.time_min[i] = Entry(self.frame4, width=20, state=self.state_time)
        self.time_max[i] = Entry(self.frame4, width=20, state=self.state_time)

        self.time_min[i].place(relx=0.2, rely=0.3 + 0.14 * i, anchor=N, width=150)
        self.time_max[i].place(relx=0.4, rely=0.3 + 0.14 * i, anchor=N, width=150)

        self.time_min[i].insert(0, self.start_date)
        self.time_max[i].insert(0, self.end_date)

        self.bkg_start_index[i] = 0
        self.bkg_end_index[i] = len(self.times)

        self.btn_graphical[i] = Button(self.frame4, text="Time Profile", state=self.state_time,
                                       command=lambda: self.graphical_interval(i))
        self.btn_graphical[i].place(relx=0.59, rely=0.35 + 0.14 * i, anchor=W)

        self.btn_spectrogram[i] = Button(self.frame4, text="Spectrogram", state=self.state_time,
                                       command=lambda: self.spectrogram_interval(i))
        self.btn_spectrogram[i].place(relx=0.68, rely=0.35 + 0.14 * i, anchor=W)

        self.method_selection[i] = OptionMenu(self.frame4, self.method_var[i], *self.method_choices)
        self.method_selection[i].place(relx=0.85, rely=0.35 + 0.14 * i, anchor="center")
        self.method_selection[i].config(state=self.state_time)

        self.time_min[0].config(state=self.state_bkg)
        self.time_max[0].config(state=self.state_bkg)
        self.btn_graphical[0].config(state=self.state_bkg)
        self.btn_spectrogram[0].config(state=self.state_bkg)
        self.method_selection[0].config(state=self.state_bkg)

    # def graphical_interval(self, i):
    #     """Calls for interval_selection.py to plot data graph. Time interval can directly be chosen on the graph.
    #     Stores starting and ending times in two lists. Dependencies: interval_selection.py"""
    #     self.type = self.unit_var.get()
    #     self.add_bands()
    #     self.get_data()
    #     self.plot_options()

    #     self.bkg_start_time[i], self.bkg_end_time[i] = \
    #         IntervalSelector(self.times, self.data,
    #                         #  x_scale=self.time_scale,
    #                         x_scale= self.times_datetime,
    #                          plot_label=self.plot_label,
    #                          col_label=self.data_label,
    #                          title=self.title,
    #                          xlabel=self.xlabel,
    #                          ylabel=self.ylabel,
    #                          color=self.color,
    #                          samefig=self.var_sep_times.get(),
    #                          band=i).graphical_selection()
        
    #     print(f"Graphical selection raw: start={self.bkg_start_time[i]}, end={self.bkg_end_time[i]}")
    #     print(f"Start_time ref: {self.start_time}, End_time ref: {self.end_time}")


    #     if self.bkg_start_time[i] < min(self.times):
    #         self.bkg_start_time[i] = min(self.times)
    #     if self.bkg_end_time[i] > max(self.times):
    #         self.bkg_end_time[i] = max(self.times)


    #     self.time_min[i].delete(0, 'end')
    #     self.time_max[i].delete(0, 'end')

    #     self.time_min[i].insert(0, self.convert_time_to_date(self.bkg_start_time[i]))
    #     self.time_max[i].insert(0, self.convert_time_to_date(self.bkg_end_time[i]))

    #     print(" start selction on graph : ", self.convert_time_to_date(self.bkg_start_time[i])) 
    #     print(" end selction on graph  : ", self.convert_time_to_date(self.bkg_end_time[i]))   

    #     self.bkg_start_index[i] = self.round_value(self.times, self.bkg_start_time[i])
    #     self.bkg_end_index[i] = self.round_value(self.times, self.bkg_end_time[i])

    def date_to_times_index(self, date_value):
        """
        Calcule la valeur de self.times approximative (via proportionnalité) pour une date donnée.
        """
        t_min = np.min(self.times)
        t_max = np.max(self.times)

        start_datetime = datetime.datetime.strptime(self.start_date, "%Y-%m-%dT%H:%M:%S.%f")
        end_datetime = datetime.datetime.strptime(self.end_date, "%Y-%m-%dT%H:%M:%S.%f")
        date_value = datetime.datetime.strptime(date_value, "%Y-%m-%dT%H:%M:%S.%f")

        total_seconds = (end_datetime - start_datetime).total_seconds()
        delta_seconds = (date_value - start_datetime).total_seconds()

        # linear mapping from date_value to self.times
        t_estimated = t_min + (delta_seconds / total_seconds) * (t_max - t_min)

        # to get the closest index in self.times
        index = (np.abs(self.times - t_estimated)).argmin()

        print('index:', index, 't_estimated:', t_estimated, 'date_value:', date_value  )

        return index

    def graphical_interval(self, i):
        """Affiche le graphique interactif pour sélectionner un intervalle de temps, avec axe X en datetime."""

        self.type = self.unit_var.get()
        self.add_bands()
        self.get_data()
        self.plot_options()

        # Lancement de la sélection graphique
        dt_start, dt_end = IntervalSelector(
            self.times_datetime,  # axe X en datetime
            self.data,
            x_scale=self.times_datetime,
            plot_label=self.plot_label,
            col_label=self.data_label,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            color=self.color,
            samefig=self.var_sep_times.get(),
            band=i
        ).graphical_selection()

        if dt_start is not None and dt_end is not None:
            BackgroundWindow.DATA_BKG_SELECTED = True
            print('background selection enabled', BackgroundWindow)

        # Paramètres temporels
        start_dt = BackgroundWindow.parse_datetime_string(self.start_date)
        end_dt = BackgroundWindow.parse_datetime_string(self.end_date)
        t_min = min(self.times)
        t_max = max(self.times)
        total_seconds = (end_dt - start_dt).total_seconds()

        # Fraction temporelle des sélections
        fraction_start = (dt_start - start_dt).total_seconds() / total_seconds
        fraction_end = (dt_end - start_dt).total_seconds() / total_seconds

        # Mapping proportionnel vers l'axe self.times
        rel_start = t_min + fraction_start * (t_max - t_min)
        rel_end = t_min + fraction_end * (t_max - t_min)


        # # Convertir datetime → secondes relatives à start_date
        # start_dt = BackgroundWindow.parse_datetime_string(self.start_date)
        # start_ts = start_dt.timestamp()
        
        # # Valeurs relatives (en secondes)
        # rel_start = (dt_start - start_dt).total_seconds()
        # rel_end = (dt_end - start_dt).total_seconds()

        # print(f"Graphical selection raw (datetime): start={dt_start}, end={dt_end}")
        # print(f"Relative seconds: start={rel_start}, end={rel_end}")
        # print(f"Start_time ref: {self.start_time}, End_time ref: {self.end_time}")

        
        # Clamp les valeurs dans les bornes
        rel_start = max(rel_start, min(self.times))
        rel_end = min(rel_end, max(self.times))

        if rel_start > rel_end:
            rel_start, rel_end = rel_end, rel_start

        self.bkg_start_time[i] = rel_start
        self.bkg_end_time[i] = rel_end

        # Mettre à jour les champs affichés
        self.time_min[i].delete(0, 'end')
        self.time_max[i].delete(0, 'end')
        self.time_min[i].insert(0, self.convert_time_to_date(rel_start))
        self.time_max[i].insert(0, self.convert_time_to_date(rel_end))

        # Send rel_start and rel_end to the fit_all
        BackgroundWindow.DATA_BKG_START = self.date_to_times_index(self.convert_time_to_date(rel_start))
        BackgroundWindow.DATA_BKG_END = self.date_to_times_index(self.convert_time_to_date(rel_end))

        # print(" start selection on graph : ", self.convert_time_to_date(rel_start))
        # print(" end selection on graph   : ", self.convert_time_to_date(rel_end))

        # Index dans self.times
        self.bkg_start_index[i] = self.round_value(self.times, rel_start)
        self.bkg_end_index[i] = self.round_value(self.times, rel_end)



    # def convert_time_to_date(self, time):
    #     """Converts a time in seconds to a date at format YYYY-MM-DD-HH-MM-SS. \n
    #     Parameters: \n
    #         time: time to convert in sec. \n
    #     Returns: \n
    #         date: date at format YYYY-MM-DD-HH-MM-SS."""
    #     days = int(self.date_day[2])
    #     hours = int(np.floor(time / 3600))
    #     minutes = int(np.floor((time / 3600 - hours) * 60))
    #     seconds = float(int((((time / 3600 - np.floor(time / 3600)) * 60) - minutes) * 60 * 1000) / 1000)

    #     # Delaying the day if observation if made on several days
    #     while hours < 0 or hours >= 24:
    #         diff_days = int(np.floor(hours / 24))
    #         hours -= diff_days * 24
    #         days += diff_days

    #     # Adding zeros to strings to have a better display
    #     if hours < 10:
    #         hours = '0' + str(hours)
    #     if minutes < 10:
    #         minutes = '0' + str(minutes)
    #     if seconds < 10:
    #         seconds = '0' + str(seconds)
    #     if 100 * float(seconds) - np.floor(100 * float(seconds)) == 0:
    #         seconds = str(seconds) + '0'
    #         if 10 * float(seconds) - np.floor(10 * float(seconds)) == 0:
    #             seconds = str(seconds) + '0'

    #     return str(self.date_day[0]) + '-' + str(self.date_day[1]) + '-' + str(days) + 'T' + str(hours) + ':' + \
    #         str(minutes) + ':' + str(seconds)

    def convert_time_to_date(self, t):
        """
        Convertit une valeur t (dans le référentiel self.times) en datetime réel,
        en utilisant une correspondance proportionnelle entre self.times et [start_date, end_date]
        """
        try:
            start_datetime = BackgroundWindow.parse_datetime_string(self.start_date)
            end_datetime = BackgroundWindow.parse_datetime_string(self.end_date)
        except Exception as e:
            print(f"[convert_time_to_date] Error parsing dates: {e}")
            return "Invalid Date"

        t_min = min(self.times)
        t_max = max(self.times)

        if t <= t_min:
            return start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        if t >= t_max:
            return end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

        # Proportion de la position dans self.times
        fraction = (t - t_min) / (t_max - t_min)
        delta_seconds = fraction * (end_datetime - start_datetime).total_seconds()
        new_datetime = start_datetime + datetime.timedelta(seconds=delta_seconds)
        return new_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]



    # def convert_time_to_date(self, seconds):
    #     """Ajoute un nombre de secondes à la date de début complète"""
    #     dt = self.start_datetime + datetime.timedelta(seconds=seconds)
    #     return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # ou sans ms si inutile



    def add_bands(self):
        """Gets the values of plot limits from user choice.
        energy_min and energy_max are the Entry boxes for min and max energy bands defined in open_value."""
        self.energies_low = []
        self.energies_high = []
        for i in range(int(self.nb_bands.get())):
            if self.energy_min_var[i].get() != '':
                self.round_value(self.lower_bands, int(self.energy_min_list[i].get()))
                self.energies_low.append(int(self.energy_min_list[i].get()))

            if self.energy_max_var[i].get() != '':
                self.round_value(self.upper_bands, int(self.energy_max_list[i].get()))
                self.energies_high.append(int(self.energy_max_list[i].get()))

    # =================== Plotting ===================

    # def round_value(self, liste, value):
    #     """Searches for the nearest number from the list. \n
    #     Parameters: \n
    #         liste: list where researched values are stored; \n
    #         value: number wanted to be rounded, to match one of the given list. \n
    #     Returns: \n
    #         index: index in the list of where the value has been found."""
    #     index = 0
    #     near = np.abs(value - liste[0])
    #     self.rounded = int(value)
    #     for i in range(len(liste)):
    #         diff = np.abs(value - liste[i])
    #         if near > diff:
    #             near = diff
    #             index = i
    #             self.rounded = int(liste[i])
    #     return index

    # def round_value(self, liste, value):
    #     liste = np.array(liste, dtype=np.float64)  # Cast en float64 pour éviter Overflow

    #     if value < 0:
    #         print(f"[WARNING] round_value(): valeur négative {value}, mise à 0")
    #         value = 0

    #     index = 0
    #     near = np.abs(value - liste[0])
    #     self.rounded = int(value)
    #     for i in range(len(liste)):
    #         diff = np.abs(value - liste[i])
    #         if near > diff:
    #             near = diff
    #             index = i
    #             self.rounded = int(liste[i])
    #     return index

    def round_value(self, liste, value):
        """Trouve la valeur la plus proche de 'value' dans la liste 'liste'."""
        liste = np.array(liste, dtype=np.float64)
        value = float(value)

        index = 0
        near = np.abs(value - liste[0])
        self.rounded = int(liste[0])

        for i in range(1, len(liste)):
            try:
                diff = np.abs(value - liste[i])
                if diff < near:
                    near = diff
                    index = i
                    self.rounded = int(liste[i])
            except Exception as e:
                print(f"[WARNING] Problème dans round_value avec value={value}, liste[i]={liste[i]} : {e}")
                continue

        return index



    def get_data(self):
        """Defines the Rate, Counts, and Flux for "Plot Time Profile"."""
        self.data = np.zeros((len(self.times), len(self.energies_low) + 1))
        self.data_err = np.zeros(np.shape(self.data))

        # Determines the energy distribution for different channels relative to the time of observed data
        for i in range(len(self.times)):
            for j in range(len(self.energies_low) + 1):
                # Last column is self.times
                if j == len(self.energies_low):
                    self.data[i, j] = self.times[i]
                    self.data_err[i, j] = self.times[i]

                # Each column is the data for a band
                else:
                    if np.where(self.lower_bands == self.energies_low[j])[0]:
                        a = (np.where(self.lower_bands == self.energies_low[j]))[0][0]
                    else:
                        self.round_value(self.lower_bands, self.energies_low[j])
                        a = (np.where(self.lower_bands == self.rounded))[0][0]

                    if np.where(self.upper_bands == self.energies_high[j])[0]:
                        b = (np.where(self.upper_bands == self.energies_high[j]))[0][0]
                    else:
                        self.round_value(self.upper_bands, self.energies_high[j])
                        b = (np.where(self.upper_bands == self.rounded))[0][0]

                    # Transformaing data and error data according to unit type
                    if self.type == 'Rate':
                        self.data[i, j] = sum(self.counts[i, a:(b + 1)]) / self.del_times[i]
                        self.data_err[i, j] = sum(self.counts_err[i, a:(b + 1)]) / self.del_times[i]
                    elif self.type == 'Counts':
                        self.data[i, j] = sum(self.counts[i, a:(b + 1)])
                        self.data_err[i, j] = sum(self.counts_err[i, a:(b + 1)])
                    elif self.type == 'Flux':
                        e_diff = self.area * np.abs(self.energies_high[j] - self.energies_low[j])
                        self.data[i, j] = sum(self.counts[i, a:(b + 1)]) / self.del_times[i] / e_diff
                        self.data_err[i, j] = sum(self.counts_err[i, a:(b + 1)]) / self.del_times[i] / e_diff
                    else:
                        print("No matching unit plotting type found")
        print('self.data shape:', self.data.shape)
        print('self.data values:', self.data)
        print('self.times shape:', self.times.shape)
        print('self.times values:', self.times)
        print('self.counts shape:', self.counts.shape )
        print('self.counts values:', self.counts)

    def get_bkg(self):
        """Plots the function of time by selected Unit. Uses the colormesh function provided by matplotlib library."""
        self.bkg = np.zeros(np.shape(self.data))
        for band in range(len(self.energies_low) + 1):
            if band == len(self.energies_low):
                self.bkg[:, band] = self.times
            else:
                if self.var_sep_times.get() == 1:
                    if band != 0:
                        self.method_var[band].set(self.method_var[0].get())
                        self.method_list[band] = self.method_var[band]
                        self.bkg_start_index[band] = self.bkg_start_index[0]
                        self.bkg_end_index[band] = self.bkg_end_index[0]

                start = self.bkg_start_index[band]
                end = self.bkg_end_index[band]

                # Insert conditions on the method of calculation
                if self.method_list[band].get() == "Median":
                    self.bkg[:, band] = np.median(self.data[start:end, band])

                elif self.method_list[band].get() == "Mean":
                    self.bkg[:, band] = np.mean(self.data[start:end, band])

                elif self.method_list[band].get() == "1Poly":
                    poly = np.poly1d(np.polyfit(self.times[start:end], self.data[start:end, band], 1))
                    for time in range(len(self.times)):
                        self.bkg[time, band] = poly(time)

                elif self.method_list[band].get() == "2Poly":
                    poly = np.poly1d(np.polyfit(self.times[start:end], self.data[start:end, band], 2))
                    for time in range(len(self.times)):
                        self.bkg[time, band] = poly(time)

                elif self.method_list[band].get() == "3Poly":
                    poly = np.poly1d(np.polyfit(self.times[start:end], self.data[start:end, band], 3))
                    for time in range(len(self.times)):
                        self.bkg[time, band] = poly(time)

                elif self.method_list[band].get() == "Exp":
                    poly = np.polyfit(self.times[start:end], np.log(self.data[start:end, band]), 1)
                    for time in range(len(self.times)):
                        a = np.exp(poly[1])
                        b = poly[0]
                        self.bkg[time, band] = a * np.exp(b * self.times[time])

                else:
                    self.bkg[:, band] = 0
                    print("Background detection method not found")

                # Removing all negative values in the background
                for time in range(len(self.times)):
                    if self.bkg[time, band] < 0:
                        self.bkg[time, band] = 0

        print("Background calculated for all bands :", self.bkg.shape)
        print("Background values :", self.bkg)

    def get_data_bkg(self):
        """Plots the function of time by selected Unit. Uses the colormesh function provided by matplotlib library."""
        self.data_bkg = np.zeros(np.shape(self.data))
        for band in range(len(self.energies_low) + 1):
            if band == len(self.energies_low):
                self.data_bkg[:, band] = self.times
            else:
                for time in range(len(self.times)):
                    self.data_bkg[time, band] = self.data[time, band] - self.bkg[time, band]
                    if self.data_bkg[time, band] < 0:
                        self.data_bkg[time, band] = 0
        # print("Data-Background calculated for all bands :", self.data_bkg)
        # print("Data-Background values :", self.data)
        # ✅ NEW: Share Data-Background globally
        # Store the background result
        BackgroundWindow.DATA_BKG_RESULT = self.data_bkg

    # def plot_options(self):
    #     """Saves using matplotlib all options to plot the graph to later display it. \n
    #     Dependencies: time_profile_plotting"""
    #     # Absciss legend plot
    #     file_duration = max(self.times) - min(self.times)
    #     if file_duration <= 1800:  # file duration less than 30 minutes
    #         step_x = 120
    #     elif file_duration <= 3600:  # file duration less than 1 hour
    #         step_x = 480
    #     elif file_duration <= 28800:  # file duration less than 8 hours
    #         step_x = 3600
    #     else:  # file duration more than 8 hours
    #         step_x = 7200

    #     self.time_scale = np.arange(0, file_duration, step_x)
    #     x_labels_plot = list(
    #         map(lambda x: (str(datetime.timedelta(seconds=float(x)))).split('.')[0][:-3], self.time_scale +
    #             self.start_time))

    #     self.plot_label = []
    #     for i in range(len(x_labels_plot)):
    #         if 'day' in x_labels_plot[i]:
    #             self.plot_label.append((x_labels_plot[i].split(','))[1])
    #         else:
    #             self.plot_label.append(x_labels_plot[i])

    #     self.xlabel = 'Start time: ' + str(self.start_date) + ' -- End time : ' + str(self.end_date)

    #     # Unit plotting
    #     if self.type == "Rate":
    #         self.ylabel = 'Rate (Counts/s) by Bands'
    #         self.title = 'Time Profile Plotting Rate (Counts/s)'

    #     elif self.type == "Counts":
    #         self.ylabel = 'Counts by Bands'
    #         self.title = 'Time Profile Plotting Counts'

    #     elif self.type == "Flux":
    #         self.ylabel = 'Flux by Bands'
    #         self.title = 'Time Profile Plotting Flux'

    #     else:
    #         print("No matching unit plotting type found")

    #     # Labeling energy bands on the plot
    #     self.data_label = []
    #     for data_type in ['Original data', 'Background', 'Corrected data']:
    #         label = [data_type + ' for ' + str(low) + '-' + str(high) + 'keV' for low, high in
    #                  zip(self.energies_low, self.energies_high)]
    #         label.append('Times')
    #         self.data_label.append(label)

    # def plot_options(self):
    #     """Génère 10 labels temporels bien répartis sur l’axe X, proportionnels à self.times et aux dates réelles."""
        
    #     from datetime import timedelta

    #     n_ticks = 10
    #     t_min = min(self.times)
    #     t_max = max(self.times)
    #     self.time_scale = np.linspace(t_min, t_max, n_ticks, endpoint=True).astype(int)
    #     print(f"time_scale: {self.time_scale}")

    #     # Datetimes réels de début et de fin
    #     start_datetime = BackgroundWindow.parse_datetime_string(self.start_date)
    #     end_datetime = BackgroundWindow.parse_datetime_string(self.end_date)
    #     total_seconds = (end_datetime - start_datetime).total_seconds()

    #     self.plot_label = []
    #     for t in self.time_scale:
    #         fraction = (t - t_min) / (t_max - t_min)
    #         dt_offset = fraction * total_seconds
    #         label_time = start_datetime + timedelta(seconds=int(dt_offset))

    #         # Format des labels
    #         label = label_time.strftime('%H:%M') if total_seconds <= 86400 else label_time.strftime('%m-%d\n%H:%M')
    #         print(f"Label for time {t} (fraction={fraction:.2f}): {label}")
    #         self.plot_label.append(label)

    #     self.xlabel = f'Start time: {self.start_date} -- End time: {self.end_date}'

    #     # Unité Y et titre
    #     self.type = self.unit_var.get()
    #     if self.type == "Rate":
    #         self.ylabel = 'Rate (Counts/s) by Bands'
    #         self.title = 'Time Profile Plotting Rate (Counts/s)'
    #     elif self.type == "Counts":
    #         self.ylabel = 'Counts by Bands'
    #         self.title = 'Time Profile Plotting Counts'
    #     elif self.type == "Flux":
    #         self.ylabel = 'Flux by Bands'
    #         self.title = 'Time Profile Plotting Flux'
    #     else:
    #         print("No matching unit plotting type found")

    #     # Légendes
    #     self.data_label = []
    #     for data_type in ['Original data', 'Background', 'Corrected data']:
    #         label = [f'{data_type} for {low}-{high} keV'
    #                 for low, high in zip(self.energies_low, self.energies_high)]
    #         label.append('Times')
    #         self.data_label.append(label)

    def plot_options(self):
        """Génère des labels temporels à intervalles dynamiques pour l’axe X basé sur self.times."""
        from datetime import timedelta

        # Début / fin réels
        start_datetime = BackgroundWindow.parse_datetime_string(self.start_date)
        end_datetime = BackgroundWindow.parse_datetime_string(self.end_date)
        total_seconds = (end_datetime - start_datetime).total_seconds()

        # Déterminer un pas dynamique
        if total_seconds <= 1800:             # ≤ 30 min
            interval_sec = 120                # 2 min
        elif total_seconds <= 3600:           # ≤ 1h
            interval_sec = 480                # 8 min
        elif total_seconds <= 10800:          # ≤ 3h
            interval_sec = 1200               # 20 min
        elif total_seconds <= 28800:          # ≤ 8h
            interval_sec = 1800               # 30 min
        elif total_seconds <= 43200:          # ≤ 12h
            interval_sec = 3600               # 1h
        elif total_seconds <= 86400:          # ≤ 24h
            interval_sec = 7200               # 2h
        elif total_seconds <= 2*86400:        # ≤ 2 jours
            interval_sec = 10800              # 3h
        elif total_seconds <= 5*86400:        # ≤ 5 jours
            interval_sec = 21600              # 6h
        elif total_seconds <= 10*86400:       # ≤ 10 jours
            interval_sec = 43200              # 12h
        else:                                 # > 10 jours
            interval_sec = 86400              # 1 jour

        # Générer les ticks réguliers
        tick_datetimes = []
        current = start_datetime
        while current <= end_datetime:
            tick_datetimes.append(current)
            current += timedelta(seconds=interval_sec)

        # Remap des datetimes vers l’axe X (self.times)
        t_min = min(self.times)
        t_max = max(self.times)

        self.time_scale = [
            t_min + (dt - start_datetime).total_seconds() * (t_max - t_min) / total_seconds
            for dt in tick_datetimes
        ]

        self.times_datetime = [
            start_datetime + timedelta(seconds=(t - t_min) / (t_max - t_min) * total_seconds)
            for t in self.times
        ]

        # Générer les labels (format variable selon la durée)
        self.plot_label = [
            dt.strftime('%H:%M') if total_seconds <= 86400 else dt.strftime('%m-%d\n%H:%M')
            for dt in tick_datetimes
        ]

        self.xlabel = f'Start time: {self.start_date} -- End time: {self.end_date}'

        # Unité Y et titre
        self.type = self.unit_var.get()
        if self.type == "Rate":
            self.ylabel = 'Rate (Counts/s) by Bands'
            self.title = 'Time Profile Plotting Rate (Counts/s)'
        elif self.type == "Counts":
            self.ylabel = 'Counts by Bands'
            self.title = 'Time Profile Plotting Counts'
        elif self.type == "Flux":
            self.ylabel = 'Flux by Bands'
            self.title = 'Time Profile Plotting Flux'
        else:
            print("No matching unit plotting type found")

        # Légendes
        self.data_label = []
        for data_type in ['Original data', 'Background', 'Corrected data']:
            label = [f'{data_type} for {low}-{high} keV'
                    for low, high in zip(self.energies_low, self.energies_high)]
            label.append('Times')
            self.data_label.append(label)


    # def time_profile_plotting(self):
    #     """Calls for all previous functions to get all data needed, then show the plot.
    #     Dependencies: get_data, get_bkg, get_data_bkg, plot_options"""
    #     plt.close()
    #     self.add_bands()
    #     self.type = self.unit_var.get()
    #     self.get_data()
    #     plt.figure()
    #     self.plot_options()


    #     from matplotlib.dates import AutoDateLocator, AutoDateFormatter, DateFormatter
    #     from datetime import datetime, timedelta

    #     start_datetime = BackgroundWindow.parse_datetime_string(self.start_date)
    #     end_datetime = BackgroundWindow.parse_datetime_string(self.end_date)
    #     t_min = min(self.times)
    #     t_max = max(self.times)
    #     total_seconds = (end_datetime - start_datetime).total_seconds()

    #     # Conversion de self.times → datetimes proportionnels
    #     self.times_datetime = [
    #         start_datetime + timedelta(seconds=(t - t_min) / (t_max - t_min) * total_seconds)
    #         for t in self.times
    #     ]

    #     print(f"Converted times to datetimes: {self.times_datetime}")



    #     # Plotting data
    #     if self.var_error.get():
    #         yerr = self.data_err
    #     else:
    #         yerr = np.zeros(np.shape(self.data_err))

    #     if self.var_og_data.get():
    #         df = pd.DataFrame(self.data, index=self.times_datetime, columns=self.data_label[0])
    #         for band in range(len(self.energies_low)):
    #             df.plot(x='Times', y=self.data_label[0][band], yerr=yerr[:, band], color=self.color[band],
    #                     ax=plt.gca(), linewidth=0.25)

    #     if self.var_bkg.get() or self.var_data_bkg.get():
    #         self.get_bkg()

    #         if self.var_bkg.get():
    #             df = pd.DataFrame(self.bkg, index=self.times_datetime, columns=self.data_label[1])
    #             for band in range(len(self.energies_low)):
    #                 df.plot(x='Times', y=self.data_label[1][band], color=self.color[band],
    #                         ax=plt.gca(), linewidth=0.1)

    #         if self.var_data_bkg.get():
    #             self.get_data_bkg()
    #             df = pd.DataFrame(self.data_bkg, index=self.times_datetime, columns=self.data_label[2])
    #             for band in range(len(self.energies_low)):
    #                 df.plot(x='Times', y=self.data_label[2][band], yerr=yerr[:, band], color=self.color[band],
    #                         ax=plt.gca(), linewidth=1)

    #     if self.grid_var.get():
    #         # show grid if checkbox is checked
    #         plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.8)
    #         plt.minorticks_on()
    #         plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    #     else:
    #         plt.grid(False)

    #     # plt.xticks(self.time_scale, self.plot_label)
    #     plt.xlabel(self.xlabel)
    #     plt.ylabel(self.ylabel)
    #     plt.title(self.title)


    #     # Mise en place des ticks dynamiques
    #     ax = plt.gca()
    #     locator = AutoDateLocator()
    #     formatter = DateFormatter('%H:%M')
    #     ax.xaxis.set_major_locator(locator)
    #     ax.xaxis.set_major_formatter(formatter)

    #     # Mise à jour sur zoom
    #     def update_xticks_on_zoom(event):
    #         ax = event.canvas.figure.axes[0]
    #         ax.xaxis.set_major_locator(locator)
    #         ax.xaxis.set_major_formatter(formatter)
    #         event.canvas.draw_idle()

    #     plt.gcf().canvas.mpl_connect('draw_event', update_xticks_on_zoom)
    #     plt.gcf().autofmt_xdate()

    #     plt.show()


    def time_profile_plotting(self):
        """Trace le profil temporel avec échelle dynamique des ticks temporels en fonction du zoom."""

        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from matplotlib.dates import AutoDateLocator, DateFormatter, MinuteLocator
        from datetime import datetime, timedelta

        plt.close()
        self.add_bands()
        self.type = self.unit_var.get()
        self.get_data()
        plt.figure()
        self.plot_options()

        # Conversion times → datetime (pour axe X)
        start_datetime = BackgroundWindow.parse_datetime_string(self.start_date)
        end_datetime = BackgroundWindow.parse_datetime_string(self.end_date)
        t_min = min(self.times)
        t_max = max(self.times)
        total_seconds = (end_datetime - start_datetime).total_seconds()

        self.times_datetime = [
            start_datetime + timedelta(seconds=(t - t_min) / (t_max - t_min) * total_seconds)
            for t in self.times
        ]

        print(f"Converted times to datetimes: {self.times_datetime[0]} ... {self.times_datetime[-1]}")

        # Choix des erreurs
        yerr = self.data_err if self.var_error.get() else np.zeros(np.shape(self.data_err))

        # Tracé des données originales
        if self.var_og_data.get():
            df = pd.DataFrame(self.data, index=self.times_datetime, columns=self.data_label[0])
            for band in range(len(self.energies_low)):
                df.plot(y=self.data_label[0][band], yerr=yerr[:, band], color=self.color[band],
                        ax=plt.gca(), linewidth=0.25)

        # Données background
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
                    df.plot(y=self.data_label[2][band], yerr=yerr[:, band], color=self.color[band],
                            ax=plt.gca(), linewidth=1)

        # Grille
        if self.grid_var.get():
            plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.8)
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        else:
            plt.grid(False)
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', linewidth=0.1, alpha=0.1)

        # Labels
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        # plt.grid(True, which="both", ls="--", alpha=0.5)

        # Axe X dynamique
        ax = plt.gca()
        ax.set_xlim(self.times_datetime[0], self.times_datetime[-1]) # ADJUSTE LIMITE X

        # ax.grid(False, which='major')  # principal grid

        # ax.minorticks_on()  # Active ticks secondaires
        # ax.grid(True, which='minor', linestyle=':', linewidth=0.1, alpha=0.1)

        # plt.draw()


        locator = AutoDateLocator()
        # locator = MinuteLocator(interval=10)
        formatter = DateFormatter('%H:%M:%S')  # Format par défaut : HH:MM

        ax.xaxis.set_major_locator(locator)  # Ticks toutes les 30 minutes
        ax.xaxis.set_major_formatter(formatter)

        def update_xticks_on_zoom(event):
            ax = event.canvas.figure.axes[0]
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            event.canvas.draw_idle()

        plt.gcf().canvas.mpl_connect('draw_event', update_xticks_on_zoom)
        plt.gcf().autofmt_xdate()

        plt.show()


    # def spectrogram_interval(self, i):
    #     """Open spectrogram plot with graphical selection for background definition."""
    #     import plotting
    #     import second

    #     # Create plotting instance using selected FITS file
    #     plot_instance = plotting.Input(second.SecondWindow.fname)

    #     # Call the spectrogram plotting
    #     plot_instance._Input__plot_spectrogram('rate')  # Or 'counts' or 'flux'

    #     # ⚠️ OPTIONAL : Here you could add your custom RectangleSelector to define the interval
    #     # For example, if you want to store selection in self.time_min[i] / self.time_max[i]
    #     # But this part depends on how your plotting.py supports graphical selection

    def spectrogram_interval(self, i):
        """Open spectrogram plot with graphical selection for background definition, bypassing the popup."""
        import plotting
        import second
        import matplotlib.pyplot as plt
        from matplotlib.widgets import SpanSelector
        import numpy as np

        # Create plotting instance using selected FITS file
        current_fname = BackgroundWindow.fname if BackgroundWindow.fname else second.SecondWindow.fname
        print('fname:', BackgroundWindow.fname)
        plot_instance = plotting.Input(current_fname)
        # plot_instance = plotting.Input(second.SecondWindow.fname)

        # ✅ Monkey-patch specgm_lim to avoid popup!
        def no_popup_specgm_lim():
            lower_clean = plot_instance.lower_bands[np.isfinite(plot_instance.lower_bands)]
            upper_clean = plot_instance.upper_bands[np.isfinite(plot_instance.upper_bands)]
            if lower_clean.size == 0 or upper_clean.size == 0:
                plot_instance.energy_min = 0.0
                plot_instance.energy_max = 20.0
                print("[NO POPUP] Warning: Bands contained only NaNs. Defaulting to 0-20 keV")
            else:
                plot_instance.energy_min = np.min(lower_clean)
                plot_instance.energy_max = np.max(upper_clean)
                print(f"[NO POPUP] Energy bounds set to: {plot_instance.energy_min} - {plot_instance.energy_max}")


        plot_instance.specgm_lim = no_popup_specgm_lim

        # ✅ Now call the spectrogram plotting - will use our no-popup version
        plot_instance._Input__plot_spectrogram('rate')

        # Add selector for time
        from matplotlib import dates as mdates
        from datetime import datetime

        def onselect(xmin, xmax):
            print(f"Raw selected X: {xmin} - {xmax}")

            # Convert matplotlib float date to datetime
            dt_start = mdates.num2date(xmin)
            dt_end = mdates.num2date(xmax)
            dt_start = dt_start.replace(tzinfo=None)  # Remove timezone info if present
            dt_end = dt_end.replace(tzinfo=None) 
            print(f"Selected region (datetime): {dt_start} - {dt_end}")

            # Write into your Entry widgets
            self.time_min[i].delete(0, 'end')
            self.time_max[i].delete(0, 'end')
            self.time_min[i].insert(0, dt_start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
            self.time_max[i].insert(0, dt_end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])

            # Send Start and End data to the fit_all
            BackgroundWindow.DATA_BKG_START = dt_start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            BackgroundWindow.DATA_BKG_END = dt_end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

            # ✅ Close the plot window
            plt.close()


        fig = plt.gcf()
        ax = plt.gca()
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='red'))
        

        plt.show()





