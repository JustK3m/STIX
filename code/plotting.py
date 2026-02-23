import matplotlib
matplotlib.use("TkAgg") # Use TkAgg backend for matplotlib
import numpy as np
import pandas as pd
import datetime
import re
from tkinter import *
from tkinter import messagebox
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import ticker as tck
from pandas.plotting import register_matplotlib_converters

import background
from entry_int import EntryInt
import second

from matplotlib.dates import AutoDateLocator, DateFormatter
from datetime import datetime, timedelta

register_matplotlib_converters()


class Input:
    """Class to load the parameters from input data, data times, and plot Spectrum, Time Profile and Spectrogram.
       Called Units: Rate, Counts, Flux."""

    def __init__(self, file, start=None, end=None, hours=None):
        """When opening the .fits file, the returned object, called hdulist, behaves like a Python list and each element
        maps to a Header - Data Unit(HDU). We are primarily interested in RATE extension which contains the spectral
        data. Extracted object has two important attributes: data, which behaves like an array, can be used to access
        to the numerical data and header, which behaves like a dictionary, can be used to access to the header
        information. Also, if the start and end times are specified, extracts from the file the dates and times. \n
        Parameters: \n
            file: the file in your computer containing data you want to analyse; \n
            start: starting time of the plotting; optionnal; \n
            end: ending time of the plotting; optionnal; \n
            hours: keeps the old value for data time for future comparisons; optionnal."""
        
        self.show = True                                # Shows the plots only if this variable is True
        if start and end and hours:                     # If "Entire File" checkbox is unticked:
            self.entire_file = False                        # Class will plot data for a reduced time range
        else:                                           # If "Entire File" checkbox is ticked:
            self.entire_file = True                         # Class will plot data for all time range

        data, data_energies, header_dates, header_energy = self.__load_data(file)  # Loading data

        hdu = fits.open(file)  # Opening the file
        data = second.SecondWindow.extract_stix_data(hdu)  # Extracting data from the file
        headers = second.SecondWindow.extract_stix_header(hdu)  # Extracting header from the file

        self.counts = data['counts']                       # Matrix contaning the counts per band in function of time time
        self.times = data['time']                         # Index of times for x axis
        self.del_times = self.delay_times(data['timedel'])  # List containing the difference between two successive times
        self.area = 6                                   # Value of the surface of detection (in cm²) to compute the flux
        self.energies_bin = data['counts'].shape[1]            # Constant used to define the plot

        self.energies = data

        self.nb_bands = StringVar()                         # Number of energy bands to plot
        self.lower_bands = np.array(data['e_low'])    # Lower bounds for energy bands from .fits file
        self.upper_bands = np.array(data['e_high'])   # Upper bounds for energy bands from .fits file
        self.energies_low = []                              # Lower bounds for energy bands chosen by user
        self.energies_high = []                             # Upper bounds for energy bands chosen by user
        self.energies_low_index = []                        # Lower bounds indexes for energy bands chosen by user
        self.energies_high_index = []                       # Upper bounds indexes for energy bands chosen by user
        self.rounded = int()                                # Nearest value to fit in energy list; used in self.round


        if start:
            self.start_date = start                         # Starting date from import at format YYYY-MM-DD-HH-MM-SS
        else:
            self.start_date = headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown'))           # Starting date from header at format YYYY-MM-DD-HH-MM-SS
        if end:
            self.end_date = end                             # Ending date from import at format YYYY-MM-DD-HH-MM-SS
        else:
            self.end_date = headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))                # Ending date from header at format YYYY-MM-DD-HH-MM-SS
        if hours:
            self.hours_fixed = hours                        # Backup variable of initial time for refreshing

        self.start_time = self.find_time(self.start_date)   # Starting time in seconds
        self.end_time = self.find_time(self.end_date)       # Ending time in seconds

        self.data_start = self.find_time(headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown')))  # Starting time of data in seconds
        self.data_end = self.find_time(headers.get('DATE_END', headers.get('DATE-END', 'Unknown')))    # Ending time of data in seconds

        ## uses the entire file dates
        self.start_date_file = headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown'))
        self.end_date_file = headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))

        self.index_start_list = []                  # List of all indexes included in reduced file
        self.index_end_list = []                    # List of all indexes included in reduced file
        self.index_start = int()                    # Starting time index for reduced file
        self.index_end = int()                      # Ending time index for reduced file

        self.type = str()                           # Data unit type (rate, counts, flux)
        self.data_plot = []                         # Data converted to unit type

        self.btn_info = None                        # Checkbutton "Show Information"
        self.info = IntVar()                        # Binary variable for checkbutton
        self.canvas_canal = StringVar()             # Canvas used to show summarized information
        self.window_elimits = None                  # Window opened to change scale of axis (linear / log) for spectrum
        self.canvas_plot = None                     # Canvas used to show energy bands entry boxes for time profile
        self.window_spec_limits = None              # Window opened to specify axis limits and scale for spectrogram

        self.btn_plot_spectrum = None               # Button to plot spectrum

        self.energy_min = None                      # Entrybox for min energy band to plot the time profile
        self.energy_max = None                      # Entrybox for max energy band to plot the time profile
        self.energy_min_list = list()               # List containing all minima of energy bands
        self.energy_max_list = list()               # List containing all maxima of energy bands
        self.text_min_energy = StringVar()          # Text "Min energy"
        self.text_max_energy = StringVar()          # Text "Max energy"

        self.ylim_min = int()                       # Min energy limit along y axis when plotting spectrogram
        self.ylim_max = int()                       # Max energy limit along y axis when plotting spectrogram
        self.e_ylim_min = StringVar()               # Entrybox containing min y limit for spectrogram
        self.e_ylim_max = StringVar()               # Entrybox containing max y limit for spectrogram
        self.text_spec_limits = StringVar()         # Text "Choose your scale"
        self.text_energy_elimit = StringVar()       # Text "Enter min & max energy for spectrogram"
        self.btn_plot_spgm = None                   # Button to plot spectrogram
        self.raised_power = float()                 # Format of power for spectrogram scale legend

        self.scalex = StringVar()                   # Variable in entry box for scale of x axis
        self.scalex.set("linear")                   # Set by default to linear scale
        self.scaley = StringVar()                   # Variable in entry box for scale of y axis
        self.scaley.set("linear")                   # Set by default to linear scale
        self.rx_text = StringVar()                  # Text "x axis"
        self.ry_text = StringVar()                  # Text "y axis"
        self.rx_lin = None                          # Radiobox to choose 'linear' for x axis
        self.rx_log = None                          # Radiobox to choose 'log' for x axis
        self.ry_lin = None                          # Radiobox to choose 'linear' for y axis
        self.ry_log = None                          # Radiobox to choose 'log' for y axis

    # =================== 0. Global functions ===================

    @staticmethod
    def __load_data(file):
        """Reads the Data and Header contents from input file. Loads the input file choosen in 'Select Input' section.
        Returns respectively a table containing datas, energies, dates and channels.\n
        Parameters: \n
            file: contains the data in a fits file. \n
        Retruns: \n
            data, data_energies, header_dates, header_energy"""
        hdulist = fits.open(file)  # Reads the data
        hdulist.info()  # Displays the content of the read file
        return hdulist[2].data, hdulist[3].data, hdulist[0].header, hdulist[3].header

    @staticmethod
    def delay_times(data):
        """Creates a TEMPORARY -1 interval, because of the step between deltimes data. Returns the shifted data. \n
        Parameters: \n
            data: original data. \n
        Returns: \n
            del_times: times delayed"""
        del_times = np.zeros_like(data)
        for i in range(1, len(data)):
            del_times[i] = data[i - 1]
        del_times[0] = data[0]
        return del_times

    @staticmethod
    def find_time(date):
        """Converts the starting time in the data. \n
        Parameters: \n
            date: starting date as a list. \n
        Returns: \n
            found_time: time converted in seconds."""
        date_split = re.split('T', date)  # Saves a tuple containing the days, and the time
        date_time = re.split(':', date_split[1])  # Splits time in 3 coponents: hr, min, s
        date_time_int = re.split('\\.', date_time[2])  # Keeps only the integer part in seconds
        found_time = int(date_time[0]) * 3600 + int(date_time[1]) * 60 + int(date_time_int[0])
        return found_time

    def acq_time(self, label):
        """If not using the entire file, checks if acquisition time is > 24 h and reduces data
        according to the size of chosen time period.\n
        Parameters: \n
            label: list of times indexes, in seconds."""
        startIndexh = self.data_start - self.start_time
        endIndexh = self.data_end - self.start_time
        day = False  # Condition "Acquisition time > 24 h"
        for i in label:
            if i > 86400:  # If in label there is a time > 86400 s
                day = True  # Acquisition time now > 24 h
        if day and self.data_start < 3600:  # If acquisition time > 24 h
            startIndexh += 86400  # Adding 24 h
        if day and self.data_end < 3600:  # If acquisition time > 24 h
            endIndexh += 86400  # Adding 24 h
        self.index_start, self.index_end = self.time_index(self.times, startIndexh, endIndexh)


    # def acq_time(self, label):
    #     day = False
    #     for i in label:
    #         if i > 86400:
    #             day = True

    #     # Correct: use self.start_time and self.end_time directly
    #     startIndexh = self.start_time
    #     endIndexh = self.end_time

    #     if day and self.start_time < 3600:
    #         startIndexh += 86400
    #     if day and self.end_time < 3600:
    #         endIndexh += 86400

    #     self.index_start, self.index_end = self.time_index(self.times, startIndexh, endIndexh)
    #     print(f"Acquisition time adjusted: start index {self.index_start}, end index {self.index_end}")
    #     print(f"Start time: {self.start_time}, End time: {self.end_time}")
    #     print(f"Adjusted start index: {startIndexh}, Adjusted end index: {endIndexh}")


    def log_axis(self, window, relx, rely):
        """Displays the radio boxes allowing the user to choose between linear and logarithmic axis. \n
        Parameters: \n
            window: current window on which the radio boxes need to be displayed; \n
            relx and rely: relative position of the North-West of the whole set on the window."""
        self.rx_text = Label(window, text="x axis:", fg='blue')
        self.ry_text = Label(window, text="y axis:", fg='blue')
        self.rx_text.place(relx=relx, rely=rely)
        self.ry_text.place(relx=relx, rely=rely + 0.10)

        self.rx_lin = Radiobutton(window, text="Linear", variable=self.scalex, value="linear")
        self.ry_lin = Radiobutton(window, text="Linear", variable=self.scaley, value="linear")
        self.rx_log = Radiobutton(window, text="Logarithmic", variable=self.scalex, value="log")
        self.ry_log = Radiobutton(window, text="Logarithmic", variable=self.scaley, value="log")
        self.rx_lin.place(relx=relx + 0.13, rely=rely)
        self.rx_log.place(relx=relx + 0.32, rely=rely)
        self.ry_lin.place(relx=relx + 0.13, rely=rely + 0.10)
        self.ry_log.place(relx=relx + 0.32, rely=rely + 0.10)

    # =================== 1. Time Profile Plotting ===================

    def rate_vs_time_plotting(self):
        """If the selected unit is rate, passes the string 'rate' to the time profile setting. \n
        Dependencies: second.py"""
        self.type = 'rate'
        self.__time_profile_setting()

    def counts_vs_time_plotting(self):
        """If the selected unit is counts, passes the string 'counts' to the time profile setting. \n
        Dependencies: second.py"""
        self.type = 'counts'
        self.__time_profile_setting()

    def flux_vs_time_plotting(self):
        """If the selected unit is flux, passes the string 'flux' to the time profile setting. \n
        Dependencies: second.py"""
        self.type = 'flux'
        self.__time_profile_setting()

    def __time_profile_setting(self):
        """Plots the function of time by selected Unit.
            Uses two frames: the first one to select values, the second one to plot figures. \n
            Parameters: \n
                typ (str): the type of plotting between 'rate', 'counts', and 'flux'; \n
                show (bool): if True, displays the window for plot time options."""
        self.from_import = False  # For now, no other file calls this function

        self.toptime = Toplevel()  # Creating new window for time profile settings
        self.toptime.title('STIX PlotTime Options')
        self.toptime.geometry("400x600")

        Label(self.toptime, text="Plot Time Profile", fg="red",
              font="Helvetica 12 bold italic").place(relx=0.5, rely=0.1, anchor=N)
        self.frame_time = LabelFrame(self.toptime, relief=RAISED, borderwidth=1)
        self.frame_time.place(relx=0.01, rely=0.05, relheight=0.9, relwidth=0.98)

        # Energy bands selection
        self.text_bands = Label(self.frame_time, text="Selection of Energy bands number: ")
        self.text_bands.place(relx=0.02, rely=0.06, anchor=W)
        self.bands_choices = ('1', '2', '3', '4', '5')
        self.nb_bands = StringVar(self.frame_time)
        self.nb_bands.set('-')  # Default value
        self.selection_bands = OptionMenu(self.frame_time, self.nb_bands, *self.bands_choices)
        self.selection_bands.place(relx=0.52, rely=0.06, anchor=W)
        self.nb_bands.trace("w", self.entries_list)  # When selecting a number of bands, Entryboxes appear

        self.btn_refresh = Button(self.frame_time, text='Refresh', command=self.do_refresh)
        self.btn_refresh.place(relx=0.25, rely=0.92, anchor=N)
        self.btn_close = Button(self.frame_time, text='Close', command=self.destroy_bis)
        self.btn_close.place(relx=0.50, rely=0.92, anchor=N)

        if self.type == 'counts':
            self.btn_plot = Button(self.frame_time, text='Do Plot', command=self.__do_plot_counts)
        elif self.type == 'rate':
            self.btn_plot = Button(self.frame_time, text='Do Plot', command=self.__do_plot_rate)
        elif self.type == 'flux':
            self.btn_plot = Button(self.frame_time, text='Do Plot', command=self.__do_plot_flux)
        else:
            self.btn_plot = None
            print("Type not found.")

        self.btn_plot.place(relx=0.75, rely=0.92, anchor=N)

    def do_refresh(self):
        """Destroys the current plotting window, resets all values for matrix plotting, and reopens the plotting window.
        Function only called as a command function in the time profile settings."""
        self.toptime.destroy()
        self.__time_profile_setting()
        self.energies_low = []
        self.energies_high = []
        self.energies_low_index = []
        self.energies_high_index = []
        if not self.entire_file:
            self.start_time = self.find_time(self.hours_fixed)
        self.energy_min = None
        self.energy_max = None
        print("\nValues refreshed")

    def destroy_bis(self):
        """Closes 'STIX PlotTime Options' window when clicking 'Close' button. Used in the time profile setting."""
        self.toptime.destroy()

    def entries_list(self, frame=None, iterations=str(), log=True):
        """Allows to create all buttons and Entry boxes to choose energy values.
        Creates a new window called 'canvas_plot' if none already exists.
        Parameters:
            iterations: number of entry boxes;
            frame: where the entries are displayed; default value: None, displayed in plotting window;
            log: if True, will show radioboxes to set axes with a log scale."""

        if self.from_import:  # If function is called from another file
            self.canvas_plot = frame
            self.nb_bands.set(iterations)
            entry_relx = 0.11
            entry_rely = 0.16
        else:
            self.canvas_plot = Canvas(self.toptime, width=350, height=300)
            self.canvas_plot.place(relx=0.5, rely=0.15, anchor=N)
            entry_relx = 0.2
            entry_rely = 0.05

        self.text_min_energy = Label(self.canvas_plot, text="Min energy")
        self.text_min_energy.place(relx=0.5 - entry_relx, rely=entry_rely, anchor=N)
        self.text_max_energy = Label(self.canvas_plot, text="Max energy")
        self.text_max_energy.place(relx=0.5 + entry_relx, rely=entry_rely, anchor=N)

        if log:
            """If the user wants to set the scale of the axes, creates a new window called 'window_spec_limits'."""
            self.grid_var2 = BooleanVar()
            self.grid_check2 = Checkbutton(self.canvas_plot, text="Show grid", variable=self.grid_var2)
            self.grid_check2.place(relx=0.21, rely=0.82)
            # self.grid_var2.trace_add('write', self.__update_grid_2)

            # To print information about the canal sum
            self.btn_info = Checkbutton(self.canvas_plot, text="Show Information",
                                        variable=self.info, command=self.sum_canal, state=NORMAL)
            self.btn_info.place(relx=0.21, rely=0.89)

            # Scaling
            self.log_axis(self.canvas_plot, 0.12, 0.64)

        if self.nb_bands.get().isdigit():
            self.energy_min_list = [''] * int(self.nb_bands.get())
            self.energy_max_list = [''] * int(self.nb_bands.get())
            for i in range(int(self.nb_bands.get())):
                # Getting values for each band
                self.open_value(i)
                self.energy_min_list[i] = self.energy_min_var
                self.energy_max_list[i] = self.energy_max_var

            return self.energy_min_list, self.energy_max_list

    def sum_canal(self):
        """Prints the sum of canal if the box "Show information" is ticked."""
        if self.info.get() == 1:
            self.canvas_canal = Canvas(self.toptime, width=350, height=100, bg='ivory')
            self.canvas_canal.place(relx=0.5, rely=0.65, anchor=N)
            txt = []
            if self.energies_low_index:
                for i in range(len(self.energies_low_index)):
                    txt.append([i + 1, self.energies_low_index[i], self.energies_high_index[i]])
            else:
                print('No information currently available')

            frame_summarize_canal = LabelFrame(self.canvas_canal, relief=RAISED, borderwidth=2)
            frame_summarize_canal.place(relheight=1, relwidth=1)

            txtcanal = []
            for i in range(len(txt)):
                txtcanal.append('\nCanal selected for Energy band {} is:'.format(txt[i][0]))
                txtcanal.append(txt[i][1])
                txtcanal.append('to')
                txtcanal.append(txt[i][2])

            listcanal = Text(frame_summarize_canal)
            listcanal.insert(END, str(txtcanal))
            listcanal.pack()

        else:
            print('Destroying canvas')
            self.canvas_canal.destroy()

    def open_value(self, i):
        """Creates entry boxes to let the user choose the limits of the plot. \n
        Dependencies: entry_int.py"""

        # Créer les listes de valeurs d'énergie 
        e_low_values = sorted(set(self.energies['e_low'])) 
        e_high_values = sorted(set(self.energies['e_high'])) 

        
        # Filtrer les valeurs infinies et NaN dans e_high_values
        e_high_values = [e for e in e_high_values 
                if not np.isinf(e) and not np.isnan(e)] 

        # Convertir les valeurs en entiers
        e_low_values_int = [int(e) for e in e_low_values if e != 0]
        e_high_values_int = [int(e) for e in e_high_values]

        # Créer les variables StringVar pour les OptionMenu
        self.energy_min_var = IntVar()
        self.energy_max_var = IntVar()

        # Définir les valeurs par défaut pour min et max énergie
        self.energy_min_var.set(min(e_low_values_int))
        self.energy_max_var.set(max(e_high_values_int))

        # Créer les OptionMenu pour l'énergie minimale et maximale
        self.energy_min = OptionMenu(self.canvas_plot, self.energy_min_var, *e_low_values_int)
        self.energy_max = OptionMenu(self.canvas_plot, self.energy_max_var, *e_high_values_int)

        # Positionner les OptionMenu dans la fenêtre (avec place(), ajuster la position si nécessaire)
        if self.from_import:
            self.energy_min.place(relx=0.375, rely=0.27 + 0.14 * i, anchor=N)
            self.energy_max.place(relx=0.625, rely=0.27 + 0.14 * i, anchor=N)
        else:
            self.energy_min.place(relx=0.3, rely=0.14 + 0.1 * i, anchor=N)
            self.energy_max.place(relx=0.7, rely=0.14 + 0.1 * i, anchor=N)


    def __do_plot_rate(self):
        """Collects rate data to do the plotting."""
        self.add_bands()
        data = self.__get_data("rate")
        self.__time_profile_plotting(data, "rate")

    def __do_plot_counts(self):
        """Collects counts data to do the plotting."""
        self.add_bands()
        data = self.__get_data("counts")
        self.__time_profile_plotting(data, "counts")

    def __do_plot_flux(self):
        """Collects flux data to do the plotting."""
        self.add_bands()
        data = self.__get_data("flux")
        self.__time_profile_plotting(data, "flux")

    def add_bands(self):
        """Gets the values of plot limits from user choice."""
        for i in range(int(self.nb_bands.get())):
            if self.energy_min_var.get() != '':
                self.rounded = self.round_energy(self.lower_bands, int(self.energy_min_list[i].get()))
                a_index = (np.where(self.lower_bands == float(self.rounded)))[0][0]
                self.energies_low.append(int(self.energy_min_list[i].get()))
                self.energies_low_index.append(int(a_index))

            if self.energy_max_var.get() != '':
                self.rounded = self.round_energy(self.upper_bands, int(self.energy_max_list[i].get()))
                b_index = (np.where(self.upper_bands == float(self.rounded)))[0][0]
                self.energies_high.append(int(self.energy_max_list[i].get()))
                self.energies_high_index.append(int(b_index))

    @staticmethod
    def round_energy(liste, value):
        """Searches for the nearest number from the list. \n
        Parameters: \n
            liste: storage of researched values; \n
            value: number to round to match one of the given list."""
        near = 150
        rounded = int(value)
        for i in range(len(liste)):
            diff = np.abs(value - liste[i])
            if near > diff:
                near = diff
                rounded = int(liste[i])
        return rounded

    def __get_data(self, typ):
        """Defines the Rate, Counts, and Flux for "Plot Time Profile". \n
        Parameters: \n
            typ: type of data : rate, counts, or flux."""
        data = np.zeros((len(self.times), len(self.energies_low) + 1))
        for i in range(len(self.times)):
            for j in range(len(self.energies_low) + 1):
                if j == len(self.energies_low):
                    data[i, j] = self.times[i]

                else:
                    if np.where(self.lower_bands == self.energies_low[j])[0]:
                        a = (np.where(self.lower_bands == self.energies_low[j]))[0][0]
                    else:
                        self.rounded = self.round_energy(self.lower_bands, self.energies_low[j])
                        a = (np.where(self.lower_bands == self.rounded))[0][0]
                    self.energies_low_index.append(a + 1)

                    if np.where(self.upper_bands == self.energies_high[j])[0]:
                        b = (np.where(self.upper_bands == self.energies_high[j]))[0][0]
                    else:
                        self.rounded = self.round_energy(self.upper_bands, self.energies_high[j])
                        b = (np.where(self.upper_bands == self.rounded))[0][0]
                    self.energies_high_index.append(b + 1)
                    # Determines the energy distribution for different channels relative to the time of observed data

                    # Energy channels calculation in function of plot unit
                    if typ == 'rate':
                        data[i, j] = sum(self.counts[i, a:(b + 1)]) / self.del_times[i]
                    elif typ == 'counts':
                        data[i, j] = sum(self.counts[i, a:(b + 1)])
                    elif typ == 'flux':
                        self.e_diff = np.abs(self.energies_high[j] - self.energies_low[j])
                        data[i, j] = sum(self.counts[i, a:(b + 1)]) / self.del_times[i] / (self.area * self.e_diff)
                    else:
                        print("Error")
                        return None

        return data  # Returns data for energy bands that can be plot

    # def __time_profile_plotting(self, data, typ, show=True):
    #     """Plots the function of time by selected Unit. Uses the colormesh function provided by matplotlib library. \n
    #     Parameters:
    #         data: data to read; \n
    #         typ: type of data: 'rate', 'counts', or 'flux'; \n
    #         show: if True, displays the plot."""
    #     # plt.figure()
    #     self.columns_label = [str(low) + '-' + str(high) + 'keV' for low, high in
    #                           zip(self.energies_low, self.energies_high)]
    #     self.columns_label.append('Times')
    #     color = ['blue', 'red', 'green', 'black', 'orange']

    #     # Absciss data transformation
    #     self.label_time_plot = np.asarray(self.times) + self.start_time

    #     if self.entire_file:
    #         data_reduced = data
    #         times_reduced = self.times
    #     else:
    #         self.acq_time(self.label_time_plot)
    #         times_reduced = data[self.index_start:self.index_end + 1, -1]
    #         data_reduced = data[self.index_start:self.index_end + 1, :]

    #     # Plotting different figures according to chosen parameter
    #     if typ == "rate":
    #         # Choosing the specific color for each energy channel
    #         df = pd.DataFrame(data_reduced, index=times_reduced, columns=self.columns_label)
    #         plt.ylabel('Rate (Counts/s) by Bands')
    #         plt.title('Time Profile Plotting Rate (Counts/s)')

    #     elif typ == "counts":
    #         # Choosing the specific color for each energy channel
    #         df = pd.DataFrame(data_reduced, index=times_reduced,
    #                           columns=self.columns_label)
    #         plt.ylabel('Counts by Bands')
    #         plt.title('Time Profile Plotting Counts')

    #     elif typ == "flux":
    #         # Choosing the specific color for each energy channel
    #         df = pd.DataFrame(data_reduced, index=times_reduced,
    #                           columns=self.columns_label)
    #         plt.ylabel('Flux by Bands')
    #         plt.title('Time Profile Plotting Flux')

    #     else:
    #         df = None
    #         print("Error")

    #     ax_bis1 = plt.gca()
    #     for l_bands in range(len(self.energies_low)):
    #         df.plot(x='Times', y=self.columns_label[l_bands], color=color[l_bands], ax=ax_bis1)

    #     # Absciss legend plot
    #     if show:
    #         if self.entire_file:
    #             file_duration = max(self.times) - min(self.times)
    #         else:
    #             file_duration = self.times[self.index_end] - self.times[self.index_start]

    #         if file_duration <= 1800:  # file duration less than 30 minutes
    #             step_x = 120
    #         elif file_duration <= 3600:  # file duration less than 1 hour
    #             step_x = 480
    #         elif file_duration <= 28800:  # file duration less than 8 hours
    #             step_x = 3600
    #         else:
    #             step_x = 7200  # file duration more than 8 hours

    #         # Coloration
    #         if self.entire_file:
    #             x_positions = np.arange(0, file_duration, step_x)
    #         else:
    #             x_positions = np.arange(data[self.index_start, -1],
    #                                     file_duration + data[self.index_start, -1], step_x)
    #         x_positions_bis = x_positions + self.start_time  # pixel count at label position
    #         x_labels_plot = list(
    #             map(lambda x: (str(datetime.timedelta(seconds=float(x)))).split('.')[0][:-3], x_positions_bis))
    #         x_labels_plot_days = []
    #         for i in range(len(x_labels_plot)):
    #             if 'day' in x_labels_plot[i]:
    #                 x_labels_plot_days.append((x_labels_plot[i].split(','))[1])
    #             else:
    #                 x_labels_plot_days.append(x_labels_plot[i])
    #         # plt.xticks(x_positions, x_labels_plot_days)
    #         plt.xticks(x_positions, x_labels_plot_days, rotation=45)
    #         plt.gca().xaxis.set_major_locator(tck.MaxNLocator(nbins=10))


    #         # Scaling
    #         if self.scalex.get() == 'log':
    #             plt.xscale('log')
    #         if self.scaley.get() == 'log':
    #             plt.yscale('log')

    #         # Plotting and closing parameters window
    #         # plt.xlabel('Start time: ' + str(self.start_date) + ' -- End time : ' + str(
    #         #     self.end_date))  # load start time from header and display it in X - axis
    #         plt.xlabel(str(self.start_date) + '   ---   ' + str(self.end_date))
    #         self.toptime.destroy()
            
    #         if hasattr(self, 'grid_var2') and self.grid_var2.get():
    #             plt.grid(True)
    #         else:
    #             plt.grid(False)

    #         plt.show()


    def __time_profile_plotting(self, data, typ, show=True):
        """Plots the function of time by selected Unit."""
        self.columns_label = [str(low) + '-' + str(high) + 'keV' for low, high in zip(self.energies_low, self.energies_high)]
        self.columns_label.append('Times')
        color = ['blue', 'red', 'green', 'black', 'orange']

        # Convert instrument time indices (self.times) to real UTC datetimes.
        # The FITS file only stores arbitrary index values (e.g., 640 ... 2,190,010), not absolute times.
        # This formula linearly scales those indices between the actual start_datetime and end_datetime:
        #   - t_min maps to start_datetime
        #   - t_max maps to end_datetime
        # Each t is proportionally converted to its corresponding real datetime within the file's time range.
            

        # Conversion en datetime
        start_datetime = datetime.strptime(self.start_date_file, "%Y-%m-%dT%H:%M:%S.%f")
        end_datetime = datetime.strptime(self.end_date_file, "%Y-%m-%dT%H:%M:%S.%f")
        t_min = min(self.times)
        t_max = max(self.times)
        total_seconds = (end_datetime - start_datetime).total_seconds()

        if self.entire_file:
            data_reduced = data
            times_reduced = self.times
        else:
            if self.start_date < self.start_date_file or self.end_date > self.end_date_file:
                messagebox.showwarning("❌", "Start date or end date is outside the file's date range.")
                return
            elif self.start_date > self.end_date:
                messagebox.showwarning("❌", "Start date cannot be after end date.")
                return
            
            self.acq_time(np.asarray(self.times) + self.start_time)
            index_start = self.date_to_times_index(self.start_date)
            index_end = self.date_to_times_index(self.end_date)
            # print(f"Index start: {index_start}, Index end: {index_end}")  
            times_reduced = data[index_start:index_end + 1, -1]
            data_reduced = data[index_start:index_end + 1, :]
            print('index start:', index_start, 'index end:', index_end)
            print('times reduced:', times_reduced[index_end])

        times_datetime = [
            start_datetime + timedelta(seconds=(t - t_min) / (t_max - t_min) * total_seconds)
            for t in times_reduced
        ]
        print('last times :', times_datetime[-1])

        if len(times_datetime) != data_reduced.shape[0]:
            print("❌ Times and data size mismatch.")
            return


        # df = pd.DataFrame(data_reduced, index=times_datetime, columns=self.columns_label)

        if typ == "rate":
            plt.ylabel('Rate (Counts/s) by Bands')
            plt.title('Time Profile Plotting Rate (Counts/s)')
        elif typ == "counts":
            plt.ylabel('Counts by Bands')
            plt.title('Time Profile Plotting Counts')
        elif typ == "flux":
            plt.ylabel('Flux by Bands')
            plt.title('Time Profile Plotting Flux')
        else:
            print("Type inconnu")
            return

        # Affichage des courbes
        for l_bands in range(len(self.energies_low)):
            # df.plot(x='Times', y=self.columns_label[l_bands], color=color[l_bands], ax=plt.gca())
            plt.plot(times_datetime, data_reduced[:, l_bands], color=color[l_bands], label=self.columns_label[l_bands])


        ax = plt.gca()
        ax.set_xlim([times_datetime[0], times_datetime[-1]])  # Bien aligner le graphe
        ax.minorticks_on()  
        ax.grid(True, which='minor', linestyle=':', linewidth=0.1, alpha=0.1)

        # Format de temps automatique
        locator = AutoDateLocator()
        formatter = DateFormatter('%H:%M:%S' if total_seconds <= 86400 else '%m-%d\n%H:%M')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.xlabel(f"{self.start_date}   ---   {self.end_date}")
        if self.scalex.get() == 'log':
            plt.xscale('log')
        if self.scaley.get() == 'log':
            plt.yscale('log')

        if self.grid_var2.get():
            plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        else:
            plt.grid(False)

        # if hasattr(self, 'grid_var2') and self.grid_var2.get():
        #     # self.__update_grid_2()
        #     plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
        # else:
        #     # plt.minorticks_on()
        #     # plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        #     plt.grid(False)

        plt.xticks(rotation=45)
        if show:
            plt.show()
 
    def date_to_times_index(self, date_value):
        """
        Calcule la valeur de self.times approximative (via proportionnalité) pour une date donnée.
        """
        t_min = np.min(self.times)
        t_max = np.max(self.times)

        start_datetime = datetime.strptime(self.start_date_file, "%Y-%m-%dT%H:%M:%S.%f")
        end_datetime = datetime.strptime(self.end_date_file, "%Y-%m-%dT%H:%M:%S.%f")
        date_value = datetime.strptime(date_value, "%Y-%m-%dT%H:%M:%S.%f")

        total_seconds = (end_datetime - start_datetime).total_seconds()
        delta_seconds = (date_value - start_datetime).total_seconds()

        # Proportionnalité linéaire
        t_estimated = t_min + (delta_seconds / total_seconds) * (t_max - t_min)

        # Pour obtenir un index le plus proche dans self.times
        index = (np.abs(self.times - t_estimated)).argmin()

        return index

    
    def time_index(self, index_times, value_start, value_end):
        """Finds the index of start time corresponding to the chosen data. \n
        Parameters: \n
            index_times: list of indexes for time range; \n
            value_start: starting value of time range; \n
            value_end: ending value of time range."""
        for i in range(len(index_times)):
            if index_times[i] >= value_start:
                self.index_start_list.append(i)
            if index_times[i] <= value_end:
                self.index_end_list.append(i)
        return self.index_start_list[0], self.index_end_list[-1]

    # =================== 2. Spectrum plotting ===================

    def plot_spectrum_rate(self):
        """If the selected unit is rate, passes the string 'rate' to the spectrum setting. \n
        Dependencies: second.py"""
        self.type = 'rate'
        self.__plot_spectrum()

    def plot_spectrum_counts(self):
        """If the selected unit is counts, passes the string 'counts' to the spectrum setting. \n
        Dependencies: second.py"""
        self.type = 'counts'
        self.__plot_spectrum()

    def plot_spectrum_flux(self):
        """If the selected unit is flux, passes the string 'flux' to the spectrum setting. \n
        Dependencies: second.py"""
        self.type = 'flux'
        self.__plot_spectrum()

    def __plot_spectrum(self):
        """Preparing figure plotting with the data. \n
        Parameters: \n
            typ: type of data: rate, counts, or flux."""
        if self.show:
            plt.figure()
        data = np.zeros(self.energies_bin)
        # print('energies_bin:', self.energies_bin)
        # print('data shape:', data.shape)
        self.label_time_plot_spectro = np.asarray(self.times) + self.start_time

        if not self.entire_file:
            self.acq_time(self.label_time_plot_spectro)

        if self.type == 'rate':
            self.data_plot = self.convert_counts_rate()
            for i in range(self.energies_bin):  # for each channel
                # Determines Rate for "Plot Spectrum"
                if self.entire_file:
                    data[i] = np.mean(self.data_plot[:, i])
                else:
                    data[i] = np.mean(self.data_plot[self.index_start:self.index_end + 1, i])
            ylabel = 'counts/s'
            title = 'STIX SOLAR Range vs Energy'

        elif self.type == 'counts':
            for i in range(self.energies_bin):  # for each channel
                # Determines Counts for "Plot Spectrum"
                if self.entire_file:
                    data[i] = np.mean(self.counts[:, i])
                else:
                    data[i] = np.mean(self.counts[self.index_start:self.index_end + 1, i])
            ylabel = 'counts'
            title = 'STIX SOLAR Counts vs Energy'

        elif self.type == 'flux':
            self.data_plot = self.convert_counts_flux()
            for i in range(self.energies_bin):  # for each channel
                # Determines Flux for "Plot Spectrum"
                if self.entire_file:
                    data[i] = np.mean(self.data_plot[:, i]) / self.area
                else:
                    data[i] = np.mean(self.data_plot[self.index_start:self.index_end + 1, i])
            ylabel = 'counts/s/mm²/keV'
            title = 'STIX SOLAR Flux vs Energy'

        else:
            ylabel = 'Unknown unit'
            title = 'STIX SOLAR ??? vs Energy'
            print('Unit not found')

        if not self.show:
            return data

        # print('data shape:', data.shape)
        # print('lower bands shape:', self.lower_bands.shape)
        self.win_log_spec()
        plt.rcParams["figure.figsize"] = [6, 6]
        plt.title(title)
        plt.plot(self.lower_bands, data, drawstyle='steps-post')  # Unit vs Energy
        # plt.xlabel('Energy(keV) / Start time: ' + str(self.start_date) + ' -- End time : ' + str(self.end_date))
        plt.xlabel(str(self.start_date) + '    ---    ' + str(self.end_date))
        plt.ylabel(ylabel)
        plt.xlim(left=self.lower_bands[0])


        # If the grid_var is defined and set to True, show the grid otherwise hide it
        if hasattr(self, 'grid_var') and self.grid_var.get():
            plt.grid(True)
        else:
            plt.grid(False)
            

        
    # def __plot_spectrum(self):
    #     """Preparing figure plotting with the data."""
    #     if self.show:
    #         plt.figure()
        
    #     # Initialize data array with correct shape (32,) instead of (1,)
    #     data = np.zeros(self.energies_bin) if isinstance(self.energies_bin, int) else np.zeros(len(self.energies_bin))
    #     self.label_time_plot_spectro = np.asarray(self.times) + self.start_time

    #     if not self.entire_file:
    #         self.acq_time(self.label_time_plot_spectro)

    #     if self.type == 'rate':
    #         self.data_plot = self.convert_counts_rate()
    #         for i in range(len(data)):  # for each channel
    #             # Determines Rate for "Plot Spectrum"
    #             if self.entire_file:
    #                 data[i] = np.mean(self.data_plot[:, i])
    #             else:
    #                 data[i] = np.mean(self.data_plot[self.index_start:self.index_end + 1, i])
    #         ylabel = 'counts/s'
    #         title = 'STIX SOLAR Range vs Energy'

    #     elif self.type == 'counts':
    #         for i in range(len(data)):  # for each channel
    #             # Determines Counts for "Plot Spectrum"
    #             if self.entire_file:
    #                 data[i] = np.mean(self.counts[:, i])
    #             else:
    #                 data[i] = np.mean(self.counts[self.index_start:self.index_end + 1, i])
    #         ylabel = 'counts'
    #         title = 'STIX SOLAR Counts vs Energy'

    #     elif self.type == 'flux':
    #         self.data_plot = self.convert_counts_flux()
    #         for i in range(len(data)):  # for each channel
    #             # Determines Flux for "Plot Spectrum"
    #             if self.entire_file:
    #                 data[i] = np.mean(self.data_plot[:, i]) / self.area
    #             else:
    #                 data[i] = np.mean(self.data_plot[self.index_start:self.index_end + 1, i])
    #         ylabel = 'counts/s/mm²/keV'
    #         title = 'STIX SOLAR Flux vs Energy'

    #     else:
    #         ylabel = 'Unknown unit'
    #         title = 'STIX SOLAR ??? vs Energy'
    #         print('Unit not found')

    #     if not self.show:
    #         return data

    #     print('data shape:', data.shape)
    #     print('lower bands shape:', self.lower_bands.shape)
        
    #     # Ensure shapes match before plotting
    #     if len(data) != len(self.lower_bands):
    #         data = data.reshape(-1)  # Flatten the data array if needed
        
    #     self.win_log_spec()
    #     plt.rcParams["figure.figsize"] = [6, 6]
    #     plt.title(title)
    #     plt.plot(self.lower_bands, data, drawstyle='steps-post')  # Unit vs Energy
    #     plt.xlabel('Energy(keV) / Start time: ' + str(self.start_date) + ' -- End time : ' + str(self.end_date))
    #     plt.ylabel(ylabel)

    #     # If the grid_var is defined and set to True, show the grid otherwise hide it
    #     if hasattr(self, 'grid_var') and self.grid_var.get():
    #         plt.grid(True)
    #     else:
    #         plt.grid(False)


    def __update_grid(self, *args):
        """Updates the grid display when the checkbox is checked or unchecked."""
        try:
            ax = plt.gca()
            show_grid = self.grid_var.get()

            ax.grid(show_grid, which='major')  # principal grid

            ax.minorticks_on()  # Active ticks secondaires
            ax.grid(show_grid, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

            plt.draw()
        except Exception as e:
            print("Grille non modifiable :", e)

    # def __update_grid_2(self, *args):
    #     """Updates the grid display when the checkbox is checked or unchecked."""
    #     try:
    #         ax = plt.gca()
    #         show_grid = self.grid_var2.get()

    #         ax.grid(show_grid, which='major')

    #         ax.minorticks_on() 
    #         ax.grid(show_grid, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

    #         plt.draw()
    #     except Exception as e:
    #         print("Grille non modifiable :", e)




    def win_log_spec(self):
        """Creates a new window 'window_spec_limits' to choose the linear or logarithmic axis for spectrum plotting.
        Calls for the log_axis function."""

        self.window_spec_limits = Toplevel()
        self.window_spec_limits.title('Scales for axis')
        self.window_spec_limits.geometry("400x300")
        self.text_spec_limits = Label(self.window_spec_limits, text="Choose your scale for both x and y axis.",
                                      fg='black', font=("Times", 11))
        self.text_spec_limits.place(relx=0.5, rely=0.09, anchor='center')
        self.log_axis(self.window_spec_limits, 0.10, 0.30)

        # show grid
        self.grid_var = BooleanVar()
        self.grid_check = Checkbutton(self.window_spec_limits, text="Show grid", variable=self.grid_var)
        self.grid_check.place(relx=0.5, rely=0.65, anchor='center')
        self.grid_var.trace_add('write', self.__update_grid)

        self.btn_plot_spectrum = Button(self.window_spec_limits, text="Plot spectrum", command=self.__plot_show)
        self.btn_plot_spectrum.place(relx=0.5, rely=0.80, anchor=S, relheight=0.11, relwidth=0.25)

    def __plot_show(self):
        """Closes the previous window and shows the plot for the spectrum."""
        if self.scalex.get() == 'log':
            plt.xscale('log')
        if self.scaley.get() == 'log':
            plt.yscale('log')
        self.window_spec_limits.destroy()
        plt.show()

    # =================== 3. Spectrogram Plotting ===================

    def plot_spectrogram_rate(self):
        """If the selected unit is rate, passes the string 'rate' to the spectrogram setting. \n
        Dependencies: second.py"""
        self.__plot_spectrogram('rate')

    def plot_spectrogram_counts(self):
        """If the selected unit is counts, passes the string 'counts' to the spectrogram setting. \n
        Dependencies: second.py"""
        self.__plot_spectrogram('counts')

    def plot_spectrogram_flux(self):
        """If the selected unit is flux, passes the string 'flux' to the spectrogram setting. \n
        Dependencies: second.py"""
        self.__plot_spectrogram('flux')

    def __plot_spectrogram(self, typ):
        """The spectrogram is a function of Rate/Counts/Flux as a function of energy and time. x = tick(Time in h:m:s)
        and y(Energy bounds) are bounds ; z is the value *inside* those bounds (Rate/Counts/Flux). \n
        Parameters: \n
            typ: data type: rate, counts, or flux."""
        # pcolormesh function(below) does not work with pandas time conversion function(TimeNew), we have to rewrite it.
        # Convert instrument time indices (self.times) to real UTC datetimes.
        # The FITS file only stores arbitrary index values (e.g., 640 ... 2,190,010), not absolute times.
        # This formula linearly scales those indices between the actual start_datetime and end_datetime:
        #   - t_min maps to start_datetime
        #   - t_max maps to end_datetime
        # Each t is proportionally converted to its corresponding real datetime within the file's time range.


        start_datetime = datetime.strptime(self.start_date_file, "%Y-%m-%dT%H:%M:%S.%f")
        end_datetime = datetime.strptime(self.end_date_file, "%Y-%m-%dT%H:%M:%S.%f")
        total_seconds = (end_datetime - start_datetime).total_seconds()

        t_min = min(self.times)
        t_max = max(self.times)

        self.times_datetime = [
            start_datetime + timedelta(seconds=(t - t_min) / (t_max - t_min) * total_seconds)
            for t in self.times
        ]

        # Absciss data transformation
        self.label_time_plot_spectro = np.asarray(self.times) + self.start_time

        if self.entire_file:
            self.times_sequences = self.times_datetime
        else:
            if self.start_date < self.start_date_file or self.end_date > self.end_date_file:
                messagebox.showwarning("❌", "Start date or end date is outside the file's date range.")
                return
            elif self.start_date > self.end_date:
                messagebox.showwarning("❌", "Start date cannot be after end date.")
                return
            
            self.acq_time(self.label_time_plot_spectro)
            self.index_chosen = self.date_to_times_index(self.start_date)
            self.end_chosen = self.date_to_times_index(self.end_date)
            self.times_sequences = self.times_datetime[self.index_chosen:self.end_chosen + 1]


        fig, self.ax = plt.subplots(1, 1, figsize=(15, 5), sharey="all", facecolor='w')
        fig.canvas.draw()

        # Limits and display
        self.specgm_lim()

        # Plotting rate
        if typ == 'rate':
            self.data_plot = self.convert_counts_rate()
            # self.data_plot =  np.where(self.data_plot != 0, self.data_plot, 1e-5)
            if self.entire_file:
                plt.pcolormesh(self.times_datetime, self.lower_bands, np.log10(np.transpose(self.data_plot)),
                               shading='auto', cmap='coolwarm')
            else:
                data_sequences = self.data_plot[self.index_chosen:self.end_chosen + 1, :]
                plt.pcolormesh(self.times_sequences, self.lower_bands, np.log10(np.transpose(data_sequences)),
                               shading='auto', cmap='coolwarm')
            plt.title('STIX SOLAR Rate Spectrogram')
            self.spgm_label = 'Counts/sec'

        # Plotting counts
        elif typ == 'counts':
            if self.entire_file:
                plt.pcolormesh(self.times_datetime, self.lower_bands, np.log10(np.transpose(self.counts)),
                               shading='auto', cmap='coolwarm')
            else:
                data_sequences = self.counts[self.index_chosen:self.end_chosen + 1, :]
                plt.pcolormesh(self.times_sequences, self.lower_bands, np.log10(np.transpose(data_sequences)),
                               shading='auto', cmap='coolwarm')
            plt.title('STIX SOLAR Counts Spectrogram')
            self.spgm_label = 'Counts'

        # Plotting flux
        elif typ == 'flux':
            self.data_plot = self.convert_counts_flux()
            if self.entire_file:
                plt.pcolormesh(self.times_datetime, self.lower_bands, np.log10(np.transpose(self.data_plot)),
                               shading='auto', cmap='coolwarm')
            else:
                data_sequences = self.data_plot[self.index_chosen:self.end_chosen + 1, :]
                plt.pcolormesh(self.times_sequences, self.lower_bands, np.log10(np.transpose(data_sequences)),
                               shading='auto', cmap='coolwarm')
            plt.title('STIX SOLAR Flux Spectrogram')
            self.spgm_label = 'counts/s/mm²/keV'

        else:
            print('error')

        # # Defining x step
        # file_duration = max(self.times_sequences) - min(self.times_sequences)
        # if file_duration <= 1800:  # file duration less than 30 minutes
        #     step_x = 120
        # elif file_duration <= 3600:  # file duration less than 1 hour
        #     step_x = 480
        # elif file_duration <= 28800:  # file duration less than 8 hours
        #     step_x = 3600
        # else:  # file duration more than 8 hours
        #     step_x = 7200

        # # x axis construction
        # x_positions = np.arange(0, file_duration, step_x)

        # if self.entire_file:
        #     x_positions_bis = x_positions + self.start_time  # pixel count at label position
        #     x_labels_plot = list(
        #         map(lambda x: (str(timedelta(seconds=float(x)))).split('.')[0][:-3], x_positions_bis))
        # else:
        #     x_positions_bis = np.asarray(x_positions) + self.times[self.index_start]
        #     x_positions_bis_label = np.asarray(x_positions) + self.label_time_plot_spectro[self.index_start]
        #     x_labels_plot = list(
        #         map(lambda x: (str(timedelta(seconds=float(x)))).split('.')[0][:-3], x_positions_bis_label))

        # x_labels_plot_days = []
        # for i in range(len(x_labels_plot)):
        #     if 'day' in x_labels_plot[i]:
        #         x_labels_plot_days.append((x_labels_plot[i].split(','))[1])
        #     else:
        #         x_labels_plot_days.append(x_labels_plot[i])
        # if self.entire_file:
        #     plt.xticks(x_positions, x_labels_plot_days)
        # else:
        #     plt.xticks(x_positions_bis, x_labels_plot_days)

        # # Ordinate legend plot
        # if self.entire_file:
        #     self.energy_idx = self.lower_bands
        #     self.ene_leg = [str(a) + '-' + str(b) for a, b in zip(self.lower_bands, self.upper_bands)]
        #     self.ene_leg = np.array(self.ene_leg)

        # Axe X dynamique propre (datetime)
        ax = plt.gca()

        # ax.grid(False, which='major')  # principal grid

        ax.minorticks_on()  # Active ticks secondaires
        # ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.1, alpha=0.1)

        # plt.draw()

        locator = AutoDateLocator()
        formatter = DateFormatter('%H:%M:%S' if total_seconds <= 86400 else '%m-%d\n%H:%M')

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Aligner le graphe au début
        if self.entire_file:
            ax.set_xlim(self.times_datetime[0], self.times_datetime[-1])
        else:
            ax.set_xlim(self.times_sequences[0], self.times_sequences[-1])

        plt.gcf().autofmt_xdate()


    def convert_counts_rate(self):
        """Converts counts data to rate data. \n
        Rate = Counts / Time"""
        (self.ligne, self.colonne) = np.shape(self.counts)
        self.data_plot = np.zeros((self.ligne, self.colonne))
        for i in range(self.ligne):
            for j in range(self.colonne):
                self.data_plot[i][j] = self.counts[i][j] / self.del_times[i]
        return self.data_plot

    def convert_counts_flux(self):
        """Converts counts data to flux data. \n
        Flux = Rate / (Area * Bands length) = Counts / (Time * Area * Bands length)"""
        (self.ligne, self.colonne) = np.shape(self.counts)
        self.data_plot = np.zeros((self.ligne, self.colonne))
        delta_e = np.zeros(shape=(len(self.lower_bands)))
        for i in range(self.ligne):
            for j in range(self.colonne):
                delta_e[j] = self.upper_bands[j] - self.lower_bands[j]  # Difference between bands
                self.data_plot[i][j] = self.counts[i][j] / self.del_times[i] / self.area / delta_e[j]
        return self.data_plot

    def specgm_lim(self):
        """Displays a window to allow user to change plot scale axis for spectrogram plotting using OptionMenus."""
        # Créer la fenêtre
        self.window_elimits = Toplevel()
        self.window_elimits.title('Energy limits')
        self.window_elimits.geometry("400x300")

        # Texte d'instruction
        self.text_energy_elimit = Label(
            self.window_elimits,
            text="Choose the minimum and maximum energy to plot the spectrogram.",
            fg='black',
            font=("Times", 11)
        )
        self.text_energy_elimit.place(relx=0.5, rely=0.09, anchor='center')


        # Créer les listes de valeurs d'énergie 
        e_low_values = sorted(set(self.energies['e_low'])) 
        e_high_values = sorted(set(self.energies['e_high'])) 

        # Filtrer les valeurs infinies et NaN dans e_high_values
        e_high_values = [e for e in e_high_values 
                if not np.isinf(e) and not np.isnan(e)]    

        # Convertir les valeurs en entiers
        e_low_values_int = [int(e) for e in e_low_values if e != 0]
        e_high_values_int = [int(e) for e in e_high_values]

        # Créer les variables StringVar pour les OptionMenu
        self.energy_min_var = IntVar()
        self.energy_max_var = IntVar()

        # Variables IntVar pour OptionMenu
        self.ylim_min_var = IntVar()
        self.ylim_max_var = IntVar()
        self.ylim_min_var.set(min(e_low_values_int))
        self.ylim_max_var.set(max(e_high_values_int))

        # OptionMenus
        self.e_ylim_min = OptionMenu(self.window_elimits, self.ylim_min_var, *e_low_values_int)
        self.e_ylim_min.place(relx=0.5, rely=0.22, relheight=0.10, relwidth=0.25, anchor='center')

        self.e_ylim_max = OptionMenu(self.window_elimits, self.ylim_max_var, *e_high_values_int)
        self.e_ylim_max.place(relx=0.5, rely=0.37, relheight=0.10, relwidth=0.25, anchor='center')

        # Scaling (log/linear checkbox ou autre, selon ta fonction log_axis)
        self.log_axis(self.window_elimits, 0.10, 0.50)

        # Bouton pour lancer le tracé du spectrogramme
        self.btn_plot_spgm = Button(self.window_elimits, text="Plot spectrogram", command=self.show_specgm)
        self.btn_plot_spgm.place(relx=0.5, rely=0.85, anchor=N)


    @staticmethod
    def colorbar_scale(plot, pos):
        """Formats the colorbar plotted for spectrogram, displaying units as 10^x. \n
        Parameters: \n
            plot: traced spectrogram data; \n
            pos: argument necessary for the FuncFormatter from Tkinter; not of use in this function."""
        decimal, power = '{:.2e}'.format(plot).split('e')
        raised_power = np.floor(float(decimal) * 10 ** (1 + float(power))) / 10
        return r'$10^{{{}}}$'.format(raised_power)

    def show_specgm(self):
        """Adds all chosen values to the plot and shows the figure."""
        # Limits
        # if self.e_ylim_min.get().isdigit():
        #     self.ylim_min = int(self.e_ylim_min.get())
        # else:
        #     self.ylim_min = 1  # Default min value
        # if self.e_ylim_max.get().isdigit():
        #     self.ylim_max = int(self.e_ylim_max.get())
        # else:
        #     self.ylim_max = 150  # Default max value

        # Scaling
        if self.scalex.get() == 'log':
            plt.xscale('log')
        if self.scaley.get() == 'log':
            plt.yscale('log')

        # Plotting
        self.window_elimits.destroy()
        plt.ylim(self.ylim_min_var.get(), self.ylim_max_var.get())
        plt.ylabel('Energy (keV)')
        plt.xlabel('Time(UT)')
        # plt.xlabel('Start time: ' + str(self.start_date) + ' -- End time : ' + str(self.end_date))
        plt.xlabel(str(self.start_date) + '   ---   ' + str(self.end_date))
        plt.colorbar(ax=self.ax, label=self.spgm_label, format=tck.FuncFormatter(self.colorbar_scale))
        plt.show()


# Demo if file is run
if __name__ == '__main__':

    plots = Input(background.BackgroundWindow.fname)    # any input file with .fits extension

    plots.rate_vs_time_plotting()                       # plot Count Rate vs Time
    plots.counts_vs_time_plotting()                     # plot Counts vs Time
    plots.flux_vs_time_plotting()                       # plot Count Flux vs Time
    plots.plot_spectrum_rate()                          # plot Count Rate vs Energy
    plots.plot_spectrum_counts()                        # plot Counts vs Energy
    plots.plot_spectrum_flux()                          # plot Flux vs Energy
    plots.plot_spectrogram_rate()                       # plot Count Rate Spectrogram
    plots.plot_spectrogram_counts()                     # plot Counts Spectrogram
    plots.plot_spectrogram_flux()                       # plot Flux Spectrogram
