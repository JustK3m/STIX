from tkinter import *
from tkinter.filedialog import askopenfilename
from astropy.io import fits
from astropy.table import Table
from .graphics import plotting
import os
import sys

from .io import loader

_PLOT_DISPATCH = {
    ('Rate',   'time'):   'rate_vs_time_plotting',
    ('Rate',   'spec'):   'plot_spectrum_rate',
    ('Rate',   'specgr'): 'plot_spectrogram_rate',
    ('Counts', 'time'):   'counts_vs_time_plotting',
    ('Counts', 'spec'):   'plot_spectrum_counts',
    ('Counts', 'specgr'): 'plot_spectrogram_counts',
    ('Flux',   'time'):   'flux_vs_time_plotting',
    ('Flux',   'spec'):   'plot_spectrum_flux',
    ('Flux',   'specgr'): 'plot_spectrogram_flux',
}

class InputWindow:
    """Class to create "Select Plotting Window" menu option in STIX.
    Rout: Go to top menu bars -> File -> Select Plotting.
    Initiates the parameters and widgets"""


    def __init__(self, root):
        """Creating a new window for 'Select Plotting' part, using the 'place' geometry manager, allowing to explicitly
        set the position and dimensions of window. The place manager can be accessed through the place method.
        It can be applied to all standard widgets."""
        self.headers = None
        self.data = None
        self.e_high = None
        self.e_low = None
        self.top1 = Toplevel()
        self.top1.title('STIX Plotting Options')
        self.top1.geometry("1000x400")
        Label(self.top1, text="Select Plotting", fg="red", font="Helvetica 12 bold italic").place(relx=0.5, rely=0.01,
                                                                                                  anchor=N)
        # Creates a window to inform user hdul is empty
        self.hdul = None
        self.root = root

        # 'Select input' window should contain 2 frames:
        # 1) Widgets to load the data, read the content of .fits file extensions, etc;
        # 2) Widgets to  select the energy bands and time bands intervals, and plot.

        # ==================== First frame and additional widgets ====================

        # The name of the widget is in the 'text' parameter
        self.frame1 = LabelFrame(self.top1, relief=RAISED, borderwidth=1)
        self.frame1.place(relx=0.5, rely=0.08, relheight=0.60, relwidth=0.9, anchor=N)

        self.name = loader.activeFile()

        Label(self.frame1, text="Spectrum or Image File: ").place(relx=0.01, rely=0.2)

        self.textFilename = Entry(self.frame1, width=20)
        self.textFilename.place(relx=0.2, rely=0.18, relheight=0.16, relwidth=0.57)

        if self.name is not None:
            self.textFilename.insert(0, self.name)
            self.open_file(self.name)
        else:
            self.textFilename.insert(0, "No file chosen")

        self.browseButton = Button(self.frame1, text='Browse ->', command=self.open_file)
        self.browseButton.place(relx=0.8, rely=0.186)

        self.chkBtEntireFile = Checkbutton(self.frame1, text="Entire file", command=self.checked)
        self.chkBtEntireFile.place(relx=0.03, rely=0.37)
        self.chkBtEntireFile.select()

        # Open file
        self.evt_data = Table()
        self.energies = Table()
        self.data_file = None
        self.time_summarize = Table()
        self.energie_summarize = Table()
        self.detector_summarize = Table()

        # Select Start data and End data for plotting if different of file's information(start-end)
        self.SetFromButton = Label(self.frame1, text="Set from -> ", state=DISABLED)
        self.SetFromButton.place(relx=0.1, rely=0.55)

        # Change Button to Label and insert the date data
        self.Times_range = []  # value for Times ranges

        self.start_date_var = StringVar()
        self.end_date_var = StringVar()
        self.start_time_plotting = StringVar()
        self.end_time_plotting = StringVar()

        self.StartButton = Label(self.frame1, text="Start", state=DISABLED)
        self.StartButton.place(relx=0.24, rely=0.55)

        self.textStart = Entry(self.frame1, textvariable=self.start_date_var, width=20)
        self.textStart.place(relx=0.31, rely=0.55, height=33, width=190)
        self.textStart['state'] = 'disabled'

        self.EndButton = Label(self.frame1, text="End", state=DISABLED)
        self.EndButton.place(relx=0.53, rely=0.55)

        self.textEnd = Entry(self.frame1, textvariable=self.end_date_var, width=20)
        self.textEnd.place(relx=0.59, rely=0.55, height=33, width=190)
        self.textEnd['state'] = 'disabled'

        # "Summarize" button. When clicked, creates a new window with the name "SPEX::PREVIEW,
        # then gives information from self.hdul[1].header.
        Button(self.frame1, text="Summarize", command=self.summarize).place(relx=0.45, rely=0.81, anchor=NE)

        # "Show Header" button. When clicked, gives the information from primary_header = hdulist[0].header.
        Button(self.frame1, text="Show Header", command=self.ShowHeader).place(relx=0.55, rely=0.81, anchor=NW)

        self.header_primary = None

        # ==================== Second frame and plotting section ====================

        self.frame2 = LabelFrame(self.top1, relief=RAISED, borderwidth=2)
        self.frame2.place(relx=0.5, rely=0.7, relheight=0.15, relwidth=0.9, anchor=N)

        # Plot Units for Rate, Counts, and Flux
        Label(self.frame2, text="Plot Units: ").place(relx=0.013, rely=0.5, anchor=W)

        # Option menu for Rate, Counts, Flux
        self.Component_choices = ('Rate', 'Counts', 'Flux')
        self.var = StringVar(self.frame1)
        self.var.set(self.Component_choices[0])
        self.var.set(self.Component_choices[0])
        self.selection = OptionMenu(self.frame2, self.var, *self.Component_choices)
        self.selection.place(relx=0.11, rely=0.5, anchor=W)

        Button(self.frame2, text="Plot Spectrum",
                                         command=lambda: self.show_plot("spec")).place(relx=0.31, rely=0.5, anchor=W)

        Button(self.frame2, text="Plot Time Profile",
                                            command=lambda: self.show_plot("time")).place(relx=0.457, rely=0.5, anchor=W)

        Button(self.frame2, text="Plot Spectrogram",
                                            command=lambda: self.show_plot("specgr")).place(relx=0.62, rely=0.5, anchor=W)

        Button(self.top1, text="Close", command=self.destroy).place(relx=0.5, rely=0.9, anchor=N)

    # ============================ Main methods ============================

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
        self.textFilename.delete(0, 'end')

        if self.name:
            with fits.open(self.name) as hdul:
                self.hdul = hdul
            self.textFilename.insert(0, self.name)  # Displays the input file name in Entry box

            # Loads data to do the plot
            self.data = loader.get_data(self.name)

            self.evt_data = self.data['counts']
            self.e_low = self.data['e_low']
            self.e_high = self.data['e_high']

            self.headers = loader.get_header(self.name)
            self.time_summarize = [self.data['time'][-1], self.headers.get('DATE_BEG', self.headers.get('DATE-BEG', 'Unknown')),
                                   self.headers.get('DATE_END', self.headers.get('DATE-END', 'Unknown'))]  # time data

            self.textFilename.insert(0, self.name)  # Displays the input file name in Entry box
        else:
            self.textFilename.insert(0, "No file chosen")

    def summarize(self):
        """Creating a new window for Summarize button.
        Reads the information from Header and Data and display it in new window"""
        top = Toplevel()
        top.title('SPEX::PREVIEW')
        top.geometry("400x300")
        frameSummarize = LabelFrame(top, relief=RAISED, borderwidth=2)
        frameSummarize.pack(side=TOP, expand=True)

        # Loads headers informations
        self.data_file = self.headers['INSTRUME']  # data type
        self.time_summarize = [self.data['time'][-1], self.headers.get('DATE_BEG', self.headers.get('DATE-BEG', 'Unknown')),
                               self.headers.get('DATE_END', self.headers.get('DATE-END', 'Unknown'))]  # time data
        self.energie_summarize = [self.data['counts'].shape[1],
                                  max(min(self.data['e_low']), min(self.data['e_high'])),
                                  min(max(self.data['e_low']),
                                      max(self.data['e_high']))]  # energies data
        self.detector_summarize = ["No information found."]

        txt = ["\n\n\nSpectrum or Image File Summary",
               "\nData Type: ", self.data_file,
               "\nFile name: ", self.name,
               "\n#Time Bins: ", self.time_summarize[0], "\nTime range: ", self.time_summarize[1], ' to ',
               self.time_summarize[2],
               "\n#Energy Bins: ", self.energie_summarize[0],
               "\nEnergy range: ", self.energie_summarize[1], ' to ', self.energie_summarize[2],
               "\nDetectors Used: ", self.detector_summarize[0]]
        #    ,"\nResponse Info: ", self.name]
        liste = Text(frameSummarize)
        for t in txt:
            liste.insert(END, t)
        liste.pack()

    # FIXME: Text should be well-organized in the window 'SPEX::FITSHEADER'
    def ShowHeader(self):
        """Reads text information from header and display it in new window."""
        self.header_primary = self.hdul[0].header
        top = Toplevel()
        top.title('SPEX::FITSHEADER')
        top.geometry("500x700")
        scrollbar1 = Scrollbar(top)
        scrollbar1.pack(side=RIGHT, fill=Y)
        header = Text(top, width=450, height=450)

        for i in range(len(self.header_primary.cards)):
            header.insert(INSERT, self.header_primary.cards[i])
            header.insert(INSERT, "\n")

        header.insert(END, "Fin")
        header.config(state=DISABLED)
        header.pack()
        scrollbar1.config(command=header.yview)

    def destroy(self):
        """Closing 'Select Plotting' window, clicking 'Close' button."""
        self.top1.destroy()

    def submit(self):
        """Submit the value of start and end times for plotting with the date."""
        self.Times_range = []
        self.start_time_plotting = self.start_date_var.get()
        self.end_time_plotting = self.end_date_var.get()

        # Creation of value of editInterval_times
        self.Times_range.append(self.start_time_plotting)
        self.Times_range.append(self.end_time_plotting)

    def checked(self):
        """Toggle the 'Set From' date-range entries on or off."""
        entries = (self.textStart, self.textEnd)
        is_off = self.SetFromButton['state'] == 'disabled'

        new_state = 'normal' if is_off else 'disabled'
        self.SetFromButton['state'] = new_state
        for entry in entries:
            entry['state'] = new_state
            entry.delete(0, 'end')

        if is_off:
            self.textStart.insert(0, self.time_summarize[1])
            self.textEnd.insert(0, self.time_summarize[2])

    # ==================== Calling for plotting.py ====================

    def _build_plot_instance(self):
        """Instantiate a Plotting object from the currently loaded file."""

        if self.SetFromButton['state'] == 'disabled':
            return plotting.Plotting(data=self.data, headers=self.headers)

        self.submit()
        return plotting.Plotting(
            self.Times_range[0], self.Times_range[1],
            self.time_summarize[1], self.data, self.headers,
        )

    def show_plot(self, plot_type):
        """Dispatch to the appropriate Plotting method.

        Parameters
        ----------
        plot_type : {'time', 'spec', 'specgr'}
        """
        plots = self._build_plot_instance()
        if plots is None:
            print('Error: could not build plot instance.')
            return

        unit = self.var.get()
        method = _PLOT_DISPATCH.get((unit, plot_type))
        if method is None:
            print(f"No plot handler for unit={unit!r}, type={plot_type!r}")
            return

        getattr(plots, method)()