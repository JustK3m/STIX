from tkinter import *
from tkinter.filedialog import askopenfilename
from astropy.io import fits
from astropy.table import Table
import plotting
import background
import os
import sys


class SecondWindow:
    """Class to create "Select Input Window" menu option in STIX.
    Rout: Go to top menu bars -> File -> Select Input.
    Initiates the parameters and widgets"""

    # fname = 'solo_L1A_stix-sci-spectrogram' \
    #         '-2102140001_20210214T014006-20210214T015515_008648_V01.fits'
    # rname = 'stx_srm_2021feb14_0140_0155.fits'

    fname_r = 'data/solo_L1_stix-sci-xray-spec_20230319T175504-20230320T000014_V02_2303197888-65462.fits'

    def resource_path(relative_path):
        """Renvoie le chemin absolu même si l'app est congelée avec PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
    
    fname = resource_path(fname_r) 

    def __init__(self, root):
        """Creating a new window for 'Select Input' part, using the 'place' geometry manager, allowing to explicitly
        set the position and dimensions of window. The place manager can be accessed through the place method.
        It can be applied to all standard widgets."""
        self.top1 = Toplevel()
        self.top1.title('STIX Input Options')
        self.top1.geometry("1000x400")
        Label(self.top1, text="Select Input", fg="red", font="Helvetica 12 bold italic").place(relx=0.5, rely=0.01,
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

        self.name = None

        self.lblFilename = Label(self.frame1, text="Spectrum or Image File: ")
        self.lblFilename.place(relx=0.01, rely=0.2)

        self.textFilename = Entry(self.frame1, width=20)
        self.textFilename.place(relx=0.2, rely=0.18, relheight=0.16, relwidth=0.57)

        # if background.BackgroundWindow.fname:
        #     self.textFilename.insert(0, background.BackgroundWindow.fname)
        #     self.open_file(background.BackgroundWindow.fname)
        # else:
        #     self.textFilename.insert(0, "No file chosen")
        if SecondWindow.fname:
            self.textFilename.insert(0, SecondWindow.fname)
            self.open_file(SecondWindow.fname)
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
        self.SummarizeButton = Button(self.frame1, text="Summarize", command=self.summarize)
        self.SummarizeButton.place(relx=0.45, rely=0.81, anchor=NE)

        # "Show Header" button. When clicked, gives the information from primary_header = hdulist[0].header.
        self.ShowHeaderButton = Button(self.frame1, text="Show Header", command=self.ShowHeader)
        self.ShowHeaderButton.place(relx=0.55, rely=0.81, anchor=NW)

        self.header_primary = None

        self.frame2 = LabelFrame(self.top1, relief=RAISED, borderwidth=2)

        # ==================== Second frame and plotting section ====================

        self.frame2.place(relx=0.5, rely=0.7, relheight=0.15, relwidth=0.9, anchor=N)

        # Plot Units for Rate, Counts, and Flux
        self.lblPlotUnits = Label(self.frame2, text="Plot Units: ")
        self.lblPlotUnits.place(relx=0.013, rely=0.5, anchor=W)

        # Option menu for Rate, Counts, Flux
        self.Component_choices = ('Rate', 'Counts', 'Flux')
        self.var = StringVar(self.frame1)
        self.var.set(self.Component_choices[0])
        self.var.set(self.Component_choices[0])
        self.selection = OptionMenu(self.frame2, self.var, *self.Component_choices)
        self.selection.place(relx=0.11, rely=0.5, anchor=W)

        self.PlotSpectrumButton = Button(self.frame2, text="Plot Spectrum",
                                         command=lambda: self.show_plot("spec"))
        self.PlotSpectrumButton.place(relx=0.31, rely=0.5, anchor=W)

        self.PlotTimeProfileButton = Button(self.frame2, text="Plot Time Profile",
                                            command=lambda: self.show_plot("time"))
        self.PlotTimeProfileButton.place(relx=0.457, rely=0.5, anchor=W)

        self.PlotSpectrogramButton = Button(self.frame2, text="Plot Spectrogram",
                                            command=lambda: self.show_plot("specgr"))
        self.PlotSpectrogramButton.place(relx=0.62, rely=0.5, anchor=W)

        self.closeButton = Button(self.top1, text="Close", command=self.destroy)
        self.closeButton.place(relx=0.5, rely=0.9, anchor=N)


        # Loads data for time selection
        hdu = fits.open(SecondWindow.fname)
        headers = SecondWindow.extract_stix_header(hdu)
        data = SecondWindow.extract_stix_data(hdu)

        self.time_summarize = [data['time'][-1], headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown')),
                                headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))]  # time data

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
        SecondWindow.fname = self.name

        if self.name:
            with fits.open(self.name) as hdul:
                self.hdul = hdul
                self.textFilename.insert(0, self.name)  # Displays the input file name in Entry box
                # Loads data to do the plot
                data = SecondWindow.extract_stix_data(self.hdul)
                # self.evt_data = Table(self.hdul[2].data)
                # self.energies = Table(self.hdul[3].data)

                self.evt_data = data['counts']
                self.e_low = data['e_low']
                self.e_high = data['e_high']
                
                headers = SecondWindow.extract_stix_header(self.hdul)
                self.time_summarize = [data['time'][-1], headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown')),
                                        headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))]  # time data

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

        # Loads data to do the plot
        hdu = fits.open(SecondWindow.fname)
        headers = SecondWindow.extract_stix_header(hdu)
        data = SecondWindow.extract_stix_data(hdu)

        # Loads headers informations
        self.data_file = headers['INSTRUME']  # data type
        self.time_summarize = [data['time'][-1], headers.get('DATE_BEG', headers.get('DATE-BEG', 'Unknown')),
                                headers.get('DATE_END', headers.get('DATE-END', 'Unknown'))]  # time data
        self.energie_summarize = [data['counts'].shape[1],
                                    max(min(data['e_low']), min(data['e_high'])),
                                    min(max(data['e_low']),
                                        max(data['e_high']))]  # energies data
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
        """Closing 'Select Input' window, clicking 'Close' button."""
        self.top1.destroy()

    def submit(self):
        """Submit the value of start and end times for plotting with the date."""
        self.Times_range = []
        self.start_time_plotting = self.start_date_var.get()
        self.end_time_plotting = self.end_date_var.get()
        # print('Start time: ' + self.start_time_plotting)
        # print('End time: ' + self.end_time_plotting)

        # Updating new text
        # self.textStart = self.start_time_plotting
        # self.textEnd = self.end_time_plotting

        # Creation of value of editInterval_times
        self.Times_range.append(self.start_time_plotting)
        self.Times_range.append(self.end_time_plotting)

    # def checked(self):
    #     """ 'Set From' button activation.
    #     Allows user to select between two distinct values (e.g. on/off)."""
    #     if self.SetFromButton['state'] == 'disabled':
    #         self.SetFromButton['state'] = 'normal'
    #         self.StartButton['state'] = 'normal'
    #         self.EndButton['state'] = 'normal'

    #         self.textStart['state'] = 'normal'
    #         self.textEnd['state'] = 'normal'

    #         # Delete content from position 0 to end
    #         self.textStart.delete(0, 'end')
    #         self.textEnd.delete(0, 'end')

    #         # Insert new_text at position 0
    #         self.textStart.insert(0, self.time_summarize[1])
    #         self.textEnd.insert(0, self.time_summarize[2])

    #     else:
    #         # Delete content from position 0 to end
    #         self.textStart.delete(0, 'end')
    #         self.textEnd.delete(0, 'end')
            
    #         self.SetFromButton['state'] = 'disabled'
    #         self.StartButton['state'] = 'disabled'
    #         self.EndButton['state'] = 'disabled'

    #         self.textStart['state'] = 'disabled'
    #         self.textEnd['state'] = 'disabled'

    def checked(self):
        """ 'Set From' button activation.
        Allows user to select between two distinct values (e.g. on/off)."""
        if self.SetFromButton['state'] == 'disabled':
            # ACTIVER
            self.SetFromButton['state'] = 'normal'
            self.StartButton['state'] = 'normal'
            self.EndButton['state'] = 'normal'
            self.textStart['state'] = 'normal'
            self.textEnd['state'] = 'normal'

            # REMPLIR
            self.textStart.delete(0, 'end')
            self.textEnd.delete(0, 'end')
            self.textStart.insert(0, self.time_summarize[1])
            self.textEnd.insert(0, self.time_summarize[2])

        else:
            # DESELECTION = VIDER et DESACTIVER
            self.textStart.delete(0, 'end')
            self.textEnd.delete(0, 'end')

            self.SetFromButton['state'] = 'disabled'
            self.StartButton['state'] = 'disabled'
            self.EndButton['state'] = 'disabled'
            self.textStart['state'] = 'disabled'
            self.textEnd['state'] = 'disabled'

    # ==================== Calling for plotting.py ====================

    def show_plot(self, e):
        """Calls the class to plot Spectrum, Time profile, Spectrogram.
        Parameters are taken from .fits file, chosen(loaded) by user."""

        # If entire file:
        if self.SetFromButton['state'] == 'disabled':
            plots = plotting.Input(self.name)
        # If not entire file:
        elif self.SetFromButton['state'] == 'normal':
            self.submit()
            plots = plotting.Input(self.name, self.Times_range[0], self.Times_range[1], self.time_summarize[1])
        else:
            plots = None
            print('Error')

        if self.var.get() == 'Rate':
            if e == 'time':
                plots.rate_vs_time_plotting()
            elif e == 'spec':
                plots.plot_spectrum_rate()
            elif e == 'specgr':
                plots.plot_spectrogram_rate()

        if self.var.get() == 'Counts':
            if e == 'time':
                plots.counts_vs_time_plotting()
            elif e == 'spec':
                plots.plot_spectrum_counts()
            elif e == 'specgr':
                plots.plot_spectrogram_counts()

        if self.var.get() == 'Flux':
            if e == 'time':
                plots.flux_vs_time_plotting()
            elif e == 'spec':
                plots.plot_spectrum_flux()
            elif e == 'specgr':
                plots.plot_spectrogram_flux()

    def extract_stix_header(hdulist):
        result = {}

        for key, value, comment in hdulist[0].header.cards:
            result[key] = value

        for key, value, comment in hdulist[3].header.cards:
            result[key] = value
        
        return result
    
    def extract_stix_data(hdulist):
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