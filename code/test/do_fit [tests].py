from tkinter import *
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from pandas.plotting import register_matplotlib_converters
import numpy as np
import new_window
import background
import rebin_flux as rebin

register_matplotlib_converters()


class Fitting:
    """
    Class to perform a spectrum fitting
    """

    E_min = None
    """Energy in keV"""
    setEVal = None
    """Energy setting value"""
    evalue = None
    """Value from Energy array chosen by user"""

    # create a new window called 'SPEX Fit Options'
    def __init__(self, root):
        """Creates a new window, providing widgets to perform fitting analysis"""
        self.sender = None

        self.top2 = Toplevel()
        self.top2.title('SPEX Fit Options')  # title of the window
        self.top2.geometry("1000x600")  # size of the new window
        Label(self.top2,
              text="Fit Options",  # place the text at the top of the window
              fg="red",  # in red
              font="Helvetica 12 bold italic").pack()  # with specific text font

        self.root = root
        self.sepBkVar = IntVar()

        self.lbl1 = Label(self.top2, text="Choose Fit Function Model:", fg='blue',
                          font=("Helvetica", 11, "bold"))  # name the listbox
        self.lbl1.place(relx=0.07, rely=0.07)  # set the position on window

        self.lbl2 = Label(self.top2, text="Information:", fg='blue',
                          font=("Helvetica", 11, "bold"))  # name the scrollbar
        self.lbl2.place(relx=0.44, rely=0.07)  # set the position

        self.lbl3 = Label(self.top2, text="Set function components and energy_data, spec_data parameters:", fg='blue',
                          font=("Helvetica", 11, "bold"))  # name the scrollbar
        self.lbl3.place(relx=0.65, rely=0.07)  # set the position

        self.lblFunc = Label(self.top2, text="Set function components: ")  # name the scrollbar
        self.lblFunc.place(relx=0.73, rely=0.20)  # set the position

        # FIXME: Currently opening a new window, which is not convenient; to be replaced with EntryInt textboxes.
        setY = str(Fitting.setEVal) if Fitting.setEVal is not None else '8 - 40'
        """Set spec_data axis. User choice from the interafce"""
        print("set val", setY, new_window.Set_Energy.yVal)  # Displays "set val 8 - 40 None"
        Fitting.evalue = StringVar()
        Fitting.evalue.set(setY)
        self.show_Button = Button(self.top2, textvariable=Fitting.evalue, command=lambda: self.editEnergy(self.top2))
        """Button to select the value(s) for spec_data axis.
           By default it is called '8 - 40'. Changes the name by user choice. For example: 
           If user selected Energy range '30 - 100', the name of the button will display this info('30 - 100')"""
        self.show_Button.place(relx=0.81, rely=0.39, relheight=0.05, relwidth=0.07)

        self.fit_model = str()

        # FIXME: Empty Window, surely to be removed later on
        def Set_Function():  # new window Set_Function definition
            """Creates a new window for "Set spec_data axis" part"""
            newwin = Toplevel(root)
            newwin.title('Function values')  # title of the window
            newwin.geometry("600x400")  # size of the new window
            display = Label(newwin, text="Choose function values: ", fg='blue', font=("Helvetica", 11, "bold"))
            display.place(relx=0.04, rely=0.07)

        self.Value_Button = Button(self.top2, text="Function value(s)",
                                   command=Set_Function)  # place a "Function value" button
        self.Value_Button.place(relx=0.75, rely=0.26, relheight=0.05, relwidth=0.13)  # locate

        self.X_Label = Label(self.top2, text="Energy range(s) to fit: ")  # name "Energy range(s) to fit"
        self.X_Label.place(relx=0.65, rely=0.40)  # locate

        # ============== Main window description ==============

        self.lbox = Listbox(self.top2, selectmode=EXTENDED, highlightcolor='red', bd=4, selectbackground='grey')
        """ 
        On the left side of the 'SPEX Fit Options' window: place a list of text alternatives (listbox).
        The user can choose(highlight) one of the options.
        Options(functions):
        1) One Dimensional Power Law;
        2) 1-D Broken Power Law;
        3) Gaussian;
        4) Polynomial;
        5) Exponential;
        6) Single Power Law Times an Exponetial
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

        self.DoFit5_Button = Button(self.frameFit, text="Do Fit",
                                    command=self._selective_fit)  # place a "Do Fit" button
        self.DoFit5_Button.place(relx=0.65, rely=0.38, relheight=0.23, relwidth=0.15)  # locate

        self.refreshButton5 = Button(self.top2, text="Refresh")  # add Refresh button at the buttom
        # resets original view
        self.refreshButton5.place(relx=0.4, rely=0.94)

        """Scrollbar with information related to each function"""
        self.closeButton5 = Button(self.top2, text="Close", command=self.destroy5)  # add Close button
        # Close "Fit Options" window
        self.closeButton5.place(relx=0.5, rely=0.94)
        self.models = ['PowerLaw1D', 'BrokenPowerLaw1D', 'Gaussian', 'Polynomial', 'Exponential',
                       'Single Power Law Times an Exponential', 'Logistic Regression', 'Lorentz', 'Moffat',
                       'Voigt Profile']  # function names
        for p in self.models:
            """On the right: place an 'entry text' Scrollbar widget (scrollbar) When user highlight the function, 
            displays the text information about function description and input parameters"""
            self.lbox.insert(END, p)
        self.lbox.bind("<<ListboxSelect>>", self.onSelect)
        self.list = {'PowerLaw1D': {'One dimensional power law model', '\n\n',
                                    'amplitude – model amplitude at the reference energy', '\n',
                                    'energy_data – reference energy', '\n', 'alpha – power law index'},
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
                                    'c5 - energy_data offset, such that function value at energy_data = c5 is C0 '},  # Polynomial
                     'Exponential': {'Exponential function', '\n', 't0 - Normalization', '\n',
                                     't1 - Pseudo temperature'},  # Exponential
                     'Single Power Law Times an Exponetial': {'Multiplication of Single Power Law and Exponential',
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
                                       'amplitude centered at x_0 with the specified Lorentzian and Gaussian widths'}
                     # Voigt
                     }
        self.list_selection = Listbox(self.top2, highlightcolor='red', bd=4)
        self.list_selection.place(relx=0.33, rely=0.15, relheight=0.45, relwidth=0.30)

    @staticmethod
    def editEnergy(p1):
        """Call new class to edit spec_data axis"""
        new_window.Set_Energy(p1)

    def onSelect(self, event):
        """Determine the function selection from the list"""
        widget = event.widget
        selection = widget.curselection()
        files_avalibe = []

        if selection:
            for s_i in selection:
                selected_i = self.models[s_i]
                files_avalibe += self.list[selected_i]
                print(files_avalibe)
                # Displays ['\n', 'Exponential function', 't1 - Pseudo temperature', 't0 - Normalization']

                self.update_file_list(files_avalibe)

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

    def _selective_fit(self):
        """Selection depending on Plot Units and Function Model
          Predefine Input Data in energy_data and spec_data
          We equate three components to rate_data, counts_data, flux_data. The value of energy_data is the same for all cases
          energy_data - independent variable, nominally energy in keV
          spec_data - Plot Unit"""
        # load chosen file in Select Input section
        fname = background.BackgroundWindow.fname
        if fname is None:  # if file not choosen, print
            print('Please, choose input file')

        else:
            # self.hdulist[2].data, self.hdulist[3].data, self.hdulist[0].header, self.hdulist[3].header
            # data,                 data_energies,        header_dates,           header_energy
            hdulist = fits.open(fname)
            data = hdulist[2].data
            data_energies = hdulist[3].data
            counts = data.counts
            counts_err = data.counts_err
            Time = data.time - 2
            Time_del = data.timedel
            Rate = np.zeros(np.shape(counts))
            Rate_err = np.zeros(np.shape(counts_err))
            for j in range(len(Time)):
                Rate[j] = counts[j]
                Rate_err[j] = counts_err[j]
            Fitting.E_min = data_energies.e_low
            E_max = data_energies.e_high
            Area = 6

            """Define Spectrum Units: Rate, Counts, Flux"""

            # Define the range for Low and High energies
            n = len(Fitting.E_min)
            deltaE = np.zeros(shape=n)
            for i in range(n):
                deltaE[i] = E_max[i] - Fitting.E_min[i]

            # Next, we determine the PLot Units components
            # Rate
            CountRate = np.zeros(shape=n)
            CountRate_err = np.zeros(shape=n)
            for i in range(n):
                CountRate[i] = np.mean(Rate[:, i])
                CountRate_err[i] = np.mean(Rate_err[:, i])

            # Counts
            Counts = np.zeros(shape=n)
            Counts_err = np.zeros(shape=n)
            for i in range(n):
                Counts[i] = np.mean(Rate[:, i] * Time_del[:])
                Counts_err[i] = np.mean(Rate_err[:, i] * Time_del[:])

            # Flux
            Flux = np.zeros(shape=n)
            Flux_err = np.zeros(shape=n)
            for i in range(n):
                Flux[i] = np.mean(Rate[:, i] / (Area * deltaE[i] - 2))
                Flux_err[i] = np.mean(Rate_err[:, i] / (Area * deltaE[i] - 2))

            # Set the conditions to Set spec_data axis
            if Fitting.setEVal is None:
                energy_data = Fitting.E_min
                rate_data = CountRate
                counts_data = Counts
                flux_data = Flux
                rate_data_err = CountRate_err
                counts_data_err = Counts_err
                flux_data_err = Flux_err
            else:
                # Energy boundaries
                energy_min = int(Fitting.setEVal.split(' - ')[0])
                energy_max = int(Fitting.setEVal.split(' - ')[1])
                assert energy_max > energy_min
                # Energy value mask
                energy_mask = (Fitting.E_min >= energy_min) & (Fitting.E_min <= energy_max)
                energy_data = Fitting.E_min[energy_mask]
                rate_data = CountRate[energy_mask]
                counts_data = Counts[energy_mask]
                flux_data = Flux[energy_mask]
                rate_data_err = CountRate_err[energy_mask]
                counts_data_err = Counts_err[energy_mask]
                flux_data_err = Flux_err[energy_mask]

            # def find_all_indexes(input_str, search_str):
            #     l1 = []
            #     length = len(input_str)
            #     index = 0
            #     while index < length:
            #         i = input_str.find(search_str, index)
            #         if i == -1:
            #             return l1
            #         l1.append(i)
            #         index = i + 1
            #     return l1
            # print(find_all_indexes(str(E_min), str(E_min[0:-1])))
            # indexesX = np.where((energy_data <= energy_data[-1]) & (energy_data >= energy_data[0]))
            # print(indexesX)
            # indexesY1 = np.where((flux_data < flux_data[-1]) & (flux_data > flux_data[0]))
            # print(indexesY1)
            # nX = int(input(self.e1.get()))
            # nY = int(input(self.e1.get()))
            # keyword_arrayX = []
            # keyword_arrayY = []
            # first_E_min = indexes[0]
            # last_E_min = indexes[-1]
            #
            # if first_E_min < arrayX[0] and last_E_min<arrayX[-1]:

            # ============== Define Fitters ==============

            # Fitter creates a new model for energy_data and у, with finding the best fit values
            fit = fitting.LevMarLSQFitter()
            # print(fitg1)

            """ 
            Levenberg - Marquandt algorithm for non - linear least - squares optimization
    
            The algorithm works by minimizing the squared residuals, defined as:
                
                    Residual^2 = (spec_data - f(t))^2 ,
     
            where spec_data is the measured dependent variable;
    
            f(t) is the calculated value
    
            The LM algorithm is an iterative process, guessing at the solution of the best minimum
            """

            # ============== Fitting the data using astropy.modeling ==============

            # Define a One dimensional broken power law model
            broken_power_law = models.BrokenPowerLaw1D(amplitude=1, x_break=3, alpha_1=400, alpha_2=1.93,
                                                       fixed={'alpha_1': True, 'alpha_2': True})

            """
            BrokenPowerLaw1D(amplitude=1, x_break=1, alpha_1=1, alpha_2=1, **kwargs)


            One dimensional power law model with a break.

            Parameters:	

                amplitude : float. Model amplitude at the break point.
    
                x_break : float. Break point.
    
                alpha_1 : float. Power law index for energy_data < x_break.
    
                alpha_2 : float. Power law index for energy_data > x_break.
            """

            # Define a Gaussian model
            ginit = models.Gaussian1D(1000, 6.7, 0.1, fixed={'mean': True, 'stddev': True})
            # (1000, 6.7, 0.1)

            """
            One dimensional Gaussian model
    
            Parameters:
    
                amplitude: Amplitude of the Gaussian.
                
                mean: Mean of the Gaussian.
    
                stddev: Standard deviation of the Gaussian.
           
            Other Parameters:
    
                fixed : optional. A dictionary {parameter_name: boolean} of parameters to not be varied during fitting. 
                True means the parameter is held fixed. Alternatively the fixed property of a parameter may be used.
    
        
                tied: optional. A dictionary {parameter_name: callable} of parameters which are linked to some other 
                parameter.
    
                The dictionary values are callables providing the linking relationship. Alternatively the tied property 
                of a parameter may be used.
    
        
                bounds: optional. A dictionary {parameter_name: value} of lower and upper bounds of parameters. Keys are 
                parameter names. Values are a list or a tuple of length 2 giving the desired range for the parameter. 
                Alternatively, the min and max properties of a parameter may be used.
    
                eqcons: optional. A list of functions of length n such that eqcons[j](x0,*args) == 0.0 in a successfully 
                optimized problem.
    
            
                ineqcons: optional. A list of functions of length n such that ieqcons[j](x0,*args) >= 0.0 is a 
                successfully optimized problem.
            """
            p_init = models.Polynomial1D(2)  # Define 2nd order Polynomial function
            # p_init.parameters = [1,1,1]

            """
            1D Polynomial model.
            
            
            Parameters:
    
                degree: Degree of the series.
    
            
                domain: Optional.
    
                window: Optional. If None, it is set to [-1,1] Fitters will remap the domain to this window.
    
            
                **params: Keyword. Value pairs, representing parameter_name: value.
    
            
    
            Other Parameters:
    
                fixed: optional. A dictionary {parameter_name: boolean} of parameters to not be varied during fitting. 
                True means the parameter is held fixed. Alternatively the fixed property of a parameter may be used.
    
                tied: optional. A dictionary {parameter_name: callable} of parameters which are linked to some other 
                parameter. The dictionary values are callables providing the linking relationship. Alternatively the 
                tied property of a parameter may be used.
       
                bounds: optional. A dictionary {parameter_name: value} of lower and upper bounds of parameters. Keys are 
                parameter names. Values are a list or a tuple of length 2 giving the desired range for the parameter. 
                Alternatively, the min and max properties of a parameter may be used.
    
                eqcons: optional.  A list of functions of length n such that eqcons[j](x0,*args) == 0.0 in a 
                successfully optimized problem.
    
           
                ineqcons: optional. A list of functions of length n such that ieqcons[j](x0,*args) >= 0.0 is a 
                successfully optimized problem.
                """

            gaussian = ginit + p_init
            # gaussian = ginit

            """ The Model(function) returns the sum of a Gaussian and 2nd order Polynomial """

            # Define 6th order Polynomial function
            poly = models.Polynomial1D(5, window=[-10, 10], fixed={'c3': True, 'c4': True})
            poly.parameters = [1, 1, 1, 1, 1, 50]

            # Define Exponential function
            @custom_model
            def func_exponential(e, t1=1., t2=1.):
                print(type(np.exp(t1 - e / t2)))
                return np.exp(t1 - e / t2)

            exp = func_exponential(t1=1., t2=1.)

            """
            Purpose: Exponential function
    
            Category: spectral fitting
    
            Inputs:
            t0 - Normalization
            t1 - Pseudo temperature
    
            Outputs:
            result of function, exponential
            """

            # Define Single Power Law Times an Exponential
            @custom_model
            def func_exponential_powerlaw(epl, p0=1., p1=1., p2=1., e3=1., e4=1.):
                return (p0 * (epl / p2) ** p1) * (np.exp(e3 - epl / e4))

            exp_powerlaw = func_exponential_powerlaw(p0=1., p1=3., p2=50., e3=1., e4=1., fixed={'p2': True})

            """
            Purpose: single power - law times an exponential
    
            Category: spectral fitting
    
            Inputs:
            p - first 3 parameters describe the single power - law, e - describes the exponential
     
            p0 = normalization at epivot for power - law
            p1 = negative power - law index
            p2 = epivot (keV) for power - law
    
            e3 = normalization for exponential
            e4 = pseudo temperature for exponential
    
            Outputs:
            result of function, a power - law times an exponential
            """

            # Define Logistic Regression
            @custom_model
            def func_logisticReg(lr):
                return 1 / (1 + np.exp(-lr))

            logistic_regression = func_logisticReg()

            # Define One dimensional Lorentzian model
            lorentz = models.Lorentz1D(1000, 6.7, 0.1)

            # Define Moffat1D  model
            moffat = models.Moffat1D(1000, 6.7, 0.1)

            # Define Voigt model
            voigt = models.Voigt1D(6.7, 1000, 0.05, 0.05) + p_init

            # =========================== Fitting plot ===========================

            plt.figure()

            maxiter = 10**5

            # Unit selection
            if self.var.get() == 'Rate':
                spec_data = rate_data[:]
                spec_data_err = rate_data_err
                plt.plot(energy_data, spec_data, drawstyle='steps-post', color='blue', label="Rate Data")
                plt.ylabel('Rate (Counts/s)')

            elif self.var.get() == 'Counts':
                spec_data = counts_data[:]
                spec_data_err = counts_data_err
                plt.plot(energy_data, spec_data, drawstyle='steps-post', color='blue', label="Counts Data")
                plt.ylabel('Counts (Counts)')

            elif self.var.get() == 'Flux':
                spec_data = flux_data[:]
                spec_data_err = flux_data_err
                plt.plot(energy_data, spec_data, drawstyle='steps-post', color='blue', label="Flux Data")
                plt.ylabel('Flux (Counts/s/cm²/keV)')
            else:
                spec_data = counts_data[:]
                print("Unit type not found")

            weights_fitting = np.array(spec_data)
            if spec_data is not None:
                for i in range(len(spec_data)):
                    if spec_data[i] == 0:
                        weights_fitting[i] = 1
                    else:
                        weights_fitting[i] = 1 / spec_data[i]

            # FIXME: These two following indices can be directly taken from an EntryInt box that could replace
            #  new_window.py instead of being manually entered in the code. Important note: these values are indices,
            #  not energy values!
            low_bd = 0  # Default value: 0
            up_bd = 20  # Default value: 32

            # Fitting method selection
            if self.lbox.curselection()[0] == 0:
                self.fit_model = 'Power Law'
                power_law = rebin.fitting_photons(method=0, p1=1., p2=1., index_start=low_bd,
                                                  fixed={'p1': True, 'index_start': True})
                pl_fit = fit(power_law, energy_data, spec_data, maxiter=maxiter)
                print(pl_fit)
                # plt.plot(energy_data, pl_fit(energy_data),
                #          drawstyle='steps-post', color='green', label="Fitting with Power Law")

            # FIXME: Starting from here, rebin_flux.py and this code should be adapted to do the correct fitting in the
            #  photons domain, same as in Power Law.
            elif self.lbox.curselection()[0] == 1:
                self.fit_model = 'Broken Power Law'
                bpl_fit = fit(broken_power_law, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(bpl_fit)
                plt.plot(energy_data, bpl_fit(energy_data), drawstyle='steps-post', color='red', label="BrokenPowerLaw1D")

            elif self.lbox.curselection()[0] == 2:
                self.fit_model = 'Gaussian'
                gauss_fit = fit(gaussian, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(gauss_fit)
                plt.plot(energy_data, gauss_fit(energy_data), drawstyle='steps-pre', color='red', label='Gaussian Fitting')

            elif self.lbox.curselection()[0] == 3:
                self.fit_model = 'Polynomial'
                poly_fit = fit(poly, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(poly_fit)
                plt.plot(energy_data, poly_fit(energy_data), drawstyle='steps-pre', color='red', label='Polynomial')

            elif self.lbox.curselection()[0] == 4:
                self.fit_model = 'Exponential'
                exp = rebin.fitting_photons(method=4, p1=1., p2=1., index_start=low_bd,
                                            fixed={'index_start': True})
                exp_fit = fit(exp, energy_data[low_bd:up_bd], spec_data[low_bd:up_bd], maxiter=maxiter)  # , weights=weights_fitting
                print(exp_fit)
                plt.plot(energy_data, exp_fit(energy_data), drawstyle='steps-pre', color='red', label='Exponential')

            elif self.lbox.curselection()[0] == 5:
                self.fit_model = 'Exponential Power Law'
                exp_pl_fit = fit(exp_powerlaw, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(exp_pl_fit)
                plt.plot(energy_data, exp_pl_fit(energy_data), drawstyle='steps-pre', color='red', label="ExpPowerLaw")

            elif self.lbox.curselection()[0] == 6:
                self.fit_model = 'Logistic Regression'
                log_reg_fit = fit(logistic_regression, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(log_reg_fit)
                plt.plot(energy_data, log_reg_fit(energy_data), drawstyle='steps-pre', color='red', label='Logistic Regression')

            elif self.lbox.curselection()[0] == 7:
                self.fit_model = 'Lorentz'
                lorentz_fit = fit(lorentz, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(lorentz_fit)
                plt.plot(energy_data, lorentz_fit(energy_data), drawstyle='steps-pre', color='red', label='Lorentz')

            elif self.lbox.curselection()[0] == 8:
                self.fit_model = 'Moffat'
                moffat_fit = fit(moffat, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(moffat_fit)
                plt.plot(energy_data, moffat_fit(energy_data), drawstyle='steps-pre', color='red', label='Moffat')

            elif self.lbox.curselection()[0] == 9:
                self.fit_model = 'Voigt'
                voigt_fit = fit(voigt, energy_data, spec_data, weights=weights_fitting, maxiter=maxiter)
                print(voigt_fit)
                plt.plot(energy_data, voigt_fit(energy_data), drawstyle='steps-pre', color='red', label='Voigt profile')
            """
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Energy(keV)')
            # plt.legend(loc=2)
            plt.xlim(4, 40)
            # plt.ylim(ymax=100, ymin=0.1)  # FIXME [2021]: find a solution for general case
            plt.title(str(self.var.get()) + ' Fitting using 1D ' + str(self.fit_model) + ' Model')
            plt.legend()
            plt.show()
            """
            # Calculate the Reduced Chi - square, test version
            # Initial guess
            # N = len(E_min) #total number of points
            # print(N)
            # sigma = 1.0
            # spec_data_err = sigma / E_min

            # def calc_reduced_chi_square(fit, energy_data, spec_data, yerr, N, n_free):
            # """
            # fit (array) values for the fit
            # energy_data,spec_data,spec_data_err (arrays) data
            # N total number of points
            # n_free number of parameters we are fitting
            # """
            # return 1.0 / (N - n_free) * sum(((fit - spec_data) / spec_data_err) ** 2)

            # reduced_chi_squared = calc_reduced_chi_square(gPLFlux(energy_data),energy_data,flux_data,flux_data, N, 3)
            # print('Reduced Chi Squared with Levenberg - Marquandt algorithm: {}'.format(reduced_chi_squared))
