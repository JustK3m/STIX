import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from matplotlib.widgets import RectangleSelector
from matplotlib.dates import num2date
from matplotlib.dates import AutoDateLocator, DateFormatter
from datetime import datetime



class IntervalSelector:
    
    def __init__(self, xdata, ydata, x_scale=None, plot_label=None, col_label=None,
                 title=None, xlabel=None, ylabel=None, color=None, samefig=True, band=0, rounding=3):
        """Plots data to allow user to choose a time interval by selecting directly on the figure.

        Parameters: \n
            xdata: list or matrix containing data for x axis (i.e. times); \n
            ydata: list or matrix containing data for y axis (i.e. energies); \n\n

            All parameters below are optionnal:\n
            x_scale: scale of x axis (usually with np.arrange); \n
            plot_label: labels for units on x axis, linked to x_scale; \n
            col_label: label for each band in ydata; \n
            title: displayed name of the figure; \n
            xlabel: displayed label for x axis; \n
            ylabel: displayed label for y axis; \n
            color: list of colors for the ydata; \n
            samefig: if True, plots all ydata on the same graph; else, plots only the ydata of the band;
                     can be set as a binary (0/1) or as a boolean (False/True); \n
            band: when samefig is False, plots only the ydata of the given band; \n
            rounding: rounding of returned starting and ending times; set by default to 3. \n

        Returns: \n
            start x & end x: starting and ending x for selection as floats."""

        self.fig, self.ax = plt.subplots()
        self.x_start = None
        self.y_start = None
        self.x_end = None
        self.y_end = None
        self.plot = None

        if col_label:
            self.df = pd.DataFrame(ydata, index=xdata, columns=col_label[0] if col_label else None)
            if samefig:
                for band in range(np.size(ydata, 1) - 1):
                    self.df[col_label[0][band]].plot(ax=plt.gca(), color=color[band] if color else None)
            else:
                self.df[col_label[0][band]].plot(ax=plt.gca(), color=color[band] if color else None)

        else:
            plt.plot(xdata, ydata[:, 0:np.size(ydata, 1) - 1])

        self.round = int(rounding)

        # plt.xticks(x_scale, plot_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Axe X dynamique
        ax = plt.gca()
        ax.set_xlim(xdata[0], xdata[-1]) # ADJUSTE LIMITE X

        ax.grid(True, which='major')  # principal grid

        ax.minorticks_on()  # Active ticks secondaires
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

        plt.draw()

        locator = AutoDateLocator()
        formatter = DateFormatter('%H:%M:%S')  # Format par défaut : HH:MM

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        def update_xticks_on_zoom(event):
            ax = event.canvas.figure.axes[0]
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            event.canvas.draw_idle()

        plt.gcf().canvas.mpl_connect('draw_event', update_xticks_on_zoom)
        plt.gcf().autofmt_xdate()

    def line_select_callback(self, eclick, erelease):
        """Callback to record starting and ending times when holding left click."""

        self.x_start, self.y_start = eclick.xdata, eclick.ydata
        self.x_end, self.y_end = erelease.xdata, erelease.ydata

        rect = plt.Rectangle((min(self.x_start, self.x_end), min(self.y_start, self.y_end)),
                             np.abs(self.x_start - self.x_end), np.abs(self.y_start - self.y_end),
                             edgecolor='k', facecolor='r', alpha=0.25)
        self.ax.add_patch(rect)

        time.sleep(1)
        plt.close()

    def graphical_selection(self):
        """Enables the rectangle selector and returns starting and ending times."""

        self.plot = RectangleSelector(self.ax, self.line_select_callback, useblit=False, button=[1], minspanx=5,
                                      minspany=5, spancoords='pixels', interactive=True)
        


        plt.show()
        # return round(self.x_start, self.round), round(self.x_end, self.round)
       
        # if isinstance(self.x_start, datetime):
        #     return self.x_start, self.x_end
        # else:
        #     return round(self.x_start, self.round), round(self.x_end, self.round)

        dt_start = num2date(self.x_start)
        dt_end = num2date(self.x_end)

        dt_start = dt_start.replace(tzinfo=None)  # Remove timezone info if present
        dt_end = dt_end.replace(tzinfo=None)      

        return dt_start, dt_end


    # def graphical_selection(self):
    #     """Enables rectangle selection and returns start/end as timestamps."""
    #     self.plot = RectangleSelector(self.ax, self.line_select_callback, useblit=False, button=[1], minspanx=5,
    #                                 minspany=5, spancoords='pixels', interactive=True)

    #     plt.show()

    #     # Convert x values from matplotlib date num to timestamps
    #     dt_start = num2date(self.x_start)
    #     dt_end = num2date(self.x_end)

    #     ts_start = dt_start.timestamp()
    #     ts_end = dt_end.timestamp()

    #     return round(ts_start, self.round), round(ts_end, self.round)



if __name__ == '__main__':

    data_length = 41                    # Number of subdivisions contained in times
    x = np.linspace(0, 40, data_length)
    y = np.zeros([data_length, 3])
    for i in range(data_length):
        y[i, 0] = np.sin(0.25 * x[i])   # First column for first y data
        y[i, 1] = np.cos(0.25 * x[i])   # Second column for second y data
        y[i, 2] = x[i]                  # Last column is x data

    print("\nPrinting the 6 first lines of what y data should look like:")
    print(y[0:6, :], '\n')

    # Optionnal parameters
    test_x_scale = np.linspace(0, 40, 9)
    test_plot_label = [str(int(i)) for i in test_x_scale]
    test_col_label = [['Sin(x)', 'Cos(x)', 'Times']]

    test_title = "Test interval selector"
    test_xlabel = "x"
    test_ylabel = "y"
    test_color = ['red', 'green']

    test_samefig = True                 # Can be set as a binary (0/1) or as a boolean (False/True)
    test_band = 0                       # While samefig is True, this parameter has no importance
    test_rounding = 1

    start, end = IntervalSelector(x, y,
                                  x_scale=test_x_scale,
                                  plot_label=test_plot_label,
                                  col_label=test_col_label,
                                  title=test_title,
                                  xlabel=test_xlabel,
                                  ylabel=test_ylabel,
                                  color=test_color,
                                  samefig=test_samefig,
                                  band=test_band,
                                  rounding=test_rounding).graphical_selection()

    print("Interval Selector returns min and max x corresponding to the selection")
    print("x_start = ", start)
    print("x_end =   ", end)
