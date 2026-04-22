import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import SpanSelector
from matplotlib.dates import num2date, AutoDateLocator, DateFormatter
from datetime import datetime


class IntervalSelector:

    def __init__(self, xdata, ydata, x_scale=None, plot_label=None, col_label=None,
                 title=None, xlabel=None, ylabel=None, color=None, samefig=True, band=0, rounding=3):
        """Plots data to allow user to choose a time interval by selecting directly on the figure.

        Parameters:
            xdata: list or matrix containing data for x axis (i.e. times);
            ydata: list or matrix containing data for y axis (i.e. energies);

            All parameters below are optional:
            x_scale: scale of x axis;
            plot_label: labels for units on x axis;
            col_label: label for each band in ydata;
            title: displayed name of the figure;
            xlabel: displayed label for x axis;
            ylabel: displayed label for y axis;
            color: list of colors for the ydata;
            samefig: if True, plots all ydata on the same graph;
            band: when samefig is False, plots only the ydata of the given band;
            rounding: rounding of returned starting and ending times.

        Returns:
            start x & end x: starting and ending x for selection as datetimes."""

        self.fig, self.ax = plt.subplots()
        self.x_start = None
        self.x_end   = None
        self._span   = None  # Maintenu en vie dans self

        if col_label:
            self.df = pd.DataFrame(ydata, index=xdata, columns=col_label[0] if col_label else None)
            if samefig:
                for b in range(np.size(ydata, 1) - 1):
                    self.df[col_label[0][b]].plot(ax=self.ax, color=color[b] if color else None)
            else:
                self.df[col_label[0][band]].plot(ax=self.ax, color=color[band] if color else None)
        else:
            self.ax.plot(xdata, ydata[:, 0:np.size(ydata, 1) - 1])

        self.round = int(rounding)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.set_xlim(xdata[0], xdata[-1])
        self.ax.grid(True, which='major')
        self.ax.minorticks_on()
        self.ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

        locator   = AutoDateLocator()
        formatter = DateFormatter('%H:%M:%S')
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(formatter)

        def update_xticks_on_zoom(event):
            event.canvas.figure.axes[0].xaxis.set_major_locator(locator)
            event.canvas.figure.axes[0].xaxis.set_major_formatter(formatter)
            event.canvas.draw_idle()

        self.fig.canvas.mpl_connect('draw_event', update_xticks_on_zoom)
        self.fig.autofmt_xdate()

        # Force le focus sur cette fenêtre même si d'autres figures sont ouvertes
        try:
            self.fig.canvas.manager.window.lift()
            self.fig.canvas.manager.window.focus_force()
        except Exception:
            pass

    def graphical_selection(self):
        """Active le SpanSelector horizontal et retourne start/end en datetime."""

        def onselect(xmin, xmax):
            self.x_start = xmin
            self.x_end   = xmax
            # Dessine la zone sélectionnée et ferme
            self.fig.canvas.draw_idle()
            plt.close(self.fig)

        self._span = SpanSelector(
            self.ax, onselect, 'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='steelblue'),
            interactive=True
        )

        plt.show()  # Bloquant : retourne après plt.close(self.fig)

        if self.x_start is None or self.x_end is None:
            return None, None

        dt_start = num2date(self.x_start).replace(tzinfo=None)
        dt_end   = num2date(self.x_end).replace(tzinfo=None)

        return dt_start, dt_end