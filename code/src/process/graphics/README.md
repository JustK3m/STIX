# graphics/ - plotting

Two modules: `plotting.py` (the `Plotting` class, all Matplotlib rendering for time profiles, spectra, and spectrograms) and `interval_selector.py` (`IntervalSelector`, a small reusable "drag to pick a time range" widget). Both are used by `process/inputWindow.py` and `process/background.py`; `Plotting` is also reached into directly by `background.py::spectrogram_interval` (see [../background.md](../background.md)).

## `plotting.py::Plotting`

### Construction

`Plotting(start=None, end=None, hours=None, data=None, headers=None)` takes already-loaded `data`/`headers` dicts (as returned by `io/loader.py`), not a file path - the caller is responsible for loading. `entire_file` is `True` unless all three of `start`, `end`, and `hours` are given, in which case the instance will restrict plotting to that custom range instead of the whole file (see `inputWindow.py::_build_plot_instance`, which decides whether to pass these). The constructor also builds `self.del_times` via `delay_times()` (same "shift `timedel` by one, duplicate the first value" trick used independently in `background.py::_delay_times` - the two implementations are not shared, just parallel).

### Time handling

STIX FITS files store `time` as an arbitrary instrument index axis, not real timestamps - only `DATE_BEG`/`DATE_END` in the header are real dates. Every plotting path in this class therefore does the same linear-interpolation trick to get real UTC datetimes:

```
t_estimated = t_min + (delta_seconds_since_DATE_BEG / total_seconds) * (t_max - t_min)
```

or the inverse, mapping an index in `self.times` to a fraction of `[DATE_BEG, DATE_END]`. This appears as `date_to_times_index()` (date -> index) and inline in `__time_profile_plotting()` / `__plot_spectrogram()` (index -> datetime, vectorized over the whole time array). `find_time(date)` is a narrower helper that just extracts seconds-of-day from an ISO string (used to detect whether a requested custom range wraps past midnight, in `acq_time()`).

### 1. Time Profile Plotting

Entry points `rate_vs_time_plotting()` / `counts_vs_time_plotting()` / `flux_vs_time_plotting()` just set `self.type` and call `__time_profile_setting()`, which opens a secondary "STIX PlotTime Options" `Toplevel`. In that window the user picks 1-5 energy bands (`entries_list()` builds one `OptionMenu` pair per band via `open_value(i)`), optionally a log/linear scale for each axis (`log_axis()`, shared with the spectrum window - see below), and a "Show Information" checkbox that opens a small panel listing which channel indices each band resolved to (`sum_canal()`).

"Do Plot" (`__do_plot_rate` / `_counts` / `_flux`) calls `add_bands()` to snap the user's typed energy values to the nearest actual channel edge (`round_energy`, linear nearest-neighbor search, not vectorized), `__get_data(typ)` to build a `(n_times, n_bands + 1)` array in the requested unit (last column is `self.times`), and `__time_profile_plotting(data, typ)` to actually draw it: converts to real datetimes, optionally restricts to a custom sub-range (validated against the file's own range, error dialog if outside), and plots one line per band with the STIX standard 5-color cycle (`['blue', 'red', 'green', 'black', 'orange']`), an auto-scaling date axis, and optional log-scaled/gridded axes per the earlier settings.

### 2. Spectrum Plotting

`plot_spectrum_rate/counts/flux()` set `self.type` and call `__plot_spectrum()`, which time-averages every channel (over the custom range if one is set, via `acq_time()` + `self.index_start`/`index_end`, otherwise over the whole file) into a single 1D array, then opens `win_log_spec()` - a small popup to choose axis scales and toggle the grid - before actually rendering the step plot (`__plot_show()`, triggered by that popup's "Plot spectrum" button) of the chosen unit vs. `self.lower_bands`.

### 3. Spectrogram Plotting

`plot_spectrogram_rate/counts/flux(typ)` all funnel into `__plot_spectrogram(typ)`, which builds a `time x energy` heatmap with `plt.pcolormesh(..., cmap='coolwarm')` of `log10(value)`, restricted to a custom time range the same way as the time profile. Before drawing, `specgm_lim()` opens a popup to choose the y-axis (energy) limits and scale via `OptionMenu`s populated from the file's own channel edges; "Plot spectrogram" there calls `show_specgm()`, which applies the chosen `ylim`, attaches a colorbar formatted as powers of ten (`colorbar_scale`, a `FuncFormatter`), and finally shows the figure. `background.py::spectrogram_interval` reaches directly into the private `_Plotting__plot_spectrogram` method and monkey-patches `specgm_lim` to a no-op so it can skip this popup and attach its own `SpanSelector` instead (see [../background.md](../background.md)).

### Unit conversions

`convert_counts_rate()` (`counts / timedel`) and `convert_counts_flux()` (`counts / timedel / area / bandwidth`) are plain nested-loop implementations (not vectorized with NumPy broadcasting) used by both the spectrum and spectrogram code paths; `self.area = 6` cm^2 is hardcoded, matching the same constant duplicated in `background.py` and `fit_all.py`.

### Reused across scale-selection popups

`log_axis(window, relx, rely)` draws a small block of four `Radiobutton`s (x-axis linear/log, y-axis linear/log) at a given relative position and is called identically from the time-profile band-selection canvas, `win_log_spec()`, and `specgm_lim()` - it is the one piece of UI actually shared between the three plot types, via `self.scalex` / `self.scaley` `StringVar`s read back by whichever "Plot ..." button triggers the final render.

## `interval_selector.py::IntervalSelector`

A focused helper: given `xdata` (e.g. datetimes) and `ydata` (one or more series, optionally with `col_label`/`color`/`samefig`/`band` controlling whether all bands are drawn together or only one), it draws the data on a fresh Matplotlib figure with a date-aware x-axis, then `graphical_selection()` attaches a horizontal `matplotlib.widgets.SpanSelector` and calls the blocking `plt.show()`. Once the user drags a selection, `onselect` stores the start/end x-values and closes the figure, which unblocks `plt.show()`; `graphical_selection()` then converts the selection to `datetime` objects (`matplotlib.dates.num2date`, timezone stripped) and returns `(dt_start, dt_end)`, or `(None, None)` if the window was closed without making a selection. Used by `background.py::graphical_interval` as the graphical alternative to typing dates directly into the band's Start/End entries.
