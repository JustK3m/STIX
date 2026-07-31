# STIX-Solar-Orbiter

# STIX Spectrum Fitting Tool in Python

This project provides tools to fit X-ray spectra measured by the STIX instrument on Solar Orbiter. It reads STIX FITS files (spectrum and instrument response matrix), lets the user define a background and an energy/time selection, and fits a spectral model to the data either by forward folding a physical model through the response matrix or by running a pre-trained neural network. Everything is driven through a Tkinter desktop GUI.

## How the application works

### Entry point and window structure

`code/src/main.py` builds the main Tkinter window (`root = Tk()`) and a File menu. Each menu item opens its own `Toplevel` window, implemented as a separate class under `code/src/process/`. There is no shared application state object; instead, each window loads data independently through `code/src/process/io/loader.py`, and a few pieces of state are shared between windows via module-level or class-level variables (see "Shared state" below).

The four windows, in the order a user typically works through them:

1. **Download STIX Data** (`process/downloader.py`, class `STIXDownloader`) - optional. Queries the STIX Data Center (via the `stixdcpy` package) for a UTC time range and downloads matching FITS files. If more than one file matches, they are merged into a single FITS file (overlapping time steps are timedel-weighted averaged, gaps are filled with NaN). The downloaded file is not auto-selected elsewhere; it has to be opened manually via "Browse" in the other windows.

2. **Select Input** (`process/inputWindow.py`, class `InputWindow`) - loads a spectrum FITS file and plots it. Plot type (Spectrum, Time Profile, Spectrogram) and unit (Rate, Counts, Flux) are chosen from dropdowns; a dispatch table (`_PLOT_DISPATCH`) maps the (unit, plot type) pair to a method on `process/graphics/plotting.py::Plotting`. "Summarize" and "Show Header" give quick views of the FITS metadata.

3. **Select Background** (`process/background.py`, class `BackgroundWindow`) - loads a spectrum file, lets the user define 1 to 5 energy bands and, for each band, a time interval outside the flare to use as background reference (entered as dates, or picked graphically from a time profile or spectrogram via `process/graphics/interval_selector.py::IntervalSelector`). The background level is estimated per band using one of: Median, Mean, 1Poly, 2Poly, 3Poly, or Exp (exponential fit). The computed background start/end indices and result are stored on `BackgroundWindow` class attributes (`DATA_BKG_SELECTED`, `DATA_BKG_START`, `DATA_BKG_END`, `DATA_BKG_RESULT`) so the fitting window can pick them up later.

4. **Plot Fit Results** (`process/fit_all.py`, class `Fitting`) - the main fitting window. Loads a spectrum FITS file and a response matrix (SRM) FITS file, then fits the data. See "Fitting pipeline" below.

`code/src/user_guide.py` opens an in-app, scrollable User Guide window (Help menu) with a step-by-step description of the same workflow and a summary of every spectral model - it is a good first read alongside this section, and should be kept in sync with the actual behavior when the UI changes.

### Data loading

`code/src/process/io/loader.py` is the single place that reads FITS files with Astropy. It exposes:

- `get_data(fpath)` / `get_header(fpath)` for a spectrum file (reads the `DATA` and `ENERGIES` HDUs: time, timedel, counts, counts errors, energy bin edges).
- `get_srm_data(rpath)` for a response matrix file (reads `MATRIX`, `ENERG_LO`, `ENERG_HI`).
- `activeFile()` / `activeSRMfile()` return the path of the last file loaded of each kind, which is how a window opened after another (e.g. Background after Select Input) can default to the same file.

Each of these functions caches the last file it read (`_spec_cache`, `_srm_cache`) so re-opening the same path across windows does not re-parse the FITS file.

### Fitting pipeline (`process/fit_all.py`)

The `Fitting` window supports two independent methods, chosen via "Set method":

- **Forward Folding** - fits one of nine physical models (defined in `process/fitting/methods/ForwardFolded.py`: PowerLaw1D, BrokenPowerLaw1D, an empirical power-law-times-exponential, V_TH thermal bremsstrahlung, and combinations/cutoff variants) to the observed counts. Each model's `evaluate()` integrates the photon flux over each true-energy bin and folds it through the SRM (`matrix`) to produce a modelled count rate, which is compared against the data. The user can pick the fit statistic:
  - Chi2 (`astropy.modeling.fitting.LevMarLSQFitter`, default), or
  - C-stat (`process/fitting/fitters/LevMarCstatFitter.py`, a Levenberg-Marquardt fitter that minimizes the Cash statistic via `scipy.optimize.least_squares`, more appropriate for low-count/Poisson data).
  Initial values and bounds per model/parameter can be edited from "Function value(s)" (`Set_Function`); defaults live in `Fitting.default_param_values` / `default_param_bounds`. `fit_unconstrained_then_bounded` and `fit_with_bounds_check` implement a two-step strategy: fit once, then re-fit with bounds applied only if the unconstrained/internal result falls outside the requested range.

- **CNN** - reconstructs the photon spectrum directly from the observed counts using a pre-trained neural network (`process/fitting/methods/NeuralNetwork.py::NeuralNetModel`), without an explicit forward-folding fit. No model, initial values, or fit statistic are needed; the SRM is given to the network as a conditioning input. The network is a two-branch MLP (counts branch + an SRM encoder that compresses the flattened response matrix) trained on simulated power-law spectra folded through randomized synthetic SRMs. A pre-trained checkpoint ships at `code/data/nn_powerlaw_150k.pt`; re-training is done by running `NeuralNetwork.py` directly (see the module docstring for `train()` parameters).

`_selective_fit` is the entry point for the "Do Fit" button. It: averages counts over the loaded time range, optionally subtracts the background computed in the Background window (`Data-Background` checkbox - if no background was computed yet, the user is prompted to open the Background window), masks the data to the selected energy range, dispatches to Forward Folding or CNN, converts the result to the selected display unit (Rate/Counts/Flux), and plots data + model with Matplotlib. The optional "Photon" plot shows the deconvolved photon spectrum in true-energy space.

### Shared state between windows

There is no central application/session object. Instead:

- `process/io/loader.py` caches the last-opened spectrum and SRM file paths and data, exposed via `activeFile()` / `activeSRMfile()`.
- `process.background.BackgroundWindow` stores the last computed background as class attributes (not instance attributes), so `Fitting` can read `DATA_BKG_SELECTED` / `DATA_BKG_START` / `DATA_BKG_END` / `DATA_BKG_RESULT` without holding a reference to the `BackgroundWindow` instance that computed them.

Keep this in mind when extending the app: a second `BackgroundWindow` instance overwrites the background used by every other window, and there is nothing that ties a background or fit result to a particular spectrum file beyond the user re-selecting the right one.

### Tests

`code/tests/` has pytest coverage for the loader, the background window, the forward-folded models, and the C-stat fitter (`code/tests/conftest.py` builds minimal synthetic FITS files so tests do not depend on the example data). Run them from `code/`:

    cd code
    pip install -r ../requirements.txt -r ../requirements-dev.txt
    pytest -v

CI (`.github/workflows/tests.yml`) runs the same suite on `windows-latest`, because `BackgroundWindow` instantiates real Tkinter `Variable`s even when built with `show=False`, which needs a desktop session that headless Linux runners do not have.

## Features

- Load STIX FITS files (spectrum and response matrix), and download new ones from the STIX Data Center.
- Choose from nine forward-folding spectral models, or reconstruct the spectrum with a pre-trained CNN.
- Fit using Chi2 or C-stat (Cash statistic).
- Interactive visualization of results (flux, rate, counts; spectrum, time profile, spectrogram).
- Background estimation (Median, Mean, polynomial, or exponential) with optional background subtraction before fitting.
- Support for statistical error propagation.

## Installation

### Requirements

- Python 3.9+
- See `requirements.txt` for the full dependency list (Astropy, NumPy, SciPy, Matplotlib, PyTorch, stixdcpy, Tkinter, etc.)

### Setup

    git clone https://github.com/Just_K3m/STIX.git
    cd STIX-Solar-Orbiter
    pip install -r requirements.txt

### Mac Installation
If you encounter issues on macOS where Tkinter elements (buttons, windows, etc.) do not display correctly, follow these steps:

- Install Python dependencies: pip install -r requirements.txt

- Install Tcl/Tk and Python with Tkinter support:
brew install tcl-tk

brew install python-tk@3

-Call:
python3 main.py

## Data

Example FITS files (spectrum and SRM) and a pre-trained CNN checkpoint are included under `code/data/`. You can also download more via the in-app "Download STIX Data" window, or from the official Solar Orbiter data sources.

## Running the tests

    cd code
    pip install -r ../requirements.txt -r ../requirements-dev.txt
    pytest -v

## Build Executable (Windows)

You can build a standalone .exe to run the application.

Steps:

1. Install PyInstaller:

    pip install pyinstaller

2. Build the executable:

From the root of the project, run:

    pyinstaller main.py --onefile --noconsole --add-data "data;data" --name "STIX Solar Orbitor" --hidden-import matplotlib.backends.backend_tkagg

This will:

- Create a single .exe file in the dist/ folder

- Bundle the data/ directory (which includes FITS files)

- Ensure the GUI (Tkinter and Matplotlib) works correctly

3. Run the application:

Navigate to dist/ and double-click STIX Solar Orbitor.exe.

## Licence

This project is licensed under the MIT License.

## Authors

    Abdallah Hamini, Kemil Bina

    Contact : abdallah.hamini@obspm.fr
