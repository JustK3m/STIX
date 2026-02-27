# STIX-Solar-Orbiter

# STIX Spectrum Fitting Tool in Python

This project provides tools to fit spectra measured by the STIX instrument (Solar Orbiter) using custom spectral models and the instrument response matrix (SRM). It features an interactive Tkinter-based interface for loading, modeling, and comparing data.

## 📦 Features

- Load STIX FITS files.
- Choose from various spectral models: PowerLaw1D, BrokenPowerLaw1D, V_TH, etc.
- Forward folding using the SRM matrix.
- Automatic model fitting.
- Interactive visualization of results (flux, rate, counts).
- Support for statistical error propagation.

## 🖥️ Interface

The Tkinter GUI allows you to:
- Load FITS files (spectrum and SRM).
- Select energy intervals.
- Add and configure spectral models.
- Perform fitting and display results.


## ⚙️ Installation

### Requirements

- Python 3.9+
- Astropy
- Numpy
- Scipy
- Matplotlib
- Tkinter

### Setup

    git clone https://github.com/Assamoi21/STIX-Solar-Orbiter.git
    cd STIX-Solar-Orbiter
    pip install -r requirements.txt

### Mac Installation 
If you encounter issues on macOS where Tkinter elements (buttons, windows, etc.) do not display correctly, follow these steps:

- python3 -m venv venv_stix
- source venv_stix/bin/activate    

- Install Python dependencies: pip install -r requirements.txt 

- Install Tcl/Tk and Python with Tkinter support:
brew install tcl-tk

brew install python-tk@3

-Call: 
python3 main.py 

## 📁 Data

Example FITS files (spectra and SRM) are included in the repository. You can also download them from the official Solar Orbiter data sources.

## 🔖 Build Executable (Windows)

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


## 📜 Licence

This project is licensed under the MIT License.

## 👨‍🔬 Authors

    Abdallah Hamini, Assamoua Koman

    Contact : abdallah.hamini@obspm.fr


