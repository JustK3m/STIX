# process package

This package contains the four GUI windows launched from the File menu in `code/src/main.py`, plus two support subpackages they all depend on.

## Windows (one module each)

Each of these opens its own `Toplevel` window and is otherwise independent: there is no shared session object, no controller class, and no event bus between them. A window is free to be opened, closed, and reopened at any time from the main window's File menu.

## Support subpackages

- `io/` - FITS reading and per-file caching, used by all four windows. See [io/README.md](io/README.md).
- `graphics/` - Matplotlib plotting used by `inputWindow.py` and `background.py`. See [graphics/README.md](graphics/README.md).
- `fitting/` - spectral models, the neural network, and the C-stat fitter used by `fit_all.py`. See [fitting/README.md](fitting/README.md).

## How the windows share state without a session object

Since there is no central application object, three separate mechanisms carry state between windows:

1. **Last-opened file** - `io/loader.py` caches the most recently loaded spectrum file and SRM file, and exposes them via `activeFile()` / `activeSRMfile()`. Any window's constructor can call these to default its file-picker to whatever was opened elsewhere, without holding a reference to the window that opened it. This is why opening Select Input, then Select Background, pre-fills the second window with the first window's file.

2. **Background result** - `background.py`'s `BackgroundWindow` stores the last computed background as **class attributes** (`DATA_BKG_SELECTED`, `DATA_BKG_START`, `DATA_BKG_END`, `DATA_BKG_RESULT`), not instance attributes. `fit_all.py` reads these class attributes directly (`background.BackgroundWindow.DATA_BKG_START`) to subtract the background when "Data-Background" is checked, without needing a reference to the `BackgroundWindow` instance that computed them.

3. **FITS caches** - `io/loader.py` also caches the parsed contents (`_spec_cache`, `_srm_cache`), so re-opening the same path from a different window does not re-parse the file with Astropy.

The practical consequence for anyone extending this code: opening a second `BackgroundWindow` and computing a new background overwrites the one used by `fit_all.py`, even for a completely different spectrum file. Nothing ties a computed background, or a fit result, to a specific spectrum file beyond the user manually keeping track of which file they last chose in each window.

## Tests

`code/tests/` has pytest coverage for `io/loader.py`, `background.py`, `fitting/methods/ForwardFolded.py`, and `fitting/fitters/LevMarCstatFitter.py`. There is currently no test coverage for `downloader.py`, `inputWindow.py`, `fit_all.py`'s `_selective_fit` orchestration, `fitting/methods/NeuralNetwork.py`, or `graphics/`. Run the suite from `code/`:

    cd code
    pip install -r ../requirements.txt -r ../requirements-dev.txt
    pytest -v
