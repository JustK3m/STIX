import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture(scope="session")
def tk_root():
    """A hidden default Tk root, needed to instantiate tkinter Variables
    (StringVar/IntVar) outside of a full GUI window.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available for Tkinter: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def make_spectrum_fits(tmp_path):
    """Build a minimal STIX L1A-shaped spectrum FITS file, mirroring the
    HDU layout process.io.loader expects: PRIMARY(0), CONTROL(1), DATA(2),
    ENERGIES(3).
    """

    counter = [0]

    def _make(n_time=5, n_energy=4, date_beg="2023-01-01T00:00:00.000",
              date_end="2023-01-01T00:00:05.000"):
        time = np.arange(n_time, dtype="f8")
        timedel = np.full(n_time, 1.0, dtype="f8")
        counts = np.arange(n_time * n_energy, dtype="f8").reshape(n_time, n_energy)
        counts_err = np.ones((n_time, n_energy), dtype="f8")
        triggers = np.zeros(n_time, dtype="i8")

        col_data = fits.ColDefs([
            fits.Column(name="time", array=time, format="D"),
            fits.Column(name="timedel", array=timedel, format="D"),
            fits.Column(name="counts", array=counts, format=f"{n_energy}D"),
            fits.Column(name="counts_comp_err", array=counts_err, format=f"{n_energy}D"),
            fits.Column(name="triggers", array=triggers, format="K"),
        ])
        hdu_data = fits.BinTableHDU.from_columns(col_data, name="DATA")

        e_low = np.arange(1.0, n_energy + 1.0)
        e_high = e_low + 1.0
        col_energies = fits.ColDefs([
            fits.Column(name="e_low", array=e_low, format="D"),
            fits.Column(name="e_high", array=e_high, format="D"),
        ])
        hdu_energies = fits.BinTableHDU.from_columns(col_energies, name="ENERGIES")

        primary = fits.PrimaryHDU()
        primary.header["DATE_BEG"] = date_beg
        primary.header["DATE_END"] = date_end

        hdu_control = fits.BinTableHDU.from_columns(
            fits.ColDefs([fits.Column(name="x", array=np.zeros(1), format="D")]),
            name="CONTROL")

        hdul = fits.HDUList([primary, hdu_control, hdu_data, hdu_energies])
        counter[0] += 1
        path = tmp_path / f"spectrum_{counter[0]}.fits"
        hdul.writeto(path)
        return str(path)

    return _make


@pytest.fixture
def spectrum_fits_path(make_spectrum_fits):
    return make_spectrum_fits()


@pytest.fixture
def make_srm_fits(tmp_path):
    """Build a minimal STIX SRM FITS file: a 'MATRIX' HDU with ENERG_LO/
    ENERG_HI/MATRIX columns, mirroring what process.io.loader expects.
    """

    counter = [0]

    def _make(n_true=6, n_det=4):
        matrix = np.arange(n_true * n_det, dtype="f8").reshape(n_true, n_det)
        energ_lo = np.arange(n_true, dtype="f8")
        energ_hi = energ_lo + 1.0

        col_matrix = fits.ColDefs([
            fits.Column(name="ENERG_LO", array=energ_lo, format="D"),
            fits.Column(name="ENERG_HI", array=energ_hi, format="D"),
            fits.Column(name="MATRIX", array=matrix, format=f"{n_det}D"),
        ])
        hdu_matrix = fits.BinTableHDU.from_columns(col_matrix, name="SPECRESP MATRIX")

        e_min = np.arange(n_det, dtype="f4")
        e_max = e_min + 1.0
        col_ebounds = fits.ColDefs([
            fits.Column(name="E_MIN", array=e_min, format="E"),
            fits.Column(name="E_MAX", array=e_max, format="E"),
        ])
        hdu_ebounds = fits.BinTableHDU.from_columns(col_ebounds, name="EBOUNDS")

        hdul = fits.HDUList([fits.PrimaryHDU(), hdu_matrix, hdu_ebounds])
        counter[0] += 1
        path = tmp_path / f"srm_{counter[0]}.fits"
        hdul.writeto(path)
        return str(path)

    return _make


@pytest.fixture
def srm_fits_path(make_srm_fits):
    return make_srm_fits()
