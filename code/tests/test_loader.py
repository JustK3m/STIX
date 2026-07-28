import numpy as np

from process.io import loader


def test_get_data_returns_expected_keys_and_shapes(spectrum_fits_path):
    data = loader.get_data(spectrum_fits_path)

    for key in ("time", "timedel", "counts", "counts_err", "e_low", "e_high"):
        assert key in data

    n_time = len(data["time"])
    n_energy = len(data["e_low"])

    assert len(data["timedel"]) == n_time
    assert data["counts"].shape == (n_time, n_energy)
    assert data["counts_err"].shape == (n_time, n_energy)
    assert len(data["e_high"]) == n_energy


def test_get_data_values_match_source_arrays(make_spectrum_fits):
    path = make_spectrum_fits(n_time=3, n_energy=2)
    data = loader.get_data(path)

    np.testing.assert_array_equal(data["time"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(data["counts"], [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(data["e_low"], [1.0, 2.0])
    np.testing.assert_array_equal(data["e_high"], [2.0, 3.0])


def test_get_header_contains_observation_dates(make_spectrum_fits):
    path = make_spectrum_fits(date_beg="2023-03-19T17:55:04.423",
                              date_end="2023-03-20T00:00:14.523")
    header = loader.get_header(path)

    assert header["DATE_BEG"] == "2023-03-19T17:55:04.423"
    assert header["DATE_END"] == "2023-03-20T00:00:14.523"


def test_get_data_is_cached_by_path(spectrum_fits_path):
    first = loader.get_data(spectrum_fits_path)
    second = loader.get_data(spectrum_fits_path)

    assert first is second
    assert loader.activeFile() == spectrum_fits_path


def test_get_data_reloads_when_path_changes(spectrum_fits_path, make_spectrum_fits):
    loader.get_data(spectrum_fits_path)
    assert loader.activeFile() == spectrum_fits_path

    other_path = make_spectrum_fits(n_time=2, n_energy=2)
    other = loader.get_data(other_path)

    assert loader.activeFile() == other_path
    assert len(other["time"]) == 2


def test_get_srm_data_returns_expected_shapes(srm_fits_path):
    srm = loader.get_srm_data(srm_fits_path)

    assert srm["MATRIX"].ndim == 2
    n_true = srm["MATRIX"].shape[0]

    assert srm["ENERG_LO"].shape == (n_true,)
    assert srm["ENERG_HI"].shape == (n_true,)
    assert np.all(srm["ENERG_HI"] >= srm["ENERG_LO"])
    assert loader.activeSRMfile() == srm_fits_path


def test_get_srm_data_is_cached_by_path(srm_fits_path):
    first = loader.get_srm_data(srm_fits_path)
    second = loader.get_srm_data(srm_fits_path)
    assert first is second
