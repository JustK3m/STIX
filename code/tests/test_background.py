import datetime

import numpy as np
import pytest

from process.background import BackgroundWindow


@pytest.fixture
def bkg_window(tk_root):
    """A BackgroundWindow with no GUI built (show=False), for testing the
    pure data-processing methods in isolation from tkinter widgets.
    """
    return BackgroundWindow(show=False)


# ---------------------------------------------------------------------------
# parse_datetime_string / _delay_times
# ---------------------------------------------------------------------------

def test_parse_datetime_string_with_milliseconds():
    dt = BackgroundWindow.parse_datetime_string("2023-03-19T17:55:04.423")
    assert dt == datetime.datetime(2023, 3, 19, 17, 55, 4, 423000)


def test_parse_datetime_string_without_milliseconds():
    dt = BackgroundWindow.parse_datetime_string("2023-03-19T17:55:04")
    assert dt == datetime.datetime(2023, 3, 19, 17, 55, 4)


def test_parse_datetime_string_rejects_invalid_input():
    with pytest.raises(ValueError):
        BackgroundWindow.parse_datetime_string("not-a-date")


def test_delay_times_shifts_and_repeats_first_value():
    result = BackgroundWindow._delay_times(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(result, [1.0, 1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# round_value
# ---------------------------------------------------------------------------

def test_round_value_returns_index_of_nearest_element(bkg_window):
    idx = bkg_window.round_value([1.0, 2.0, 5.0, 10.0], 4.6)
    assert idx == 2
    assert bkg_window.rounded == 5


def test_round_value_ignores_nan_and_inf(bkg_window):
    idx = bkg_window.round_value([1.0, float("nan"), float("inf"), 100.0], 90.0)
    assert idx == 3


def test_round_value_raises_when_no_finite_values(bkg_window):
    with pytest.raises(ValueError):
        bkg_window.round_value([float("nan"), float("inf")], 1.0)


# ---------------------------------------------------------------------------
# _band_index
# ---------------------------------------------------------------------------

def test_band_index_exact_match(bkg_window):
    idx = bkg_window._band_index([1.0, 2.0, 3.0, 4.0], 3.0)
    assert idx == 2


def test_band_index_falls_back_to_nearest(bkg_window):
    idx = bkg_window._band_index([1.0, 2.0, 5.0, 10.0], 4.0)
    assert idx == 2  # nearest to 4.0 is 5.0


# ---------------------------------------------------------------------------
# date_to_times_index / convert_time_to_date (round trip)
# ---------------------------------------------------------------------------

def test_date_to_times_index_and_back(bkg_window):
    bkg_window.times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    bkg_window.start_date = "2023-03-19T17:55:04.423"
    bkg_window.end_date = "2023-03-19T17:56:44.423"  # 100s later

    # +40s of 100s total -> t_estimated = 0 + 0.4*40 = 16 -> nearest sample is t=20 (idx 2)
    idx = bkg_window.date_to_times_index("2023-03-19T17:55:44.423")
    assert idx == 2

    # t=20 is 50% of the [0, 40] range -> start_date + 50% of the 100s span
    date_str = bkg_window.convert_time_to_date(20.0)
    assert date_str == "2023-03-19T17:55:54.423"


def test_convert_time_to_date_clips_out_of_range_values(bkg_window):
    bkg_window.times = np.array([0.0, 10.0, 20.0])
    bkg_window.start_date = "2023-03-19T17:55:04.423"
    bkg_window.end_date = "2023-03-19T17:55:24.423"

    assert bkg_window.convert_time_to_date(-100.0) == bkg_window.convert_time_to_date(0.0)
    assert bkg_window.convert_time_to_date(1000.0) == bkg_window.convert_time_to_date(20.0)


# ---------------------------------------------------------------------------
# get_data (unit conversions)
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_bkg_window(bkg_window):
    bkg_window.lower_bands = np.array([1.0, 2.0, 3.0, 4.0])
    bkg_window.upper_bands = np.array([2.0, 3.0, 4.0, 5.0])
    bkg_window.energies_low = [1]
    bkg_window.energies_high = [3]
    bkg_window.times = np.array([0.0, 1.0, 2.0, 3.0])
    bkg_window.del_times = np.array([1.0, 1.0, 1.0, 1.0])
    bkg_window.counts = np.array(
        [[1, 2, 3, 4], [1, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0]], dtype=float)
    bkg_window.counts_err = np.ones_like(bkg_window.counts)
    return bkg_window


def test_get_data_counts_sums_selected_channel_band(populated_bkg_window):
    bw = populated_bkg_window
    bw.type = "Counts"
    bw.get_data()

    # band covers channels 0-1 (lower_bands==1 .. upper_bands==3)
    np.testing.assert_array_equal(bw.data[:, 0], [3.0, 2.0, 4.0, 0.0])
    np.testing.assert_array_equal(bw.data[:, -1], bw.times)  # last column is time


def test_get_data_rate_divides_by_timedel(populated_bkg_window):
    bw = populated_bkg_window
    bw.del_times = np.array([2.0, 2.0, 2.0, 2.0])
    bw.type = "Rate"
    bw.get_data()

    np.testing.assert_array_equal(bw.data[:, 0], [1.5, 1.0, 2.0, 0.0])


def test_get_data_flux_divides_by_area_and_energy_width(populated_bkg_window):
    bw = populated_bkg_window
    bw.area = 2.0
    bw.type = "Flux"
    bw.get_data()

    # e_diff = area * |e_high - e_low| = 2 * |3 - 1| = 4
    np.testing.assert_array_equal(bw.data[:, 0], np.array([3.0, 2.0, 4.0, 0.0]) / 4.0)


# ---------------------------------------------------------------------------
# get_bkg / get_data_bkg
# ---------------------------------------------------------------------------

def test_get_bkg_median_and_clamped_data_bkg(populated_bkg_window, tk_root):
    import tkinter as tk

    bw = populated_bkg_window
    bw.type = "Counts"
    bw.get_data()

    bw.method_var = [tk.StringVar(value="Median")]
    bw.method_list = list(bw.method_var)
    bw.var_sep_times = tk.IntVar(value=0)
    bw.bkg_start_index = [0]
    bw.bkg_end_index = [4]

    bw.get_bkg()
    np.testing.assert_array_equal(bw.bkg[:, 0], [2.5, 2.5, 2.5, 2.5])

    bw.get_data_bkg()
    np.testing.assert_array_equal(bw.data_bkg[:, 0], [0.5, 0.0, 1.5, 0.0])
    assert BackgroundWindow.DATA_BKG_RESULT is bw.data_bkg


def test_get_bkg_never_produces_negative_values(populated_bkg_window, tk_root):
    import tkinter as tk

    bw = populated_bkg_window
    bw.type = "Counts"
    bw.get_data()

    bw.method_var = [tk.StringVar(value="1Poly")]
    bw.method_list = list(bw.method_var)
    bw.var_sep_times = tk.IntVar(value=0)
    bw.bkg_start_index = [0]
    bw.bkg_end_index = [4]

    bw.get_bkg()
    assert np.all(bw.bkg >= 0)
