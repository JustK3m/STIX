import numpy as np
from astropy.io import fits

_spec_cache = {
    "fpath": None,  # path of the spectrum file currently cached
    "data": None,  # dict returned by load_data()
    "headers": None,  # dict returned by load_header()
}

_srm_cache = {
    "rpath": None,  # path of the SRM file currently cached
    "data": None,  # dict returned by load_srm_data()
}


def _load_data(hdulist):
    result = {}

    for hdu in hdulist:
        if not hasattr(hdu, 'columns'):
            continue
        colnames = hdu.columns.names

        if hdu.name == 'DATA':
            if 'time' in colnames:
                result['time']    = hdu.data['time']
            if 'timedel' in colnames:
                result['timedel'] = hdu.data['timedel']
            if 'counts' in colnames:
                result['counts']  = hdu.data['counts']
            if 'counts_comp_err' in colnames or 'counts_err' in colnames:
                err_col = 'counts_comp_err' if 'counts_comp_err' in colnames else 'counts_err'
                result['counts_err'] = hdu.data[err_col]
            if 'triggers' in colnames:
                result['triggers'] = hdu.data['triggers']
            if 'obt_start' in colnames and 'obt_end' in colnames:
                result['obt_start'] = hdu.data['obt_start']
                result['obt_end']   = hdu.data['obt_end']
            # e_low/e_high in DATA (original files)
            if 'e_low' in colnames and 'e_high' in colnames:
                result['e_low']  = hdu.data['e_low']
                result['e_high'] = hdu.data['e_high']

        elif hdu.name == 'ENERGIES':
            # e_low/e_high in ENERGIES (merged files)
            # Takes priority over DATA if both exist
            if 'e_low' in colnames and 'e_high' in colnames:
                result['e_low']  = hdu.data['e_low']
                result['e_high'] = hdu.data['e_high']

    for key in ['counts', 'counts_err', 'e_low', 'e_high', 'time', 'timedel']:
        if key not in result:
            print(f"⚠️  Warning: {key} not found in the FITS file.")

    return result


def _load_srm_data(hdulist):
    """
    Reads a STIX instrument response matrix (SRM) FITS file.

    Parameters
    ----------
    hdulist : HDUList
        Already-opened SRM FITS HDU list.

    Returns
    -------
    dict with keys:
        'MATRIX'   : ndarray (N, M) — response matrix.
        'ENERG_LO' : ndarray (N,)   — lower bounds in true energy (keV).
        'ENERG_HI' : ndarray (N,)   — upper bounds in true energy (keV).

    Notes
    -----
    A warning is printed for any required key missing from the FITS file.
    """

    result = {}

    for hdu in hdulist:
        if not hasattr(hdu, 'columns'):
            continue
        colnames = hdu.columns.names

        # Matrix
        if 'MATRIX' in colnames:
            result['MATRIX'] = hdu.data['MATRIX']

        # Energy bins
        if 'ENERG_LO' in colnames and 'ENERG_HI' in colnames:
            result['ENERG_LO'] = hdu.data['ENERG_LO']
            result['ENERG_HI'] = hdu.data['ENERG_HI']

    # Basic checks
    required_keys = ['MATRIX', 'ENERG_LO', 'ENERG_HI']
    for key in required_keys:
        if key not in result:
            print(f"⚠️  Warning: {key} not found in the FITS file.")
    return result


def _load_header(hdulist):
    result = {}

    for key, value, comment in hdulist[0].header.cards:
        result[key] = value

    for key, value, comment in hdulist[3].header.cards:
        result[key] = value

    return result


def _reload_spec(fpath):
    """Opens the file once and populates data + headers."""
    with fits.open(fpath) as hdulist:
        _spec_cache["data"] = _load_data(hdulist)
        _spec_cache["headers"] = _load_header(hdulist)
    _spec_cache["fpath"] = fpath


def get_data(fpath):
    if fpath != _spec_cache["fpath"] or _spec_cache["data"] is None:
        _reload_spec(fpath)
    return _spec_cache["data"]


def get_header(fpath):
    if fpath != _spec_cache["fpath"] or _spec_cache["headers"] is None:
        _reload_spec(fpath)
    return _spec_cache["headers"]


def get_srm_data(rpath):
    if rpath != _srm_cache["rpath"] or _srm_cache["data"] is None:
        with fits.open(rpath) as hdulist:
            _srm_cache["data"] = _load_srm_data(hdulist)
        _srm_cache["rpath"] = rpath
    return _srm_cache["data"]

def activeFile():
    return _spec_cache["fpath"]

def activeSRMfile():
    return _srm_cache["rpath"]

def concat_data(hdu1, hdu2):
    return hdu1.data + hdu2.data
