from astropy.io import fits

_spec_cache = {
    "fname": None,  # chemin du fichier spectre actuellement en cache
    "data": None,  # dict retourné par extract_stix_data()
    "headers": None,  # dict retourné par extract_stix_header()
}

_srm_cache = {
    "rname": None,  # chemin du fichier SRM actuellement en cache
    "data": None,  # dict retourné par load_srm_data()
}


def load_data(hdulist):
    result = {}

    for hdu in hdulist:
        if not hasattr(hdu, 'columns'):
            continue
        colnames = hdu.columns.names

        # Time & Timedel
        if 'time' in colnames and 'timedel' in colnames:
            result['time'] = hdu.data['time']
            result['timedel'] = hdu.data['timedel']

        # Counts
        if 'counts' in colnames:
            result['counts'] = hdu.data['counts']
        if 'counts_comp_err' in colnames or 'counts_err' in colnames:
            err_col = 'counts_comp_err' if 'counts_comp_err' in colnames else 'counts_err'
            result['counts_err'] = hdu.data[err_col]

        # Triggers (optionnel)
        if 'triggers' in colnames:
            result['triggers'] = hdu.data['triggers']

        # Energy bins
        if 'e_low' in colnames and 'e_high' in colnames:
            result['e_low'] = hdu.data['e_low']
            result['e_high'] = hdu.data['e_high']

        # Version info
        if 'obt_start' in colnames and 'obt_end' in colnames:
            result['obt_start'] = hdu.data['obt_start']
            result['obt_end'] = hdu.data['obt_end']

    # Vérifications de base
    required_keys = ['counts', 'counts_err', 'e_low', 'e_high', 'time', 'timedel']
    for key in required_keys:
        if key not in result:
            print(f"⚠️  Attention : {key} non trouvé dans le FITS.")
    return result


def load_srm_data(hdulist):
    """
    Lit un fichier FITS de matrice de réponse instrumentale STIX.

    Parameters
    ----------
    hdulist : HDUList
        Chemin complet du fichier FITS SRM.

    Returns
    -------
    dict avec les clés :
        'MATRIX'   : ndarray (N, M) — matrice de réponse.
        'ENERG_LO' : ndarray (N,)   — bornes inférieures en énergie vraie (keV).
        'ENERG_HI' : ndarray (N,)   — bornes supérieures en énergie vraie (keV).

    Notes
    -----
    Un avertissement est affiché pour toute clé requise absente du FITS.
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

    # Vérifications de base
    required_keys = ['MATRIX', 'ENERG_LO', 'ENERG_HI']
    for key in required_keys:
        if key not in result:
            print(f"⚠️  Attention : {key} non trouvé dans le FITS.")
    return result


def load_header(hdulist):
    result = {}

    for key, value, comment in hdulist[0].header.cards:
        result[key] = value

    for key, value, comment in hdulist[3].header.cards:
        result[key] = value

    return result


def _reload_spec(fname):
    """Ouvre le fichier une seule fois et remplit data + headers."""
    with fits.open(fname) as hdulist:
        _spec_cache["data"] = load_data(hdulist)
        _spec_cache["headers"] = load_header(hdulist)
    _spec_cache["fname"] = fname


def get_data(fname):
    if fname != _spec_cache["fname"] or _spec_cache["data"] is None:
        _reload_spec(fname)
    return _spec_cache["data"]


def get_header(fname):
    if fname != _spec_cache["fname"] or _spec_cache["headers"] is None:
        _reload_spec(fname)
    return _spec_cache["headers"]


def get_srm_data(rname):
    if rname != _srm_cache["rname"] or _srm_cache["data"] is None:
        with fits.open(rname) as hdulist:
            _srm_cache["data"] = load_srm_data(hdulist)
        _srm_cache["rname"] = rname
    return _srm_cache["data"]

def activeFile():
    return _spec_cache["fname"]

def activeSRMfile():
    return _srm_cache["rname"]
