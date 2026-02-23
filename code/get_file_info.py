"""
The following code is used to make SRM/counts data in consistent units from STIX spectral data.

Credits to Shane Maloney, Kris Cooper & Dan Ryan.
"""

from astropy.io import fits
import numpy as np

__all__ = ["read_stix_file", "get_srm"]


def read_stix_file(file):
    """
    Read STIX SRM (or spectrum) fits file and extract useful information from it.

    Parameters
    ----------
    file : `str`, `file-like` or `pathlib.Path`
        A STIX SRM (or spectrum) fits file (see `~astropy.fits.io.open` for details)

    Returns
    -------
    `dict`
        STIX SRM (or spectrum) data
    """
    sdict = {}
    with fits.open(file) as hdul:
        for i in range(4):
            sdict[str(i)] = [hdul[i].header, hdul[i].data]
    return sdict


def get_srm(srm_file):
    """ Return all STIX SRM data needed for fitting.
    SRM units returned as counts ph^(-1) cm^(2).

    Parameters
    ----------
    srm_file : str
            String for the STIX SRM spectral file under investigation.

    Returns
    -------
    photon_bins : list
            List containing photons data for each energy band. Rough size usually around 1020.

    channel_bins : list
            List containing all min values for energy channels. Rough size usually around 20 or 32.

    srm : array
            Array containing the SRM, the multiplication matrix used to convert photons into counts.

    energy_bands : int
            Integer value of the total number of energy bands available.
    """
    srmfrdict = read_stix_file(srm_file)

    if srmfrdict["1"][0]["SUMFLAG"] != 1:
        print("Apparently srm file\'s `SUMFLAG` should be one and I don\'t know what to do otherwise at the moment.")
        return

    energy_bands = srmfrdict["1"][1][0][4]  # number of energy bands, usually around 20 to 30

    photon_channels_elo = srmfrdict["1"][1]['ENERG_LO']  # photon channel edges, different to count channels
    photon_channels_ehi = srmfrdict["1"][1]['ENERG_HI']
    photon_bins = np.concatenate((np.array(photon_channels_elo)[:, None], np.array(photon_channels_ehi)[:, None]),
                                 axis=1)

    srm = srmfrdict["1"][1]['MATRIX']  # counts ph^-1 keV^-1
    geo_area = srmfrdict["3"][0]['GEOAREA']

    channels = srmfrdict["2"][1]  # [(chan, lowE, hiE), ...], srmfrdict["2"][0] has units etc. count channels for SRM
    channel_bins = np.concatenate((np.array(channels['E_MIN'])[:, None], np.array(channels['E_MAX'])[:, None]), axis=1)

    # srm units counts ph^(-1) kev^(-1); i.e., photons cm^(-2) go in and counts cm^(-2) kev^(-1) comes out
    # https://hesperia.gsfc.nasa.gov/ssw/hessi/doc/params/hsi_params_srm.htm#***
    # need srm units are counts ph^(-1) cm^(2)
    srm = srm * np.diff(channel_bins, axis=1).flatten() * geo_area

    return photon_bins, channel_bins, srm, energy_bands


if __name__ == '__main__':

    test_photon_bins, test_channel_bins, test_srm, test_energy_bands = get_srm('./stx_srm_2021feb14_0140_0155.fits')

    print("Photons bins:\n", test_photon_bins, '\n')
    print("Channel bins:\n", test_channel_bins, '\n')
    print("SRM size: ", np.size(test_srm, 0), np.size(test_srm, 1), '\n')
    print("Number of energy bands: ", test_energy_bands)
