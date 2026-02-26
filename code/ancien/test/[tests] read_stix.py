from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
#file = '/Users/ahamini/Nextcloud/Stage_STIX/stage_2025/stx_srm_2303197888.fits'
file = '/Users/ahamini/Nextcloud/Stage_STIX/stage_2025/STIX_2025/stx_srm_2021feb14_0140_0155.fits'

hdu_list = fits.open(file)
for i, hdu in enumerate(hdu_list):
    print(f"\n=== HDU {i} ===")

    # Affiche le header
    print(">> Header:")
    print(repr(hdu.header))

    # Affiche les données si elles existent
    print(">> Data:")
    if hdu.data is not None:
        print(hdu.data)
    else:
        print("Aucune donnée.")

header_file = hdu[0].header
#print(list(header_file.keys()))
header_image = hdu[1].header
header_time = hdu[2].header
header_energy = hdu[3].header
data_image = hdu[1].data
print(data_image.names)
data_time = hdu[2].data








