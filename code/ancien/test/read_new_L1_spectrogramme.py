"""
This module provides a STIXFileReader class to parse and extract structured information from a STIX FITS file
(such as spectrograms).
"""

from astropy.io import fits
import numpy as np
from datetime import datetime


class STIXFileReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.hdu = fits.open(filepath)

        self._parse_headers()
        self._parse_data()

    def _parse_headers(self):
        self.header_file = self.hdu[0].header
        self.date_obs_str = self.header_file.get('DATE-OBS')
        self.date_end_str = self.header_file.get('DATE-END')
        self.earth_solar_time_offset = self.header_file.get('EAR_TDEL')

        self.header_image = self.hdu[1].header
        self.header_time = self.hdu[2].header
        self.header_energy = self.hdu[3].header

    def _parse_data(self):
        self.data_image = self.hdu[1].data
        self.data_time = self.hdu[2].data
        self.data_energy = self.hdu[4].data

        self.time_delays = self.data_time['timedel']
        self.time = self.data_time['time']
        self.counts = self.data_time['counts']
        self.counts_error = self.data_time['counts_comp_err']

        self.obt_start = self.hdu[3].data['obt_start']
        self.obt_end = self.hdu[3].data['obt_end']

        self.e_low = self.data_energy['e_low']
        self.e_high = self.data_energy['e_high']

    def get_observation_dates(self):
        return self.date_obs_str, self.date_end_str

    def get_energy_bounds(self):
        return self.e_low, self.e_high

    def get_counts_info(self):
        return self.time, self.counts, self.counts_error

    def print_image_data_fields(self):
        print(self.data_image.names)

    def get_corrected_time(self):
        """
        Retourne le temps corrigé par le décalage Terre-Soleil (EAR_TDEL).
        """
        if self.earth_solar_time_offset is None:
            raise ValueError("Décalage EAR_TDEL non trouvé dans le header principal.")

        corrected_time = self.time + self.earth_solar_time_offset
        return corrected_time

    def get_time_energy_counts(self, corrected=False):
        """
        Retourne les tableaux : temps, énergies basses/hautes, et comptages.
        Si corrected=True, applique le décalage Terre-Soleil (EAR_TDEL) au temps.
        """
        self.time = self.get_corrected_time() if corrected else self.time

        return {
            "time": self.time,
            "e_low": self.e_low,
            "e_high": self.e_high,
            "counts": self.counts,
            "counts_error": self.counts_error,
            "earth_solar_time_offset": self.earth_solar_time_offset
        }

    def close(self):
        self.hdu.close()


#test
file = '/Users/ahamini/Nextcloud/Stage_STIX/stage_2025/new_data/solo_L1_stix-sci-xray-spec_20230319T175504-20230320T000014_V02_2303197888-65462.fits'
reader = STIXFileReader(file)
# Obtenir les dates d'observation
start_date, end_date = reader.get_observation_dates()
print(f"Observation start: {start_date}, end: {end_date}")

#data = reader.get_time_energy_counts(corrected=False)
data = reader.get_time_energy_counts(corrected=True)

print("Temps :", data["time"])
print("Énergies basses :", data["e_low"])
print("Énergies hautes :", data["e_high"])
print("Comptages :", data["counts"])
print("Erreurs sur les comptages :", data["counts_error"])
print("earth_solar_time_offset :" , data["earth_solar_time_offset"])
reader.close()