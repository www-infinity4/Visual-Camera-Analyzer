"""Scandium Digital Twin — Sc (Z=21)
Sense-Model-Transmit loop for Scandium REE emissions.
Hardware proxy: Alloy dust emissions
Primary use:    Al-Sc lightweight alloys (aerospace)
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Sc",
    atomic_number=21,
    name="Scandium",
    category=REECategory.LREE,
    primary_spectral_line_nm=361.3,
    rf_signature_type=RFSignatureType.OSCILLATORY,
    emission_focus="Alloy dust emissions",
    primary_use="Al-Sc lightweight alloys (aerospace)",
    ore_minerals=['wolframite', 'tin slags'],
    secondary_spectral_lines_nm=[335.4, 363.1],
    ionization_energy_ev=6.561,
    abundance_ppm_earth_crust=22.0,
    hazard_notes="",
    rf_carrier_hz=361.3,
    rf_harmonic_count=3,
)


class ScandiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
