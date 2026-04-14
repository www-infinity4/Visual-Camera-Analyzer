"""Dysprosium Digital Twin — Dy (Z=66)
Sense-Model-Transmit loop for Dysprosium REE emissions.
Hardware proxy: Magnetic field drift
Primary use:    EV motor magnets, hard disk drives
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Dy",
    atomic_number=66,
    name="Dysprosium",
    category=REECategory.HREE,
    primary_spectral_line_nm=353.1,
    rf_signature_type=RFSignatureType.OSCILLATORY,
    emission_focus="Magnetic field drift",
    primary_use="EV motor magnets, hard disk drives",
    ore_minerals=['xenotime', 'ion-adsorption clay'],
    secondary_spectral_lines_nm=[345.5, 394.5],
    ionization_energy_ev=5.939,
    abundance_ppm_earth_crust=5.2,
    hazard_notes="",
    rf_carrier_hz=353.1,
    rf_harmonic_count=3,
)


class DysprosiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
