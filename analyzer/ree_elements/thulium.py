"""Thulium Digital Twin — Tm (Z=69)
Sense-Model-Transmit loop for Thulium REE emissions.
Hardware proxy: X-ray emission
Primary use:    Portable X-ray sources, surgical lasers
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Tm",
    atomic_number=69,
    name="Thulium",
    category=REECategory.HREE,
    primary_spectral_line_nm=313.1,
    rf_signature_type=RFSignatureType.PULSED,
    emission_focus="X-ray emission",
    primary_use="Portable X-ray sources, surgical lasers",
    ore_minerals=['xenotime', 'monazite'],
    secondary_spectral_lines_nm=[303.5, 319.7],
    ionization_energy_ev=6.184,
    abundance_ppm_earth_crust=0.5,
    hazard_notes="",
    rf_carrier_hz=313.1,
    rf_harmonic_count=3,
)


class ThuliumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
