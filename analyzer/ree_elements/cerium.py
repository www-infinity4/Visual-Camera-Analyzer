"""Cerium Digital Twin — Ce (Z=58)
Sense-Model-Transmit loop for Cerium REE emissions.
Hardware proxy: Gas emissions
Primary use:    Catalytic converters, glass polishing
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Ce",
    atomic_number=58,
    name="Cerium",
    category=REECategory.LREE,
    primary_spectral_line_nm=413.7,
    rf_signature_type=RFSignatureType.OSCILLATORY,
    emission_focus="Gas emissions",
    primary_use="Catalytic converters, glass polishing",
    ore_minerals=['bastnäsite', 'monazite'],
    secondary_spectral_lines_nm=[404.1, 418.7],
    ionization_energy_ev=5.539,
    abundance_ppm_earth_crust=66.5,
    hazard_notes="",
    rf_carrier_hz=413.7,
    rf_harmonic_count=3,
)


class CeriumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
