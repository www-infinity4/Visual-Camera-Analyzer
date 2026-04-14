"""Praseodymium Digital Twin — Pr (Z=59)
Sense-Model-Transmit loop for Praseodymium REE emissions.
Hardware proxy: Mining dust
Primary use:    NdPr alloy magnets, pigments
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Pr",
    atomic_number=59,
    name="Praseodymium",
    category=REECategory.LREE,
    primary_spectral_line_nm=440.8,
    rf_signature_type=RFSignatureType.PULSED,
    emission_focus="Mining dust",
    primary_use="NdPr alloy magnets, pigments",
    ore_minerals=['bastnäsite', 'monazite'],
    secondary_spectral_lines_nm=[444.0, 469.1],
    ionization_energy_ev=5.464,
    abundance_ppm_earth_crust=9.2,
    hazard_notes="",
    rf_carrier_hz=440.8,
    rf_harmonic_count=3,
)


class PraseodymiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
