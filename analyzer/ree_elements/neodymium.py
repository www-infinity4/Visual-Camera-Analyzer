"""Neodymium Digital Twin — Nd (Z=60)
Sense-Model-Transmit loop for Neodymium REE emissions.
Hardware proxy: Mining dust density
Primary use:    High-strength magnets (EV, wind turbines)
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Nd",
    atomic_number=60,
    name="Neodymium",
    category=REECategory.LREE,
    primary_spectral_line_nm=430.3,
    rf_signature_type=RFSignatureType.STEADY_STATE,
    emission_focus="Mining dust density",
    primary_use="High-strength magnets (EV, wind turbines)",
    ore_minerals=['bastnäsite', 'monazite'],
    secondary_spectral_lines_nm=[521.8, 574.5],
    ionization_energy_ev=5.525,
    abundance_ppm_earth_crust=33.0,
    hazard_notes="",
    rf_carrier_hz=430.3,
    rf_harmonic_count=3,
)


class NeodymiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
