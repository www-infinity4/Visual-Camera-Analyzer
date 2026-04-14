"""Europium Digital Twin — Eu (Z=63)
Sense-Model-Transmit loop for Europium REE emissions.
Hardware proxy: Luminescence output
Primary use:    Red/blue phosphors in LEDs and screens
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Eu",
    atomic_number=63,
    name="Europium",
    category=REECategory.LREE,
    primary_spectral_line_nm=459.4,
    rf_signature_type=RFSignatureType.WIDEBAND_NOISE,
    emission_focus="Luminescence output",
    primary_use="Red/blue phosphors in LEDs and screens",
    ore_minerals=['bastnäsite', 'monazite'],
    secondary_spectral_lines_nm=[412.9, 443.1],
    ionization_energy_ev=5.67,
    abundance_ppm_earth_crust=2.0,
    hazard_notes="",
    rf_carrier_hz=459.4,
    rf_harmonic_count=3,
)


class EuropiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
