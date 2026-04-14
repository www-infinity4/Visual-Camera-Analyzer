"""Erbium Digital Twin — Er (Z=68)
Sense-Model-Transmit loop for Erbium REE emissions.
Hardware proxy: Fibre amplifier leakage
Primary use:    EDFA fibre optic amplifiers (1530 nm)
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Er",
    atomic_number=68,
    name="Erbium",
    category=REECategory.HREE,
    primary_spectral_line_nm=337.2,
    rf_signature_type=RFSignatureType.DECAY_PATTERN,
    emission_focus="Fibre amplifier leakage",
    primary_use="EDFA fibre optic amplifiers (1530 nm)",
    ore_minerals=['xenotime', 'monazite'],
    secondary_spectral_lines_nm=[326.2, 349.9],
    ionization_energy_ev=6.108,
    abundance_ppm_earth_crust=3.5,
    hazard_notes="",
    rf_carrier_hz=337.2,
    rf_harmonic_count=3,
)


class ErbiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
