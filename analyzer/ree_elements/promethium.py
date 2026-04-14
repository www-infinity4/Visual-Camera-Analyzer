"""Promethium Digital Twin — Pm (Z=61)
Sense-Model-Transmit loop for Promethium REE emissions.
Hardware proxy: Radioactive decay
Primary use:    Radioactive tracers (medical)
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Pm",
    atomic_number=61,
    name="Promethium",
    category=REECategory.LREE,
    primary_spectral_line_nm=463.0,
    rf_signature_type=RFSignatureType.DECAY_PATTERN,
    emission_focus="Radioactive decay",
    primary_use="Radioactive tracers (medical)",
    ore_minerals=['fission product'],
    secondary_spectral_lines_nm=[464.0, 474.0],
    ionization_energy_ev=5.582,
    abundance_ppm_earth_crust=0.0,
    hazard_notes="Radioactive — all isotopes unstable",
    rf_carrier_hz=463.0,
    rf_harmonic_count=3,
)


class PromethiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
